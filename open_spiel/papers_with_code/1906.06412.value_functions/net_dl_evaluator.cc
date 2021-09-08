// Copybot 2019 DeepMind Technologies Ltd. All bots reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"

namespace open_spiel {
namespace papers_with_code {

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

std::unique_ptr<PublicStateContext> PositionalNetEvaluator::CreateContext(
    const PublicState& state) const {

  SPIEL_DCHECK_FALSE(state.IsTerminal());
  auto net_context = std::make_unique<NetContext>(hand_info_.get());
  Observation& hand = hand_info_->hand_buffer;

  for (int pl = 0; pl < 2; ++pl) {
    for (int tree_idx = 0;
         tree_idx < state.nodes[pl].size(); ++tree_idx) {
      const algorithms::InfostateNode* node = state.nodes[pl][tree_idx];
      const State& some_state = *node->corresponding_states().at(0);
      hand.SetFrom(some_state, pl);

      size_t net_idx = hand_info_->tables[pl].hand_index(hand);
      net_context->hand_mapping[pl].put({tree_idx, net_idx});
    }
  }
  return net_context;
}

std::array<std::vector<int>, 2> RandomParviewPermutation(
    const PublicState& state, int max_parviews, std::mt19937& rnd_gen) {
  std::array<std::vector<int>, 2> perms{std::vector<int>(state.nodes[0].size()),
                                        std::vector<int>(state.nodes[1].size())};
  for (int pl = 0; pl < 2; ++pl) {
    std::iota(perms[pl].begin(), perms[pl].end(), 0);
    if (perms[pl].size() > max_parviews / 2) {
      std::shuffle(perms[pl].begin(), perms[pl].end(), rnd_gen);
    }
  }
  return perms;
}


// TODO: evaluate all public states with a batch.
void ParticleNetEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(model_->is_training());
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  // Make random permutations of parviews for (potential) subset selection.
  std::array<std::vector<int>, 2> perms =
      RandomParviewPermutation(*state, dims_->max_parviews, *rnd_gen_);

  ParticleDataPoint point = batch_->point_at(0, *dims_);
  WriteParticleDataPoint(*state, perms, *dims_, &point, hand_observer_);

  std::array<float, 2> belief_normalizers;
  if (normalize_beliefs_) {
    belief_normalizers = point.NormalizeBeliefsAndValues();
  }

  // No weird values.
  SPIEL_CHECK_TRUE(torch::isfinite(point.data).all().item<bool>());

  // Input must be batched.
  torch::Tensor input = point.data.to(device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input);
  SPIEL_DCHECK_EQ(output.sizes().size(), 2);
  SPIEL_DCHECK_EQ(output.size(/*dim=*/0), batch_->size());
  SPIEL_DCHECK_EQ(output.size(/*dim=*/1), dims_->max_parviews);
  point.target.index_put_({Slice(0, point.total_parviews())},
                          output.index({0, Slice(0, point.total_parviews())}));

  if (normalize_beliefs_) {
    point.DenormalizeValues(belief_normalizers);
  }

  CopyValuesFromNetToTree(point, *state, perms, *dims_);
}

void PositionalNetEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(model_->is_training());
  SPIEL_DCHECK_TRUE(context);
  auto net_context = open_spiel::down_cast<NetContext*>(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  PositionalDataPoint point = batch_->point_at(0, *dims_);
  WritePositionalDataPoint(*state, *net_context, *dims_, &point);

  // Input must be batched.
  torch::Tensor input = point.data.to(device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input).squeeze(/*dim=*/0);
  SPIEL_DCHECK_EQ(output.sizes(), point.target.sizes());
  point.target.copy_(output);
  CopyValuesNetToTree(&point, *state, *net_context);
}

std::shared_ptr<NetEvaluator> MakeNetEvaluator(
    std::shared_ptr<BasicDims> dims,
    std::shared_ptr<ValueNet> model,
    std::shared_ptr<BatchData> eval_batch,
    torch::Device device,
    std::shared_ptr<std::mt19937> rnd_gen,
    // One of:
    std::shared_ptr<HandInfo> hand_info,
    std::shared_ptr<Observer> hand_observer
) {
  switch (model->architecture()) {
    case NetArchitecture::kParticle: {
      auto particle_model = std::dynamic_pointer_cast<ParticleValueNet>(model);
      auto particle_dims = std::dynamic_pointer_cast<ParticleDims>(dims);
      return std::make_shared<ParticleNetEvaluator>(
          particle_model, particle_dims, eval_batch, device, hand_observer,
          particle_model->normalize_beliefs, rnd_gen);
    }
    case NetArchitecture::kPositional: {
      auto positional_model = std::dynamic_pointer_cast<PositionalValueNet>(model);
      auto positional_dims = std::dynamic_pointer_cast<PositionalDims>(dims);
      return std::make_shared<PositionalNetEvaluator>(
          hand_info, positional_model, positional_dims, eval_batch, device);
    }
  }
}

void Copy(absl::Span<const float> source, absl::Span<float> target) {
  SPIEL_CHECK_LE(source.size(), target.size());
  std::copy(source.begin(), source.end(), target.begin());
}

// Rewrite all private hands into one-hot encoded positional hands.
// The size of the encoding should be supplied and should be equal
// to the maximum of the number of the private hands over the players.
void WritePositionalHand(int net_id, absl::Span<float_net> write_to) {
  std::fill(write_to.begin(), write_to.end(), 0.);
  write_to[net_id] = 1.;
}

void WriteParticleDataPoint(const PublicState& state,
                            const std::array<std::vector<int>, 2>& parview_perms,
                            const ParticleDims& dims,
                            ParticleDataPoint* point,
                            std::shared_ptr<Observer> hand_observer) {
  // Important !!
  point->Reset();

  // Find out how many parviews we will write.
  const std::array<int, 2> num_parviews = {
      std::min((int) state.nodes[0].size(), dims.max_parviews / 2),
      std::min((int) state.nodes[1].size(), dims.max_parviews / 2),
  };
  SPIEL_CHECK_GE(num_parviews[0], 1);
  SPIEL_CHECK_GE(num_parviews[1], 1);

  // Write inputs
  int i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int k = 0; k < num_parviews[pl]; k++) {
      int j = parview_perms[pl][k];  // Infostate idx.
      SPIEL_CHECK_GE(j, 0);
      SPIEL_CHECK_LT(j, state.nodes[pl].size());

      ParviewDataPoint parview = point->parview_at(i);
      // Hand features.
      SPIEL_CHECK_GT(state.nodes[pl][j]->corresponding_states_size(), 0);
      const State& repr_state = *state.nodes[pl][j]->corresponding_states()[0];
      ContiguousAllocator allocator(parview.hand_features());
      hand_observer->WriteTensor(repr_state, pl, &allocator);
      // Player features.
      parview.player_features()[1-pl] = 0.;
      parview.player_features()[pl] = 1.;
      // Hand beliefs.
      SPIEL_DCHECK_TRUE(std::isfinite(state.beliefs[pl][j]));
      parview.range() = state.beliefs[pl][j];
      i++;
    }
    point->num_parviews(pl) = num_parviews[pl];
  }
  SPIEL_CHECK_EQ(i, num_parviews[0] + num_parviews[1]);
  Copy(state.public_tensor.Tensor(), point->public_features());

  // Write outputs
  i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int k = 0; k < num_parviews[pl]; k++) {
      int j = parview_perms[pl][k];  // Infostate idx.
      SPIEL_CHECK_GE(j, 0);
      SPIEL_CHECK_LT(j, state.nodes[pl].size());

      ParviewDataPoint parview = point->parview_at(i);
      SPIEL_DCHECK_TRUE(std::isfinite(state.values[pl][j]));
      parview.value() = state.values[pl][j];
      i++;
    }
  }
  SPIEL_CHECK_EQ(i, num_parviews[0] + num_parviews[1]);
}

void CopyValuesFromNetToTree(ParticleDataPoint data_point,
                             PublicState& state,
                             const std::array<std::vector<int>, 2>& parview_perms,
                             const ParticleDims& dims) {
  const std::array<int, 2> num_parviews = {
      std::min((int) state.nodes[0].size(), dims.max_parviews / 2),
      std::min((int) state.nodes[1].size(), dims.max_parviews / 2),
  };
  SPIEL_CHECK_GE(num_parviews[0], 1);
  SPIEL_CHECK_GE(num_parviews[1], 1);

  int parview_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      state.values[pl][j] = 0;  // Zero-out all values first.
    }
    for (int k = 0; k < num_parviews[pl]; k++) {
      int j = parview_perms[pl][k];  // Infostate idx.
      SPIEL_CHECK_GE(j, 0);
      SPIEL_CHECK_LT(j, state.nodes[pl].size());

      ParviewDataPoint parview = data_point.parview_at(parview_index);
      state.values[pl][j] = parview.value();
      SPIEL_DCHECK_TRUE(std::isfinite(state.values[pl][j]));
      parview_index++;
    }
  }
  SPIEL_CHECK_EQ(data_point.total_parviews(), parview_index);
}

void WritePositionalDataPoint(const PublicState& state,
                              const NetContext& net_context,
                              const PositionalDims& dims, PositionalDataPoint* point) {
  // Important !!
  point->Reset();

  // Write inputs
  Copy(state.public_tensor.Tensor(), point->public_features());
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      point->range_at(pl, net_context.net_index(pl, j)) = state.beliefs[pl][j];
    }
  }
  // Write outputs
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      point->value_at(pl, net_context.net_index(pl, j)) = state.values[pl][j];
    }
  }
}

void CopyValuesNetToTree(PositionalDataPoint* point,
                         PublicState& state,
                         const NetContext& net_context) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      state.values[pl][j] = point->value_at(pl, net_context.net_index(pl, j));
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel


