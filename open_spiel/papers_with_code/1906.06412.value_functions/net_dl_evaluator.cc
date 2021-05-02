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

// TODO: evaluate all public states with a batch.
void ParticleNetEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(model_->is_training());
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  ParticleDataPoint point = batch_->point_at(0, *dims_);
  // !! Do not shuffle, so that we can get back the values in an ordered way !!
  WriteParticleDataPoint(*state, *dims_, &point, hand_observer_,
                         /*rnd_gen=*/nullptr, /*shuffle_input_output=*/false);

  // Input must be batched.
  torch::Tensor input = point.data.to(device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input);
  SPIEL_DCHECK_EQ(output.sizes().size(), 1);
  SPIEL_DCHECK_EQ(output.size(/*dim=*/0), point.total_parviews());
  point.target.index_put_({Slice(0, point.total_parviews())}, output);

  // !! This does not work with shuffling !!
  CopyValuesFromNetToTree(point, *state, *dims_);
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
    std::shared_ptr<BasicDims> dims, std::shared_ptr<ValueNet> model,
    std::shared_ptr<BatchData> eval_batch, torch::Device device,
    // One of:
    std::shared_ptr<HandInfo> hand_info, std::shared_ptr<Observer> hand_observer) {
  switch (model->architecture()) {
    case NetArchitecture::kParticle: {
      auto particle_model = std::dynamic_pointer_cast<ParticleValueNet>(model);
      auto particle_dims = std::dynamic_pointer_cast<ParticleDims>(dims);
      return std::make_shared<ParticleNetEvaluator>(
          particle_model, particle_dims, eval_batch, device, hand_observer);
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
                            const ParticleDims& dims, ParticleDataPoint* point,
                            std::shared_ptr<Observer> hand_observer,
                            std::mt19937* rnd_gen, bool shuffle_input_output) {
  // Important !!
  point->Reset();

  // Find out how many parviews we will write.
  const int num_parviews = state.nodes[0].size()
                         + state.nodes[1].size();
  SPIEL_CHECK_GE(num_parviews, 2);
  SPIEL_CHECK_LE(num_parviews, dims.max_parviews);

  // Make a random permutation if something should be shuffled.
  std::vector<int> parview_placement(num_parviews);
  if (shuffle_input_output) {
    SPIEL_CHECK_TRUE(rnd_gen);
    std::iota(parview_placement.begin(), parview_placement.end(), 0);
    std::shuffle(parview_placement.begin(), parview_placement.end(),
                 *rnd_gen);
  }

  // Write inputs
  int i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      ParviewDataPoint parview = point->parview_at(shuffle_input_output
                                                   ? parview_placement[i] : i);
      // Hand features.
      SPIEL_CHECK_GT(state.nodes[pl][j]->corresponding_states_size(), 0);
      const State& repr_state = *state.nodes[pl][j]->corresponding_states()[0];
      ContiguousAllocator allocator(parview.hand_features());
      hand_observer->WriteTensor(repr_state, pl, &allocator);

      parview.player_features()[pl] = 1.;
      parview.range() = state.beliefs[pl][j];
      i++;
    }
    point->num_parviews(pl) = state.nodes[pl].size();
  }
  SPIEL_CHECK_EQ(i, num_parviews);
  Copy(state.public_tensor.Tensor(), point->public_features());

  // Write outputs
  i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      ParviewDataPoint parview = point->parview_at(shuffle_input_output
                                                   ? parview_placement[i] : i);
      parview.value() = state.values[pl][j];
      i++;
    }
  }

  SPIEL_CHECK_EQ(i, num_parviews);
}

void CopyValuesFromNetToTree(ParticleDataPoint data_point,
                             PublicState& state,
                             const ParticleDims& dims) {
  int parview_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      ParviewDataPoint parview = data_point.parview_at(parview_index);
      state.values[pl][j] = parview.value();
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


