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
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;
using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

std::unique_ptr<PublicStateContext> NetEvaluator::CreateContext(
    const PublicState& state) const {

  SPIEL_DCHECK_FALSE(state.IsTerminal());
  auto net_context = std::make_unique<NetContext>(hand_info_);
  Observation& hand = hand_info_->hand_buffer;

  for (int pl = 0; pl < 2; ++pl) {
    for (int tree_idx = 0;
         tree_idx < state.nodes[pl].size(); ++tree_idx) {
      const InfostateNode* node = state.nodes[pl][tree_idx];
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
  SPIEL_DCHECK_TRUE(context);
  auto net_context = open_spiel::down_cast<NetContext*>(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  ParticleDataPoint point = batch_->point_at(0, *dims_);
  // !! Do not shuffle, so that we can get back the values in an ordered way !!
  WriteParticleDataPoint(*state, *net_context, *dims_, &point,
                         /*rnd_gen=*/nullptr, /*shuffle_input_output=*/false);

  // Input must be batched.
  torch::Tensor input = point.data.to(*device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input).squeeze(/*dim=*/0);
  SPIEL_DCHECK_EQ(output.sizes().size(), 1);
  SPIEL_DCHECK_EQ(output.size(/*dim=*/0), point.num_parviews());
  point.target.index_put_({Slice(0, point.num_parviews())}, output);

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

  PositionalData point = batch_->point_at(0, *dims_);
  WritePositionalDataPoint(*state, *net_context, *dims_, &point);

  torch::Tensor input = point.data.to(*device_);
  torch::Tensor output = model_->forward(input);
  SPIEL_DCHECK_EQ(output.sizes(), point.target.sizes());
  point.target.copy_(output);
  CopyValuesNetToTree(&point, *state, *net_context);
}

std::shared_ptr<NetEvaluator> MakeNetEvaluator(
    BasicDims* dims, HandInfo* hand_info, ValueNet* model,
    BatchData* eval_batch, torch::Device* device) {
  switch (model->architecture()) {
    case NetArchitecture::kParticle: {
      auto particle_model =
          open_spiel::down_cast<ParticleValueNet*>(model);
      auto particle_dims =
          open_spiel::down_cast<ParticleDims*>(dims);
      return std::make_shared<ParticleNetEvaluator>(
          hand_info, particle_model, particle_dims, eval_batch, device);
    }
    case NetArchitecture::kPositional: {
      auto positional_model =
          open_spiel::down_cast<PositionalValueNet*>(model);
      auto positional_dims =
          open_spiel::down_cast<PositionalDims*>(dims);
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

void WriteParticleDataPoint(const algorithms::dlcfr::PublicState& state,
                            const NetContext& net_context,
                            const ParticleDims& dims, ParticleDataPoint* point,
                            std::mt19937* rnd_gen, bool shuffle_input_output) {
  // Important !!
  point->Reset();

  // Find out how many parviews we will write.
  const int num_parviews = state.nodes[0].size()
                         + state.nodes[1].size();
  SPIEL_CHECK_GE(num_parviews, 2);

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
      if(dims.write_hand_features_positionally()) {
        WritePositionalHand(net_context.net_index(pl, j),
                            parview.hand_features());
      } else {
        const Observation& hand_observation = net_context.hand_at(pl, j);
        Copy(hand_observation.Tensor(), parview.hand_features());
      }
      parview.player_features()[pl] = 1.;
      parview.range() = state.beliefs[pl][j];
      i++;
    }
  }
  SPIEL_CHECK_EQ(i, num_parviews);
  point->num_parviews() = num_parviews;
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
                             algorithms::dlcfr::PublicState& state,
                             const ParticleDims& dims) {
  int parview_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      ParviewDataPoint parview = data_point.parview_at(parview_index);
      state.values[pl][j] = parview.value();
      parview_index++;
    }
  }
  SPIEL_CHECK_EQ(data_point.num_parviews(), parview_index);
}

void WritePositionalDataPoint(const algorithms::dlcfr::PublicState& state,
                              const NetContext& net_context,
                              const PositionalDims& dims, PositionalData* point) {
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

void CopyValuesNetToTree(PositionalData* point,
                         algorithms::dlcfr::PublicState& state,
                         const NetContext& net_context) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.nodes[pl].size(); j++) {
      state.values[pl][j] = point->value_at(pl, net_context.net_index(pl, j));
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel


