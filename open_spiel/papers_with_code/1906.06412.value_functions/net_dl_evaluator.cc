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
    const LeafPublicState& leaf_state) const {

  SPIEL_DCHECK_FALSE(leaf_state.IsTerminal());
  auto net_context = std::make_unique<NetContext>(hand_info_);
  Observation& hand = hand_info_->hand_buffer;

  for (int pl = 0; pl < 2; ++pl) {
    for (int tree_idx = 0;
         tree_idx < leaf_state.leaf_nodes[pl].size(); ++tree_idx) {
      const InfostateNode* node = leaf_state.leaf_nodes[pl][tree_idx];
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
    algorithms::dlcfr::LeafPublicState* state,
    algorithms::dlcfr::PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(model_->is_training());
  SPIEL_DCHECK_TRUE(context);
  auto net_context = open_spiel::down_cast<NetContext*>(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  ParticlesInContext point = batch_->point_at(0);
  // !! Do not shuffle, so that we can get back the values in an ordered way !!
  WriteParticles(*state, *net_context, *dims_, &point, /*rnd_gen=*/nullptr,
                 /*shuffle_input=*/false, /*shuffle_output=*/false);

  // Input must be batched.
  torch::Tensor input = point.data.to(*device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input).squeeze(/*dim=*/0);
  SPIEL_DCHECK_EQ(output.sizes().size(), 1);
  SPIEL_DCHECK_EQ(output.size(/*dim=*/0), point.num_particles());
  point.target.index_put_({Slice(0, point.num_particles())}, output);

  // !! This does not work with shuffling !!
  CopyValuesNetToTree(point, *state, *dims_);
}

void PositionalNetEvaluator::EvaluatePublicState(
    algorithms::dlcfr::LeafPublicState* state,
    algorithms::dlcfr::PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(model_->is_training());
  SPIEL_DCHECK_TRUE(context);
  auto net_context = open_spiel::down_cast<NetContext*>(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  PositionalData point = batch_->point_at(0, *dims_);
  WritePositional(*state, *net_context, *dims_, &point);

  torch::Tensor input = point.data.to(*device_);
  torch::Tensor output = model_->forward(input);
  SPIEL_DCHECK_EQ(output.sizes(), point.target.sizes());
  point.target.copy_(output);
  CopyValuesNetToTree(&point, *state, *net_context);
}

}  // namespace papers_with_code
}  // namespace open_spiel


