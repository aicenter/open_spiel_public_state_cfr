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

std::unique_ptr<PublicStateContext> NetEvaluator::CreateContext(
    const LeafPublicState& leaf_state) const {
  if (leaf_state.IsTerminal()) {
    return nullptr;
  }

  Observation hand_observation(*game_, hand_observer_);
  return std::make_unique<HandContext>(leaf_state, hand_observation);
}

// TODO: evaluate all public states with a batch.
void NetEvaluator::EvaluatePublicState(
    algorithms::dlcfr::LeafPublicState* state,
    algorithms::dlcfr::PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_TRUE(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.
  auto* hand_context = open_spiel::down_cast<HandContext*>(context);

  ParticleData point = batch_->point_at(*dims_, 0);
  SPIEL_DCHECK_TRUE(point.is_valid_view());
  WriteParticles(*state, *hand_context, &point);

  torch::Tensor input = point.data.to(*device_);
  torch::Tensor output = model_->forward(input);
  SPIEL_DCHECK_EQ(output.sizes(), point.target.sizes());
  point.target.copy_(output);
  // TODO
//  CopyValuesNetToTree(point, *state, tables_);
}

}  // namespace papers_with_code
}  // namespace open_spiel


