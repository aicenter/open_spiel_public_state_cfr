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

// TODO: evaluate all public states with a batch.
void NetEvaluator::EvaluatePublicState(
    algorithms::dlcfr::LeafPublicState* state,
    algorithms::dlcfr::PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  SPIEL_DCHECK_FALSE(context);
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  ParticlesInContext point = batch_->point_at(0);

  // !! Do not shuffle, so that we can get back the values in an ordered way !!
  WriteParticles(*state, hand_tables_, *dims_, &point,
                 nullptr, /*shuffle_input=*/false, /*shuffle_output=*/false);

  // Input must be batched.
  torch::Tensor input = point.data.to(*device_).unsqueeze(/*dim=*/0);
  // The output must be "unbatched".
  torch::Tensor output = model_->forward(input).squeeze(/*dim=*/0);
  SPIEL_DCHECK_EQ(output.sizes(), point.target.sizes());
  SPIEL_DCHECK_EQ(output.size(/*dim=*/0), dims_->point_output_size());
  point.target.copy_(output);

  // !! This does not work with shuffling !!
  CopyValuesNetToTree(point, *state, hand_tables_, *dims_);
}

}  // namespace papers_with_code
}  // namespace open_spiel


