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

void NetEvaluator::EvaluatePublicState(
    algorithms::dlcfr::LeafPublicState* state,
    algorithms::dlcfr::PublicStateContext* context) const {
  SPIEL_DCHECK_FALSE(context);  // Nets do not use any special context.
  SPIEL_DCHECK_FALSE(state->IsTerminal());  // Only non-terminal leafs.
  torch::NoGradGuard no_grad_guard;  // We run only inference.

  // TODO: evaluate all public states with a batch.
  PositionalData point = batch_->point_at(0, dims_);
  point.Reset();
  CopyFeatures(state->public_tensor.Tensor(), point);
  CopyRangesTreeToNet(*state, point, tables_);

  torch::Tensor input = point.data.to(*device_);
  torch::Tensor output = model_->forward(input);

  point.target.copy_(output);
  CopyValuesNetToTree(point, *state, tables_);
}

}  // namespace papers_with_code
}  // namespace open_spiel


