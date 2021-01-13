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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_

#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

class NetEvaluator final : public algorithms::dlcfr::LeafEvaluator {
  ValueNet* model_;
  torch::Device* device_;
  const std::vector<algorithms::dlcfr::RangeTable>& tables_;
  BatchData* batch_;
  PositionalDataDims dims_;

 public:
  NetEvaluator(ValueNet* model, torch::Device* device,
               const std::vector<algorithms::dlcfr::RangeTable>& tables,
               BatchData* batch, const PositionalDataDims& dims)
      : model_(model), device_(device), tables_(tables), batch_(batch),
        dims_(dims) {}

  void EvaluatePublicState(
      algorithms::dlcfr::LeafPublicState* state,
      algorithms::dlcfr::PublicStateContext* context) const override;
};

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
