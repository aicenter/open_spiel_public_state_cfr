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

#include "open_spiel/papers_with_code/1906.06412.value_functions/hand_table.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

using PublicStateContext = algorithms::dlcfr::PublicStateContext;
using LeafPublicState = algorithms::dlcfr::LeafPublicState;
using LeafEvaluator = algorithms::dlcfr::LeafEvaluator;

class NetEvaluator final : public LeafEvaluator {
  ValueNet* model_;
  torch::Device* device_;
  const std::vector<HandTable>& tables_;
  BatchData* batch_;
  ParticleDims* const dims_;
  std::shared_ptr<const Game> game_;
  std::shared_ptr<Observer> hand_observer_;

 public:
  NetEvaluator(ValueNet* model, torch::Device* device,
               const std::vector<HandTable>& tables,
               BatchData* batch, ParticleDims* const dims,
               std::shared_ptr<const Game> game,
               std::shared_ptr<Observer> hand_observer)
      : model_(model), device_(device), tables_(tables), batch_(batch),
        dims_(dims), game_(game), hand_observer_(hand_observer) {}

  std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& leaf_state) const override;

  void EvaluatePublicState(LeafPublicState* state,
                           PublicStateContext* context) const override;
};

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
