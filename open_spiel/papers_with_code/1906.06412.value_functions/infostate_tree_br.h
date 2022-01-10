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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_INFOSTATE_TREE_BR_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_INFOSTATE_TREE_BR_

#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

// BR on infostate trees.
std::vector<double> BestResponse(
    std::vector<std::shared_ptr<algorithms::InfostateTree>> trees,
    const Policy& fixed_policy);

std::vector<algorithms::BanditVector> MakeResponseBandits(
    const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees,
    const std::array<std::vector<double>, 2>& beliefs,
    const Policy& optimal_brs);

std::vector<algorithms::BanditVector> MakeResponseBandits(
    const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees,
    const Policy& optimal_brs);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_INFOSTATE_TREE_BR_
