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

#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"

namespace open_spiel {
namespace papers_with_code {

std::unique_ptr<SubgameSolver> SolverFactory::MakeSolver(
    std::shared_ptr<Subgame> subgame,
    std::shared_ptr<const PublicStateEvaluator> custom_leaf_evaluator,
    std::string custom_bandits_for_cfr) const {
  auto evaluator = custom_leaf_evaluator == nullptr ? leaf_evaluator
                                                    : custom_leaf_evaluator;
  auto bandits = custom_bandits_for_cfr.empty() ? use_bandits_for_cfr
                                                : custom_bandits_for_cfr;
  return std::make_unique<SubgameSolver>(subgame, evaluator,
                                         terminal_evaluator, bandits);
}

}  // papers_with_code
}  // open_spiel

