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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_FACTORY_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_FACTORY_

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/algorithms/infostate_tree.h"


namespace open_spiel {
namespace papers_with_code {

constexpr const char* kDefaultDlCfrBandit = "RegretMatchingPlus";
constexpr int kDefaultCfrIterations = 100;

// Produce a solver given a subgame.
struct SolverFactory {
  std::shared_ptr<const TerminalEvaluator> terminal_evaluator;
  std::shared_ptr<const NetEvaluator> leaf_evaluator;
  int cfr_iterations = kDefaultCfrIterations;
  std::string use_bandits_for_cfr = kDefaultDlCfrBandit;
  SaveValuesPolicy save_values_policy = SaveValuesPolicy::kAveragedCfValues;

  std::unique_ptr<SubgameSolver> MakeSolver(
      std::shared_ptr<Subgame> subgame,
      std::shared_ptr<const PublicStateEvaluator> custom_leaf_evaluator = nullptr,
      std::string custom_bandits_for_cfr = "") const;
};


} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_FACTORY_

