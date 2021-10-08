// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/evaluator.h"

namespace open_spiel {
namespace papers_with_code {

// CFR-based subgame solver that evaluates public leaves using terminal
// or non-terminal evaluator.
class SubgameSolver {
 public:
  SubgameSolver(
      std::shared_ptr<Subgame> subgame,
      const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
      const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
      const std::shared_ptr<std::mt19937> rnd_gen,
      const std::string& bandit_name,
      algorithms::PolicySelection save_values_policy =
          algorithms::kDefaultPolicySelection,
      bool safe_resolving = false,
      bool beliefs_for_average = false,
      double noisy_values = 0.);

  SubgameSolver(std::shared_ptr<Subgame> subgame,
                std::vector<algorithms::BanditVector> bandits);

  void RunSimultaneousIterations(int iterations);
  void Reset();
  void ResetCumulValues();

  // Accessors.
  PublicState& initial_state() { return subgame_->initial_state(); }
  std::vector<algorithms::BanditVector>& bandits() { return bandits_; }
  Subgame* subgame() { return subgame_.get(); }
  int num_iterations() const { return num_iterations_; }

  // Policy available only for the infostates of the subgame!
  std::shared_ptr<Policy> AveragePolicy();
  std::shared_ptr<Policy> CurrentPolicy();
 private:
  const std::shared_ptr<Subgame> subgame_;
  const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator_;
  const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator_;
  const std::shared_ptr<std::mt19937> rnd_gen_;
  const bool safe_resolving_;
  const bool beliefs_for_average_;
  const double noisy_values_;

  // -- Mutable values to keep track of. --
  // These have the size at largest depth of the tree, i.e. the size of the
  // leaf infostate nodes.
  std::vector<algorithms::BanditVector> bandits_;
  std::vector<std::vector<double>> reach_probs_;
  std::vector<std::vector<double>> cf_values_;
  // Save evaluator-specific information for any public state.
  // If no information should be saved, a nullptr is used.
  std::vector<std::unique_ptr<PublicStateContext>> contexts_;

  size_t num_iterations_ = 0;
  algorithms::PolicySelection init_save_values_;

  void EvaluateLeaves();
  void EvaluateLeaf(PublicState* state, PublicStateContext* context);
  void CopyCurrentValuesToInitialState();
  void IncrementallyAverageValuesInState(PublicState* state);
};

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVER_
