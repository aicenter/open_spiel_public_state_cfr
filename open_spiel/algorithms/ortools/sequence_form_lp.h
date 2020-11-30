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

#ifndef OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_
#define OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_

#include <vector>
#include <string>
#include <memory>
#include <array>
#include <unordered_map>

#include "ortools/linear_solver/linear_solver.h"

#include "open_spiel/policy.h"
#include "open_spiel/algorithms/infostate_tree.h"


// An implementation of a sequence-form linear program for computing Nash
// equilibria in sequential games, based on [1]. The implementation constructs
// infostate trees for both players, connects them through the terminals and
// recursively specifies constraints on reach probability of the player and
// counterfactual values of the opponent.
//
// [1]:  Efficient Computation of Equilibria for Extensive Two-Person Games
//       http://www.maths.lse.ac.uk/Personal/stengel/TEXTE/geb1996b.pdf


namespace open_spiel {
namespace algorithms {
namespace ortools {


template<class T>
struct BijectiveContainer {
  std::map<T, T> x2y;
  std::map<T, T> y2x;

  void put(std::pair<T, T> xy) {
    const T& x = xy.first;
    const T& y = xy.second;
    SPIEL_CHECK_TRUE(x2y.find(x) == x2y.end());
    SPIEL_CHECK_TRUE(y2x.find(y) == y2x.end());
    x2y[x] = y;
    y2x[y] = x;
  }

  const std::map<T, T>& association(int direction) const {
    SPIEL_CHECK_TRUE(direction == 0 || direction == 1);
    if (direction == 0) return x2y;
    else return y2x;
  }
};

BijectiveContainer<const InfostateNode*> ConnectTerminals(
    const InfostateTree& tree_a, const InfostateTree& tree_b);

// Variables needed for solving the LP.
struct SolverVariables {
  operations_research::MPVariable* var_cf_value;
  operations_research::MPVariable* var_reach_prob;
  operations_research::MPConstraint* ct_child_cf_value;
  operations_research::MPConstraint* ct_parent_cf_value;
  operations_research::MPConstraint* ct_child_reach_prob;
  operations_research::MPConstraint* ct_parent_reach_prob;
};

class SequenceFormLpSolver {
 using MPSolver = operations_research::MPSolver;
 public:
  SequenceFormLpSolver(
      std::array<std::shared_ptr<InfostateTree>, 2> solver_trees,
      const std::string& solver_id = "GLOP_LINEAR_PROGRAMMING");
  SequenceFormLpSolver(const Game& game);

  // Specify the linear program for given player.
  void SpecifyLinearProgram(Player pl);

  // Solve the linear program.
  // Returns the objective value (root value for the player).
  double Solve();

  // Reset the solver and erase all pointers.
  // This is called automatically when you call SpecifyLinearProgram.
  void ClearSpecification();

  // Transform the computed sequence form policy into a behavioral policy.
  // This function can be called only after call for Solve().
  TabularPolicy OptimalPolicy(Player for_player);

  // Transform the computed realization plan into a behavioral policy.
  // This function can be called only after call for Solve().
  SfStrategy OptimalSfStrategy(Player for_player);

  // For debugging.
  void PrintProblemSpecification();

  const std::array<std::shared_ptr<InfostateTree>, 2>& trees() const {
    return solver_trees_;
  }
  std::unordered_map<
      const InfostateNode*, SolverVariables>& lp_specification() {
    return lp_spec_;
  }
  operations_research::MPSolver* solver() { return solver_.get(); }

 protected:
  const std::array<std::shared_ptr<InfostateTree>, 2> solver_trees_;
  const BijectiveContainer<const InfostateNode*> terminal_bijection_;
  std::unique_ptr<operations_research::MPSolver> solver_;
  std::unordered_map<const InfostateNode*, SolverVariables> lp_spec_;

  void SpecifyReachProbsConstraints(InfostateNode* node);
  void SpecifyCfValuesConstraints(InfostateNode* node);
  void SpecifyRootConstraints(InfostateNode* root_node);
  void SpecifyObjective(InfostateNode* root_node);
};


}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_
