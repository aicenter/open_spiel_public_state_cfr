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

struct ZeroSumSequentialGameSolution {
  double game_value;
  // Optimal policy. Could be computed only for a single player, see below.
  TabularPolicy policy;
};

// A basic implementation: computes game value and tabular policy for both
// players in the whole game.
std::unique_ptr<ZeroSumSequentialGameSolution> SolveZeroSumSequentialGame(
    const Game& game);

// A more advanced implementation, where we can restrict the computation only
// to a "subset" of the infostate tree (specified by the starting states and
// their chance reach probabilities).
// This is useful for the computation of optimal extensions of depth-limited
// subgames [2].
//
// [2]: Value Functions for Depth-Limited Solving in Imperfect-Information Games
//      https://arxiv.org/abs/1906.06412
std::unique_ptr<ZeroSumSequentialGameSolution> SolveZeroSumSequentialGame(
    std::shared_ptr<Observer> infostate_observer,
    const std::vector<const State*>& start_states,
    const std::vector<float>& chance_reach_probs,
    std::optional<int> solve_only_player = {},
    bool collect_tabular_policy = true);


}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ORTOOLS_SEQUENCE_FORM_LP_H_
