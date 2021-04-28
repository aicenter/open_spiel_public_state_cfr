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

#ifndef OPEN_SPIEL_ALGORITHMS_ORTOOLS_DL_ORACLE_EVALUATOR_H_
#define OPEN_SPIEL_ALGORITHMS_ORTOOLS_DL_ORACLE_EVALUATOR_H_

#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

// Unfortunately sometimes the LP solver may end up with an infeasible problem
// statement due to numerical rounding errors, especially in larger games: I had
// issues in GoofSpiel 5. A workaround is to make an epsilon convex combination
// of each behavioral (infostate) strategy with the uniform strategy. This will
// increase exploitability, however only slightly. This is usually not an issue
// for evaluation of the iterative algorithms (CFR) that approach Nash
// equilibria, as the error will be small. If you want to turn this behavior
// off, simply pass zero to the function calls, however this might unexpectedly
// terminate the process, if the problem is indeed infeasible!
constexpr double kStrategyEpsilon = 1e-5;

double ComputeRootValueWhileFixingStrategy(
    SequenceFormLpSpecification* specification, const Policy& fixed_policy,
    Player fixed_player, double strategy_epsilon = kStrategyEpsilon);

// Based on Proposition 3.11 in the value functions paper [1], as the average of
// the individual player exploitabilities -- we don't need to know the game
// value.
double TrunkExploitability(SequenceFormLpSpecification* spec,
                           const Policy& trunk_policy,
                           double strategy_epsilon = kStrategyEpsilon);

// Based on Proposition 3.11 in the value functions paper [1].
// If you do not supply game value, it will be calculated automatically
// (takes longer time).
double TrunkPlayerExploitability(
    SequenceFormLpSpecification* spec, const Policy& trunk_policy, Player p,
    absl::optional<double> maybe_game_value = {},
    double strategy_epsilon = kStrategyEpsilon);

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_ORTOOLS_DL_ORACLE_EVALUATOR_H_
