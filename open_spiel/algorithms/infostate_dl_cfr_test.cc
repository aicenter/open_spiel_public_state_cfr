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

#include "open_spiel/algorithms/infostate_cfr.h"

#include <cmath>
#include <iostream>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "infostate_dl_cfr.h"

namespace open_spiel {
namespace algorithms {
namespace {

void TestTerminalEvaluatorHasSameIterations() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  const int cfr_iterations = 10;

  InfostateCFR vec_solver(*game);
  std::unordered_map<std::string, CFRInfoStateValues const*> vec_ptable =
      vec_solver.InfoStateValuesPtrTable();

  std::shared_ptr<LeafEvaluator> terminal_evaluator = MakeTerminalEvaluator();

  // We use only the terminal evaluator.
  DepthLimitedCFR dl_solver(game, /*depth_limit=*/100,
                            /*leaf_evaluator=*/nullptr, terminal_evaluator);
  std::unordered_map<std::string, CFRInfoStateValues const*> dl_ptable =
      dl_solver.InfoStateValuesPtrTable();

  SPIEL_CHECK_EQ(vec_ptable.size(), dl_ptable.size());

  for (int i = 0; i < cfr_iterations; ++i) {
    vec_solver.RunSimultaneousIterations(1);
    dl_solver.RunSimultaneousIterations(1);

    for (const auto& [infostate, dl_ptr] : dl_ptable) {
      const CFRInfoStateValues& dl_values = *dl_ptr;
      const CFRInfoStateValues& vec_values = *(vec_ptable.at(infostate));
      SPIEL_CHECK_EQ(dl_values.num_actions(), vec_values.num_actions());

      // Check regrets.
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        SPIEL_CHECK_TRUE(fabs(vec_values.cumulative_regrets[j]
                                  - dl_values.cumulative_regrets[j]) < 1e-6);
      }
      // Cumulative policy is more tricky: we need to normalize it first.
      double str_cumul_sum = 0, vec_cumul_sum = 0;
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        str_cumul_sum += dl_values.cumulative_policy[j];
        vec_cumul_sum += vec_values.cumulative_policy[j];
      }
      for (int j = 0; j < vec_values.num_actions(); ++j) {
        SPIEL_CHECK_TRUE(fabs(
            vec_values.cumulative_policy[j] / vec_cumul_sum
                - dl_values.cumulative_policy[j] / str_cumul_sum) < 1e-6);
      }
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms;

int main(int argc, char** argv) {
  algorithms::TestTerminalEvaluatorHasSameIterations();
}
