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

#include "open_spiel/algorithms/infostate_dl_cfr.h"

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
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/cfr.h"

namespace open_spiel {
namespace algorithms {
namespace dlcfr {
namespace {

void TestExploitabilityCalculation(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  DepthLimitedCFR dl_solver(game, /*depth_limit=*/100,
      /*leaf_evaluator=*/nullptr, MakeTerminalEvaluator());
  dl_solver.SimultaneousTopDownEvaluate();
  double actual_expl = dl_solver.TrunkExploitability();

  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    game = ConvertToTurnBased(*game);
  }
  CFRSolver str_solver(*game);
  double expected_expl = Exploitability(*game, *str_solver.CurrentPolicy());

  SPIEL_DCHECK_FLOAT_NEAR(expected_expl, actual_expl, 1e-6);
}

void CheckIterationConsistency(
    const CFRInfoStateValuesPtrTable& actual_table,
    const CFRInfoStateValuesPtrTable& expected_table) {
  for (const auto&[infostate, actual_ptr] : actual_table) {
    const CFRInfoStateValues& actual_values = *actual_ptr;
    const CFRInfoStateValues& expected_values = *(expected_table.at(infostate));
    SPIEL_CHECK_EQ(actual_values.num_actions(), expected_values.num_actions());

    // Check regrets.
    for (int j = 0; j < expected_values.num_actions(); ++j) {
      SPIEL_CHECK_TRUE(fabs(expected_values.cumulative_regrets[j]
                                - actual_values.cumulative_regrets[j]) < 1e-6);
    }
    // Cumulative policy is more tricky: we need to normalize it first.
    double act_cumul_sum = 0, exp_cumul_sum = 0;
    for (int j = 0; j < expected_values.num_actions(); ++j) {
      act_cumul_sum += actual_values.cumulative_policy[j];
      exp_cumul_sum += expected_values.cumulative_policy[j];
    }
    for (int j = 0; j < expected_values.num_actions(); ++j) {
      SPIEL_CHECK_TRUE(fabs(
          expected_values.cumulative_policy[j] / exp_cumul_sum
              - actual_values.cumulative_policy[j] / act_cumul_sum) < 1e-6);
    }
  }
}

void TestTerminalEvaluatorHasSameIterations(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int cfr_iterations = 10;

  InfostateCFR vec_solver(*game);
  CFRInfoStateValuesPtrTable vec_ptable = vec_solver.InfoStateValuesPtrTable();

  // We use only the terminal evaluator.
  std::shared_ptr<LeafEvaluator> terminal_evaluator = MakeTerminalEvaluator();
  DepthLimitedCFR dl_solver(game, /*depth_limit=*/100,
      /*leaf_evaluator=*/nullptr, terminal_evaluator);
  CFRInfoStateValuesPtrTable dl_ptable = dl_solver.InfoStateValuesPtrTable();

  SPIEL_CHECK_EQ(vec_ptable.size(), dl_ptable.size());
  for (int i = 0; i < cfr_iterations; ++i) {
    vec_solver.RunSimultaneousIterations(1);
    dl_solver.RunSimultaneousIterations(1);
    CheckIterationConsistency(dl_ptable, vec_ptable);
  }
}

std::unique_ptr<DepthLimitedCFR> MakeRecursiveDepthLimitedCFR(
    std::shared_ptr<const Game> game, int trunk_depth_limit,
    int subgame_depth_limit) {
  std::shared_ptr<const LeafEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  // Recursive leaf evaluator.
  auto leaf_evaluator = std::make_shared<CFREvaluator>(
      game, subgame_depth_limit, /*leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  leaf_evaluator->leaf_evaluator = leaf_evaluator;
  leaf_evaluator->num_cfr_iterations = 1;

  // Builds the root leaf public states so that we can call the recursive
  // evaluator.
  return std::make_unique<DepthLimitedCFR>(
      game, trunk_depth_limit, leaf_evaluator, terminal_evaluator);
}

void TestRecursiveDepthLimitedSolving(const std::string& game_name) {
  // If we make 1 iterations in each of the recursive subgames, it is the same
  // as if we were running CFR in the whole game. Thus we can check that we
  // compute the same regrets as the original implementation.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int trunk_iterations = 10;

  for (int trunk_depth_limit = 0; trunk_depth_limit < game->MaxMoveNumber();
       ++trunk_depth_limit) {
    for (int subgame_depth_limit = 1; subgame_depth_limit < 4;
         ++subgame_depth_limit) {

      InfostateCFR vec_solver(*game);
      CFRInfoStateValuesPtrTable vec_ptable =
          vec_solver.InfoStateValuesPtrTable();

      std::unique_ptr<DepthLimitedCFR> dl_solver = MakeRecursiveDepthLimitedCFR(
          game, trunk_depth_limit, subgame_depth_limit);
      CFRInfoStateValuesPtrTable
          dl_ptable = dl_solver->InfoStateValuesPtrTable();

      for (int j = 0; j < trunk_iterations; ++j) {
        vec_solver.RunSimultaneousIterations(1);
        dl_solver->RunSimultaneousIterations(1);
        SPIEL_CHECK_FLOAT_NEAR(
            vec_solver.RootValue(), dl_solver->RootCfValue(), 1e-6);
        CheckIterationConsistency(dl_ptable, vec_ptable);
      }
    }
  }
}

}  // namespace
}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms::dlcfr;

int main(int argc, char** argv) {
  std::vector<std::string> test_games = {
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=4,imp_info=True)",
  };

  for (const std::string& game_name : test_games) {
    algorithms::TestExploitabilityCalculation(game_name);
    algorithms::TestTerminalEvaluatorHasSameIterations(game_name);
    algorithms::TestRecursiveDepthLimitedSolving(game_name);
  }
}
