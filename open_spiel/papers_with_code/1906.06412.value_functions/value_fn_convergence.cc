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

#include <string>
#include <iostream>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");
ABSL_FLAG(int, depth, 3, "Max depth of the trunk.");
ABSL_FLAG(std::string, evaluator, "oracle", "Which evaluator to use");

#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

void RunTrunkIterationsWithValueOracle(std::string game_name, int depth) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  auto oracle_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);

  dlcfr::DepthLimitedCFR dl_solver(
      game, depth, oracle_evaluator, terminal_evaluator);

  SequenceFormLpSpecification whole_game(*game);
  auto current_policy = dl_solver.CurrentPolicy();
  auto average_policy = dl_solver.AveragePolicy();
  int num_iters = 10;

  std::cout << "iters,cur_expl,avg_expl" << std::endl;
  for (int i = 0; i < 1000; ++i) {
    double cur_expl = TrunkExploitability(&whole_game, *current_policy);
    double avg_expl = TrunkExploitability(&whole_game, *average_policy);
    std::cout << i * num_iters << ","
              << cur_expl << ","
              << avg_expl << std::endl;
    dl_solver.RunSimultaneousIterations(num_iters);
  }
}

void RunTrunkIterationsWithCfrEvaluator(std::string game_name, int depth) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  auto leaf_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      game, 100, nullptr, terminal_evaluator,
      public_observer, infostate_observer);
  leaf_evaluator->num_cfr_iterations = 100;

  dlcfr::DepthLimitedCFR dl_solver(
      game, depth, leaf_evaluator, terminal_evaluator);

  SequenceFormLpSpecification whole_game(*game);
  auto current_policy = dl_solver.CurrentPolicy();
  auto average_policy = dl_solver.AveragePolicy();
  int num_iters = 10;

  std::cout << "iters,cur_expl,avg_expl" << std::endl;
  for (int i = 0; i < 1000; ++i) {
    double cur_expl = TrunkExploitability(&whole_game, *current_policy);
    double avg_expl = TrunkExploitability(&whole_game, *average_policy);
    std::cout << i * num_iters << ","
              << cur_expl << ","
              << avg_expl << std::endl;
    dl_solver.RunSimultaneousIterations(num_iters);
  }
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "Experiment runner for the convergence of trunk strategies using "
      "the counterfactually (optimal) value functions. "
      "All games should have exploitability approaching zero.");
  absl::ParseCommandLine(argc, argv);

  std::string evaluator = absl::GetFlag(FLAGS_evaluator);
  if (evaluator == "oracle") {
    open_spiel::algorithms::ortools::RunTrunkIterationsWithValueOracle(
        absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_depth));
  } else if (evaluator == "cfr") {
    open_spiel::algorithms::ortools::RunTrunkIterationsWithCfrEvaluator(
        absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_depth));
  } else {
    open_spiel::SpielFatalError("Exhausted pattern match: no such evaluator");
  }
}
