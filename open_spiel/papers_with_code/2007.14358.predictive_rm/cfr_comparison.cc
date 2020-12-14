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
#include <open_spiel/matrix_game.h>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");
ABSL_FLAG(std::string, bandit_name, "RegretMatching", "Which bandit to use.");

#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

void RunCfrIterations(const std::string& game_name,
                      const std::string& bandit_name) {
  std::shared_ptr<const Game> game;
  if (game_name == "small_matrix") {  // From the paper.
    game =  matrix_game::CreateMatrixGame(
      {{5., -1.}, {0.,  1.}},
      {{-5., 1.}, {-0.,  -1.}});
  } else {
    game = LoadGame(game_name);
  }

  std::vector<std::shared_ptr<InfostateTree>> trees = {
      MakeInfostateTree(*game, 0),
      MakeInfostateTree(*game, 1)
  };
  std::vector<BanditVector> bandits = MakeBanditVectors(trees, bandit_name);
  InfostateCFR solver(std::move(trees), std::move(bandits));

  const std::shared_ptr<Policy> average_policy = solver.AveragePolicy();
  const std::shared_ptr<Policy> current_policy = solver.CurrentPolicy();

  // String implementation does not support simultaneous move games.
  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    game = ConvertToTurnBased(*game);
  }

  bool alternating_updates = bandit_name.find("Plus") != std::string::npos;

  size_t num_iters = 1;
  std::cout << "iters,cur_expl,avg_expl" << std::endl;
  for (int i = 0; i < 2000; ++i) {
    double cur_expl = Exploitability(*game, *current_policy);
    double avg_expl = Exploitability(*game, *average_policy);
    std::cout << i * num_iters << ","
              << cur_expl << ","
              << avg_expl << std::endl;
    if (alternating_updates) {
      solver.RunAlternatingIterations(num_iters);
    } else {
      solver.RunSimultaneousIterations(num_iters);
    }
  }
}

}  // namespace
}  // namespace algorithms
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::SetProgramUsageMessage(
      "Experiment runner for the convergence of trunk strategies using "
      "the counterfactually (optimal) value functions. "
      "All games should have exploitability approaching zero.");
  absl::ParseCommandLine(argc, argv);

  open_spiel::algorithms::RunCfrIterations(
      absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_bandit_name));
}
