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

#include "open_spiel/spiel_bots.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/bot.h"

#include <cmath>
#include <iostream>

namespace open_spiel {
namespace papers_with_code {
namespace {

void TestBotCanPlayGoofspiel() {
  const int num_games = 10;
  std::string current_dir = __FILE__;
  current_dir.resize(current_dir.rfind("/"));

  // TODO: implement game transformation that provides a chance node at the top
  //       with Left-Right actions.
  std::shared_ptr<const Game> game = LoadGame("goofspiel("
                                                "players=2,"
                                                "num_cards=3,"
                                                "imp_info=True,"
                                                "points_order=descending"
                                              ")");

  BotParameters params {
    {"seed",                   BotParameter(0)},
    {"num_layers",             BotParameter(5)},
    {"num_width",              BotParameter(5)},
    {"num_inputs_regression",  BotParameter(8)},
    {"cfr_iterations",         BotParameter(100)},
    {"max_move_ahead_limit",   BotParameter(1)},
    {"max_particles",          BotParameter(1000)},
    {"device",                 BotParameter("cpu")},
    {"use_bandits_for_cfr",    BotParameter("RegretMatchingPlus")},
    {"save_values_policy",     BotParameter("average")},
    {"zero_sum_regression",    BotParameter(false)},
    {"load_from",
     BotParameter(absl::StrCat(current_dir, "/snapshots/iigs3/random.model"))},
  };

  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < 2; ++p) {
    params["seed"] = BotParameter(p);  // Different seeds for different outcomes.
    bots.push_back(LoadBot("sherlock", game, p, params));
  }

  for (int i = 0; i < num_games; i++) {
    for (Player p = 0; p < 2; ++p) bots[p]->Restart();
    std::cout << "New game\n";
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      std::vector<std::pair<ActionsAndProbs, Action>> steps {
          bots[0]->StepWithPolicy(*state),
          bots[1]->StepWithPolicy(*state),
      };
      std::vector<Action> actions = {steps[0].second, steps[1].second};
      state->ApplyActions(actions);
      std::cout << "Played: " << actions << " with probs: \n";
      std::cout << "PL0: " << GetProbs(steps[0].first) << "\n";
      std::cout << "PL1: " << GetProbs(steps[1].first) << "\n";

    }
    std::cout << "Outcome: " << state->Returns() << "\n";
    std::cout << "---\n";
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestBotCanPlayGoofspiel();
}
