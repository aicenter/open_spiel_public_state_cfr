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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

#include <iostream>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void CheckInfostatePolicy(
    const std::string& infostate, const Policy& a, const Policy& b) {
  ActionsAndProbs vec_policy = a.GetStatePolicy(infostate);
  ActionsAndProbs str_policy = b.GetStatePolicy(infostate);
  SPIEL_CHECK_EQ(vec_policy.size(), str_policy.size());
  for (int j = 0; j < vec_policy.size(); ++j) {
    SPIEL_CHECK_EQ(vec_policy[j].first, str_policy[j].first);
    SPIEL_CHECK_FLOAT_NEAR(vec_policy[j].second, str_policy[j].second, 1e-6);
  }
}

void TestMakeAllPublicStates(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);

  for (PublicState& s : all->public_states) {
// Debug print:
//    std::cout << "----" << std::endl;
//    std::cout << "Obs: " << s.public_tensor.Tensor() << "\n";
//    for (int pl = 0; pl < 2; ++pl) {
//      std::cout << "Nodes " << pl << ":\n";
//      for (const InfostateNode* node : s.nodes[pl]) {
//        std::cout << "  " << node->TreePath() << "\n";
//        std::cout << "  States:\n";
//        for (const std::unique_ptr<State> & state
//            : node->corresponding_states()) {
//          std::cout << "    " << state->HistoryString() << "\n";
//        }
//      }
//    }

    using History = std::vector<Action>;
    std::unordered_set<History, absl::Hash<History>> state_histories;
    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_FALSE(s.nodes[pl].empty());
      SPIEL_CHECK_FALSE(s.nodes[0][0]->corresponding_states().empty());
      State* a_state = s.nodes[0][0]->corresponding_states()[0].get();

      for (const algorithms::InfostateNode* node : s.nodes[pl]) {
        SPIEL_CHECK_FALSE(node->corresponding_states().empty());
        SPIEL_CHECK_EQ(node->tree().acting_player(), pl);

        for (const std::unique_ptr<State>& state
            : node->corresponding_states()) {
          const auto& h = state->History();
          SPIEL_CHECK_EQ(a_state->MoveNumber(), state->MoveNumber());
          if (pl == 0) {
            SPIEL_CHECK_TRUE(state_histories.find(h) == state_histories.end());
            state_histories.insert(h);
          } else {
            SPIEL_CHECK_TRUE(state_histories.find(h) != state_histories.end());
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::vector<std::string> test_games = {
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)",
  };

  for (const std::string& game_name : test_games) {
    open_spiel::papers_with_code::TestMakeAllPublicStates(game_name);
  }
}
