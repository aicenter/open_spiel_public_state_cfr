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

#include "open_spiel/papers_with_code/1906.06412.value_functions/hand_table.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"


namespace open_spiel {
namespace papers_with_code {
namespace {


void TestCreateHandTable() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> hand_observer = game->MakeObserver(kHandObsType, {});
  auto leaf_evaluator = algorithms::dlcfr::MakeDummyEvaluator();
  algorithms::dlcfr::DepthLimitedCFR dl_cfr(game, 3, leaf_evaluator, nullptr);

  std::vector<HandTable> tables = CreateHandTables(*game, hand_observer,
                                                   dl_cfr.public_leaves());
  SPIEL_CHECK_EQ(tables.size(), game->NumPlayers());

  for (const HandTable& player_table : tables) {
    SPIEL_CHECK_EQ(player_table.private_hands.size(), 3);
    SPIEL_CHECK_EQ(player_table.bijections.size(),
                   dl_cfr.public_leaves().size());

    for (const BijectiveContainer<size_t>& state_bij : player_table.bijections) {
      SPIEL_CHECK_EQ(state_bij.size(), player_table.private_hands.size());
      for (const auto& xy : state_bij.x2y) SPIEL_CHECK_EQ(xy.first, xy.second);
      for (const auto& yx : state_bij.y2x) SPIEL_CHECK_EQ(yx.first, yx.second);
    }
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestCreateHandTable();
}

