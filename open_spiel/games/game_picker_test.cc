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

#include "open_spiel/games/game_picker.h"

#include "open_spiel/game_parameters.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace game_picker {
namespace {

namespace testing = open_spiel::testing;
// C(/*for_player=*/0) << g0 << g1 ^ ( C(Player{0}) << g0 << g1);
void BasicGamePickerTests() {
  std::vector<std::shared_ptr<const Game>> games = {
      LoadGame("matrix_mp"), LoadGame("kuhn_poker")
  };
  auto picker = MakeGamePicker(games);
  testing::ChanceOutcomesTest(*picker);
  testing::RandomSimTest(*picker, /*num_sims=*/100, /*serialize=*/false);
}

}  // namespace
}  // namespace game_picker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::game_picker::BasicGamePickerTests();
}
