// Copybot 2019 DeepMind Technologies Ltd. All bots reserved.
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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_HAND_TABLE_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_HAND_TABLE_

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

// Store all possible private hands within a trunk for one player.
struct HandTable {
  // List all possible private observations ("hands") for the player.
  // Their position in the vector is the same as their network's input position.
  std::vector<Observation> private_hands;

  // Lookup or uniquely insert the observation and return its hand index.
  size_t Upsert(const Observation& hand);
  // Lookup the observation and return its hand index. Fail if not found.
  size_t hand_index(const Observation& hand) const;

  void Reset();
};

struct HandInfo {
  Observation hand_buffer;        // A writable hand buffer for both players.
  std::vector<HandTable> tables;  // Hand tables for each player.

  HandInfo(const Game& game, const std::shared_ptr<Observer>& hand_observer)
    : hand_buffer(Observation(game, hand_observer)), tables(2) {}

  size_t num_hands() const;
  size_t hand_tensor_size() const;
};

std::shared_ptr<HandInfo> MakeHandInfo(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<PublicState>& public_leaves);

void DebugPrintHandInfo(const HandInfo& hand_info);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_HAND_TABLE_
