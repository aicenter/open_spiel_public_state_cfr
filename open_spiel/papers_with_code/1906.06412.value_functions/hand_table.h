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
#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace papers_with_code {

// A bijection within the scope of a public state. This is a mapping between
// LeafPublicState::ranges coming from the tree (x) and the input position
// of the neural network (y) which is assigned according to player's private
// hands across all public states of the trunk.
// This is used for encoding NN inputs (resp. outputs).
struct HandMapping : BijectiveContainer<size_t> {
  const std::map<size_t, size_t>& tree_to_net() const { return x2y; }
  const std::map<size_t, size_t>& net_to_tree() const { return y2x; }
};

// Store all possible private hands within a trunk for one player.
struct HandTable {
  // Hand mapping per public state.
  std::vector<HandMapping> bijections;

  // List all possible private observations ("hands") for the player.
  // Their position in the vector is the same as their network's input position.
  std::vector<Observation> private_hands;

  HandTable(int num_public_states) : bijections(num_public_states) {}
  size_t num_hands() const;
  size_t hand_index(const Observation& obs);
  size_t hand_tensor_size() const;
  const Observation& hand_observation_at(int public_id, int infostate_id) const;
};

std::vector<HandTable> CreateHandTables(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves);

void DebugPrintHandTables(const std::vector<HandTable>& tables);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_HAND_TABLE_
