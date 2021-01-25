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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_RANGE_TABLE_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_RANGE_TABLE_

#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace papers_with_code {

// -- Range table --------------------------------------------------------------

struct RangeTable {
  // Bijection between ranges coming from infostate tree (x)
  // and the input position (y), called also hand, for each public state.
  // This is used for encoding NN inputs (resp. outputs).
  // Forward:  tree  -> input positions
  // Backward: output positions -> tree
  std::vector<BijectiveContainer<size_t>> bijections;

  // List all possible private observations ("hands") for each player.
  // Their vector indices represent the input position for each public state.
  std::vector<Observation> private_hands;

  RangeTable(int num_public_states) : bijections(num_public_states) {}
  size_t largest_range() const;
  size_t hand_index(const Observation& obs);
};

std::vector<RangeTable> CreateRangeTables(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves);

void DebugPrintRangeTables(const std::vector<RangeTable>& tables);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_RANGE_TABLE_
