
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


#include "open_spiel/papers_with_code/1906.06412.value_functions/hand_table.h"

#include "open_spiel/utils/format_observation.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;


size_t HandTable::num_hands() const { return private_hands.size(); }

size_t HandTable::hand_index(const Observation& hand) {
  auto it = std::find(private_hands.begin(), private_hands.end(), hand);
  if (it == private_hands.end()) {
    private_hands.push_back(hand);
    return private_hands.size() - 1;
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

bool AllInfoStatesHaveDistinctHands(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    Player pl, const dlcfr::LeafPublicState& state) {
  const std::vector<const InfostateNode*>& info_states = state.leaf_nodes[pl];
  std::unordered_map<Observation, const InfostateNode*> hands_for_infostates;

  Observation hand(game, hand_observer);
  for (const InfostateNode* info_state : info_states) {
    const State& some_state = *info_state->corresponding_states().at(0);
    hand.SetFrom(some_state, pl);
    if (hands_for_infostates.find(hand) == hands_for_infostates.end()) {
      hands_for_infostates[hand] = info_state;
    } else {
      std::cerr << "Not all hands are unique in public state: \n"
                << ObservationToString(state.public_tensor) << "\n"
                << "Printing out the hands.\n-----\n";
      for (const auto& [hand, info_state] : hands_for_infostates) {
        std::cerr << "Infostate string: " << info_state->infostate_string() << "\n"
                  << "Hand observation: " << ObservationToString(hand) << "\n"
                  << "Some history in infostate: "
                  << info_state->corresponding_states()[0]->HistoryString() << "\n"
                  << "-----\n";
      }
      std::cerr << "Offending infostate: \n"
                << "Infostate string: " << info_state->infostate_string() << "\n"
                << "Hand observation: " << ObservationToString(hand) << "\n"
                << "Some history in infostate: "
                << info_state->corresponding_states()[0]->HistoryString() << "\n"
                << "-----\n";
      return false;
    }
  }
  return true;
}

bool AllStatesHaveSameHands(const Observation& expected_hand, Player player,
                            const std::vector<std::unique_ptr<State>>& states) {
  Observation actual_hand(expected_hand);
  for (const std::unique_ptr<State>& state : states) {
    actual_hand.SetFrom(*state, player);
    if (actual_hand != expected_hand) {
      return false;
    }
  }
  return true;
}

std::vector<HandTable> CreateHandTables(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves) {
  std::vector<HandTable> tables{public_leaves.size(), public_leaves.size()};
  Observation hand(game, hand_observer);
  for (int state_idx = 0; state_idx < public_leaves.size(); ++state_idx) {
    const dlcfr::LeafPublicState& state = public_leaves[state_idx];
    // Terminal states are not handled by non-terminal leaf evaluators,
    // so we don't need to create hand table for them.
    if (state.IsTerminal()) {
      continue;
    }

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_DCHECK_TRUE(  // Holds within public state.
          AllInfoStatesHaveDistinctHands(game, hand_observer, pl, state));

      for (int tree_idx = 0;
           tree_idx < state.leaf_nodes[pl].size(); ++tree_idx) {
        const InfostateNode* node = state.leaf_nodes[pl][tree_idx];
        const State& some_state = *node->corresponding_states().at(0);
        hand.SetFrom(some_state, pl);

        SPIEL_DCHECK_TRUE(  // Should hold for all states within an infostate.
            AllStatesHaveSameHands(hand, pl, node->corresponding_states()));

        size_t net_idx = tables[pl].hand_index(hand);
        tables[pl].bijections[state_idx].put({tree_idx, net_idx});
      }
    }
  }
  return tables;
}

void DebugPrintHandTables(const std::vector<HandTable>& tables) {
  SPIEL_CHECK_EQ(tables.size(), 2);
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# List of private hands for player " << pl << "\n";
    const HandTable& table = tables[pl];
    for (int i = 0; i < table.private_hands.size(); ++i) {
      std::cout << "#   private_hand[" << i << "]:\n#      "
                << ObservationToString(table.private_hands[i], "\n#      ")
                << "\n";
    }

    std::cout << "# List of bijections (tree <-> net) for player "
              << pl << "\n";
    for (size_t i = 0; i < table.bijections.size(); ++i) {
      std::cout << "#  Public state " << i << "\n";
      const std::map<size_t, size_t>& tree_to_net =
          table.bijections[i].tree_to_net();
      for (auto&[key, val] : tree_to_net) {
        std::cout << "#   " << key << " -> " << val << "\n";
      }
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

