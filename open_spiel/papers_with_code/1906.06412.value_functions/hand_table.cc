
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



size_t HandTable::Upsert(const Observation& hand) {
  auto it = std::find(private_hands.begin(), private_hands.end(), hand);
  if (it == private_hands.end()) {
    private_hands.push_back(hand);
    return private_hands.size() - 1;
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

size_t HandTable::hand_index(const Observation& hand) const {
  auto it = std::find(private_hands.begin(), private_hands.end(), hand);
  if (it == private_hands.end()) {
    SpielFatalError("Hand not found!");
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

size_t HandInfo::num_hands() const {
  int hands = 0;
  for (const auto & table : tables) {
    hands += table.private_hands.size();
  }
  return hands;
}

size_t HandInfo::hand_tensor_size() const {
  return hand_buffer.Tensor().size();
}

bool AllInfoStatesHaveDistinctHands(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    Player pl, const PublicState& state) {
  const std::vector<const algorithms::InfostateNode*>& info_states = state.nodes[pl];
  std::unordered_map<Observation, const algorithms::InfostateNode*> hands_for_infostates;

  Observation hand(game, hand_observer);
  for (const algorithms::InfostateNode* info_state : info_states) {
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

std::shared_ptr<HandInfo> MakeHandInfo(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    const std::vector<PublicState>& public_leaves) {
  auto hand_info = std::make_shared<HandInfo>(game, hand_observer);
  Observation& hand = hand_info->hand_buffer;

  for (int state_idx = 0; state_idx < public_leaves.size(); ++state_idx) {
    const PublicState& state = public_leaves[state_idx];
    // Terminal states are not handled by non-terminal leaf evaluators,
    // so we don't need to create hand table for them.
    if (state.IsTerminal()) {
      continue;
    }

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_DCHECK_TRUE(  // Holds within public state.
          AllInfoStatesHaveDistinctHands(game, hand_observer, pl, state));

      for (int tree_idx = 0;
           tree_idx < state.nodes[pl].size(); ++tree_idx) {
        const algorithms::InfostateNode* node = state.nodes[pl][tree_idx];
        const State& some_state = *node->corresponding_states().at(0);
        hand.SetFrom(some_state, pl);

        SPIEL_DCHECK_TRUE(  // Should hold for all states within an infostate.
            AllStatesHaveSameHands(hand, pl, node->corresponding_states()));

        hand_info->tables[pl].Upsert(hand);
      }
    }
  }
  return hand_info;
}

void DebugPrintHandInfo(const HandInfo& hand_info) {
  SPIEL_CHECK_EQ(hand_info.tables.size(), 2);
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# List of private hands for player " << pl << "\n";
    const HandTable& table = hand_info.tables[pl];
    for (int i = 0; i < table.private_hands.size(); ++i) {
      std::cout << "#   private_hand[" << i << "]:\n#      "
                << ObservationToString(table.private_hands[i], "\n#      ")
                << "\n";
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

