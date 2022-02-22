//
// Created by milecdav on 07.02.22.
//

#ifndef OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_1906_06412_VALUE_FUNCTIONS_POKER_DATA_H_
#define OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_1906_06412_VALUE_FUNCTIONS_POKER_DATA_H_

#include <open_spiel/spiel.h>
#include <open_spiel/games/universal_poker.h>
#include "vector"

namespace open_spiel {
namespace algorithms {

struct PokerData {
  void CreateHandsRecursion(std::vector<std::vector<int>> &hands, const std::vector<int> &hand, int depth) {
    if (depth >= cards_in_hand_) {
      hands.push_back(hand);
      return;
    }
    int last_card = -1;
    if (!hand.empty()) {
      last_card = hand.back();
    }
    for (int next_card = last_card + 1; next_card < num_cards_; next_card++) {
      std::vector<int> new_hand = hand;
      new_hand.push_back(next_card);
      CreateHandsRecursion(hands, new_hand, depth + 1);
    }
  }

  std::vector<std::vector<int>> CreateHands() {
    std::vector<std::vector<int>> hands = std::vector<std::vector<int>>();
    std::vector<int> hand = std::vector<int>();
    CreateHandsRecursion(hands, hand, 0);
    return hands;
  }

  explicit PokerData(const State &state) {
    auto poker_game = down_cast<const universal_poker::UniversalPokerGame &>(*state.GetGame()).GetACPCGame();
    num_suits_ = poker_game->NumSuitsDeck();
    num_ranks_ = poker_game->NumRanksDeck();
    num_cards_ = num_suits_ * num_ranks_;
    cards_in_hand_ = poker_game->GetNbHoleCardsRequired();
    num_hands_ = 1;
    for (int i = 0; i < cards_in_hand_; i++) {
      num_hands_ = num_hands_ * (num_cards_ - i) / (i + 1);
    }

    for (int card = 0; card < num_cards_; card++) {
      card_to_hands_.emplace(card, std::vector<int>());
    }
    auto hands = CreateHands();
    for (int hand_index = 0; hand_index < hands.size(); hand_index++) {
      for (int card : hands[hand_index]) {
        card_to_hands_[card].push_back(hand_index);
      }
      hand_to_cards_.emplace(hand_index, hands[hand_index]);
    }

    auto initial_state = state.GetGame()->NewInitialState();
    for (Action action : initial_state->LegalActions()) {
      card_mask_.push_back(initial_state->Child(action)->InformationStateString(0).substr(23, 2));
    }
  }
  int num_cards_;
  int num_hands_;
  int num_suits_;
  int num_ranks_;
  int cards_in_hand_;
  std::vector<std::string> card_mask_;
  std::unordered_map<int, std::vector<int>> hand_to_cards_;
  std::unordered_map<int, std::vector<int>> card_to_hands_;

  std::string to_string_basic() const {
    std::ostringstream ss;
    ss << "Num cards: " << num_cards_ << "\n";
    ss << "Num hands: " << num_hands_ << "\n";
    ss << "Num suits: " << num_suits_ << "\n";
    ss << "Num ranks: " << num_ranks_ << "\n";
    ss << "Cards in hand: " << cards_in_hand_ << "\n";
    ss << "Card mask: " << card_mask_ << "\n";

    return ss.str();
  }

  std::string to_string_full() const {
    std::ostringstream ss;
    ss << to_string_basic();

    ss << "Hand to cards: ";
    for (auto const &pair: hand_to_cards_) {
      ss << "{" << pair.first << ": " << pair.second << "}";
    }
    ss << "\n";

    ss << "Card to hands: ";
    for (auto const &pair: card_to_hands_) {
      ss << "{" << pair.first << ": " << pair.second << "}";
    }
    ss << "\n";

    return ss.str();
  }
};
}
}

#endif //OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_1906_06412_VALUE_FUNCTIONS_POKER_DATA_H_
