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

#ifndef OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_
#define OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/action_view.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

// TODO(author1): See if it's possible to remove any fields here.
// Stores all information relevant to exploitability calculation for each
// history in the game.
class HistoryNode {
 public:
  HistoryNode(std::unique_ptr<State> game_state);

  State* GetState() const { return state_.get(); }

  const std::string& GetInfoState() const {
    SPIEL_CHECK_GE(state_->CurrentPlayer(), 0);
    return infostates_[state_->CurrentPlayer()];
  }

  const std::string& GetInfoState(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state_->NumPlayers());
    return infostates_[player];
  }

  const std::string& GetHistory() const { return history_; }

  const StateType& GetType() const { return type_; }

//  double GetValue() const {
//    SpielFatalError("Obsolete, please use GetUtility(Player)");
//  }
  double GetUtility(Player player) const {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state_->NumPlayers());
    return terminal_utilities_[player];
  }

  Action NumChildren() const { return children_.size(); }

  void AddChild(Action outcome, std::pair<
      /*chance_probability=*/double, std::unique_ptr<HistoryNode>> child);

  std::vector<Action> GetChildActions() const;

  std::vector<Action> LegalActions() const;

  std::pair<double, HistoryNode*> GetChild(Action action) const;

  const ActionView& action_view() const { return action_view_; }

 private:
  std::unique_ptr<State> state_;
  std::vector<std::string> infostates_;
  std::string history_;
  StateType type_;
  std::vector<double> terminal_utilities_;

  ActionView action_view_;
  std::vector<std::pair<
    /*chance_probability=*/double,
    std::unique_ptr<HistoryNode>>> children_;
};

// History here refers to the fact that we're using histories- i.e.
// representations of all players private information in addition to the public
// information- as the underlying abstraction. Other trees are possible, such as
// PublicTrees, which use public information as the base abstraction, and
// InformationStateTrees, which use all of the information available to one
// player as the base abstraction.
class HistoryTree {
 public:
  // Builds a tree of histories.
  HistoryTree(std::unique_ptr<State> state);

  HistoryTree(const Game& game)
      : HistoryTree(game.NewInitialState()) {}

  HistoryNode* Root() { return root_.get(); }
  const HistoryNode& Root() const { return *root_; }

  HistoryNode* GetByHistory(const std::string& history);

  // For test use only.
  std::vector<std::string> GetHistories();

  Action NumHistories() { return state_to_node_.size(); }

 private:
  std::unique_ptr<HistoryNode> root_;

  // Maps histories to HistoryNodes.
  std::unordered_map<std::string, HistoryNode*> state_to_node_;
};

// Returns a map of infostate strings to a vector of history nodes with
// corresponding counter-factual probabilities, where counter-factual
// probabilities are calculatd using the passed policy for the opponent's
// actions, a probability of 1 for all of the best_responder's actions, and the
// natural chance probabilty for all change actions. We return all infosets
// (i.e. all sets of history nodes grouped by infostate) for the sub-game rooted
// at state, from the perspective of the player with id best_responder.
std::unordered_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
GetAllInfoSets(std::unique_ptr<State> state, Player best_responder,
               const Policy* policy, HistoryTree* tree);

// For a given state, returns all successor states with accompanying
// counter-factual probabilities.
ActionsAndProbs GetSuccessorsWithProbs(const State& state,
                                       Player best_responder,
                                       const Policy* policy);

// Returns all decision nodes, with accompanying counter-factual probabilities,
// for the sub-game rooted at parent_state.
std::vector<std::pair<std::unique_ptr<State>, double>> DecisionNodes(
    const State& parent_state, Player best_responder, const Policy* policy);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_HISTORY_TREE_H_
