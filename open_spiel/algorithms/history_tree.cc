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

#include "open_spiel/algorithms/history_tree.h"

#include <cmath>
#include <limits>
#include <unordered_set>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace {

std::unique_ptr<HistoryNode> RecursivelyBuildGameTree(
    std::unique_ptr<State> state,
    std::unordered_map<std::string, HistoryNode*>* state_to_node) {
  std::unique_ptr<HistoryNode> node(
      new HistoryNode(std::move(state)));
  if (state_to_node == nullptr) SpielFatalError("state_to_node is null.");
  (*state_to_node)[node->GetHistory()] = node.get();
  State* state_ptr = node->GetState();
  switch (node->GetType()) {
    case StateType::kChance: {
      double probability_sum = 0;
      for (const auto& outcome_and_prob : state_ptr->ChanceOutcomes()) {
        Action outcome = outcome_and_prob.first;
        double prob = outcome_and_prob.second;
        std::unique_ptr<State> child = state_ptr->Child(outcome);
        if (child == nullptr) {
          SpielFatalError("Can't add child; child is null.");
        }
        probability_sum += prob;
        std::unique_ptr<HistoryNode> child_node = RecursivelyBuildGameTree(
            std::move(child), state_to_node);
        node->AddChild(outcome, {prob, std::move(child_node)});
      }
      SPIEL_CHECK_FLOAT_EQ(probability_sum, 1.0);
      break;
    }
    case StateType::kDecision: {
      for (const auto& action : state_ptr->LegalActions()) {
        std::unique_ptr<HistoryNode> child_history = RecursivelyBuildGameTree(
            std::move(state_ptr->Child(action)), state_to_node);
        // Note: The probabilities here are meaningless if state.CurrentPlayer()
        // != player_id, as we'll be getting the probabilities from the policy
        // during the call to Value. For state.CurrentPlayer() == player_id,
        // the probabilities are equal to 1. for every action as these are
        // *counter-factual* probabilities, which ignore the probability of
        // the player that we are playing as.
        node->AddChild(action, {1., std::move(child_history)});
      }
      break;
    }
    case StateType::kTerminal: {
      // As we assign terminal utilities to node.value in the constructor of
      // HistoryNode, we don't have anything to do here.
      break;
    }
  }
  return node;
}

}  // namespace

HistoryNode::HistoryNode(std::unique_ptr<State> game_state)
    : state_(std::move(game_state)),
      action_view_(*state_),
      history_(state_->HistoryString()),
      type_(state_->GetType()) {
  infostates_.reserve(state_->NumPlayers());
  for (int pl = 0; pl < state_->NumPlayers(); ++pl) {
    infostates_.push_back(state_->InformationStateString(pl));
  }

  if (type_ == StateType::kTerminal) {
    terminal_utilities_ = state_->Returns();
  }
}

void HistoryNode::AddChild(
    Action outcome, std::pair<double, std::unique_ptr<HistoryNode>> child) {
  const auto legal_actions = LegalActions();
  SPIEL_CHECK_TRUE(
      std::find(legal_actions.begin(), legal_actions.end(), outcome)
      != legal_actions.end());
  SPIEL_CHECK_PROB(child.first);
  SPIEL_CHECK_TRUE(child.second);
  SPIEL_CHECK_LT(children_.size(), legal_actions.size());
  SPIEL_CHECK_EQ(legal_actions[children_.size()], outcome);
  children_.push_back(std::move(child));
}

std::pair<double, HistoryNode*> HistoryNode::GetChild(Action action) {
  const auto legal_actions = LegalActions();
  SPIEL_CHECK_EQ(legal_actions.size(), children_.size());

  const auto it = std::find(legal_actions.begin(),
                            legal_actions.end(), action);
  SPIEL_CHECK_TRUE(it != legal_actions.end());
  const size_t child_idx = std::distance(legal_actions.begin(), it);

  std::pair<double, HistoryNode*> child = std::make_pair(
      children_[child_idx].first, children_[child_idx].second.get());
  SPIEL_CHECK_TRUE(child.second);
  return child;
}

std::vector<Action> HistoryNode::GetChildActions() const {
  return LegalActions();
}

std::vector<Action> HistoryNode::LegalActions() const {
  if (state_->IsSimultaneousNode()) {
    return action_view_.flat_joint_actions().as_vector();
  } else {
    return action_view_.legal_actions[0];
  }
}

HistoryNode* HistoryTree::GetByHistory(const std::string& history) {
  HistoryNode* node = state_to_node_[history];
  SPIEL_CHECK_TRUE(node);
  return node;
}

std::vector<std::string> HistoryTree::GetHistories() {
  std::vector<std::string> histories;
  histories.reserve(state_to_node_.size());
  for (const auto& kv : state_to_node_) {
    histories.push_back(kv.first);
  }
  return histories;
}

HistoryTree::HistoryTree(std::unique_ptr<State> state) {
  root_ = RecursivelyBuildGameTree(std::move(state), &state_to_node_);
}

ActionsAndProbs GetSuccessorsWithProbs(const State& state,
                                       Player best_responder,
                                       const Policy* policy) {
  if (state.CurrentPlayer() == best_responder) {
    ActionsAndProbs state_policy;
    for (const auto& legal_action : state.LegalActions()) {
      // Counterfactual reach probabilities exclude the player's
      // actions, hence return probability 1.0 for every action.
      state_policy.push_back({legal_action, 1.});
    }
    return state_policy;
  } else if (state.IsChanceNode()) {
    return state.ChanceOutcomes();
  } else {
    // Finally, we look at the policy we are finding a best response to, and
    // get our probabilities from there.
    auto state_policy = policy->GetStatePolicy(state);
    if (state_policy.empty()) {
      SpielFatalError(state.InformationStateString() + " not found in policy.");
    }
    return state_policy;
  }
}

// TODO(author1): If this is a bottleneck, it should be possible
// to pass the probabilities-so-far into the call, and get everything right
// the first time, without recursion. The recursion is simpler, however.
std::vector<std::pair<std::unique_ptr<State>, double>> DecisionNodes(
    const State& parent_state, Player best_responder, const Policy* policy) {
  // If the state is terminal, then there are no more decisions to be made,
  // so we're done.
  if (parent_state.IsTerminal()) return {};

  std::vector<std::pair<std::unique_ptr<State>, double>> states_and_probs;
  // We only consider states where the best_responder is making a decision.
  if (parent_state.CurrentPlayer() == best_responder) {
    states_and_probs.push_back({parent_state.Clone(), 1.});
  }
  ActionsAndProbs actions_and_probs =
      GetSuccessorsWithProbs(parent_state, best_responder, policy);
  for (open_spiel::Action action : parent_state.LegalActions()) {
    std::unique_ptr<State> child = parent_state.Child(action);

    // We recurse here to get the correct probabilities for all children.
    // This could probably be done in a cleaner, more performant way, but as
    // this is only done once, at the start of the exploitability calculation,
    // this is fine for now.
    std::vector<std::pair<std::unique_ptr<State>, double>> children =
        DecisionNodes(*child, best_responder, policy);
    const double prob = GetProb(actions_and_probs, action);
    SPIEL_CHECK_GE(prob, 0);
    for (auto& state_and_prob : children) {
      states_and_probs.push_back(
          {std::move(state_and_prob.first),
           // We weight the child probabilities by the probability of taking
           // the action that would lead to them.
           prob * state_and_prob.second});
    }
  }
  return states_and_probs;
}

std::unordered_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
GetAllInfoSets(std::unique_ptr<State> state, Player best_responder,
               const Policy* policy, HistoryTree* tree) {
  std::unordered_map<std::string, std::vector<std::pair<HistoryNode*, double>>>
      infosets;
  // We only need decision nodes, as there's no decision to be made at chance
  // nodes (we randomly sample from the different outcomes there).
  std::vector<std::pair<std::unique_ptr<State>, double>> states_and_probs =
      DecisionNodes(*state, best_responder, policy);
  infosets.reserve(states_and_probs.size());
  for (const auto& state_and_prob : states_and_probs) {
    // We look at each decision from the perspective of the best_responder.
    std::string infostate =
        state_and_prob.first->InformationStateString(best_responder);
    infosets[infostate].push_back(
        {tree->GetByHistory(state_and_prob.first->HistoryString()),
         state_and_prob.second});
  }
  return infosets;
}

}  // namespace algorithms
}  // namespace open_spiel
