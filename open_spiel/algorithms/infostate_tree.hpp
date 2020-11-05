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

namespace open_spiel {
namespace algorithms {

template<class Self>
InfostateNode<Self>::InfostateNode(
    const InfostateTree <Self>& tree, Self* parent,
    int incoming_index, InfostateNodeType type,
    const std::string& infostate_string, double terminal_utility,
    double terminal_ch_reach_prob, const State* originating_state)
    : tree_(tree), parent_(parent),
      incoming_index_(incoming_index), type_(type),
      infostate_string_(infostate_string),
      terminal_utility_(terminal_utility),
      terminal_chn_reach_prob_(terminal_ch_reach_prob) {

  // Implications for kTerminalNode
  SPIEL_DCHECK_TRUE(type != kTerminalInfostateNode || originating_state);
  SPIEL_DCHECK_TRUE(type != kTerminalInfostateNode || parent);
  // Implications for kDecisionNode
  SPIEL_DCHECK_TRUE(type != kDecisionInfostateNode || originating_state);
  SPIEL_DCHECK_TRUE(type != kDecisionInfostateNode || parent);
  // Implications for kObservationNode
  SPIEL_DCHECK_TRUE(
      !(type == kObservationInfostateNode && parent
          && parent->type() == kDecisionInfostateNode)
          || (incoming_index >= 0
              && incoming_index < parent->legal_actions().size())
  );

  if (type == kDecisionInfostateNode) {
    legal_actions_ = originating_state->LegalActions(tree_.acting_player());
  }
}

template<class Self>
const std::string& InfostateNode<Self>::infostate_string() const {
  // Avoid working with empty infostate strings.
  // Use Hasinfostate_string() first to check.
  SPIEL_DCHECK_TRUE(has_infostate_string());
  return infostate_string_;
}

template<class Self>
bool InfostateNode<Self>::has_infostate_string() const {
  return infostate_string_ != kFillerInfostate
      && infostate_string_ != kDummyRootNodeInfostate;
}
template<class Self>
double InfostateNode<Self>::terminal_utility() const {
  SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
  return terminal_utility_;
}
template<class Self>
double InfostateNode<Self>::terminal_chance_reach_prob() const {
  SPIEL_CHECK_EQ(type_, kTerminalInfostateNode);
  return terminal_chn_reach_prob_;
}
template<class Self>
const std::vector<Action>& InfostateNode<Self>::legal_actions() const {
  SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
  return legal_actions_;
}
template<class Self>
const std::vector<std::unique_ptr<State>>&
InfostateNode<Self>::corresponding_states() const {
  return corresponding_states_;
}
template<class Self>
const std::vector<double>&
InfostateNode<Self>::corresponding_chance_reach_probs() const {
  return corresponding_ch_reaches_;
}
template<class Self>
Self* InfostateNode<Self>::AddChild(std::unique_ptr<Self> child) {
  SPIEL_CHECK_EQ(child->parent_, this);
  children_.push_back(std::move(child));
  return children_.back().get();
}
template<class Self>
Self* InfostateNode<Self>::GetChild(const std::string& infostate_string) const {
  for (const std::unique_ptr<Self>& child : children_) {
    if (child->infostate_string() == infostate_string) return child.get();
  }
  return nullptr;
}
template<class Self>
const Self*
InfostateNode<Self>::FindNode(const std::string& infostate_lookup) const {
  if (infostate_string_ == infostate_lookup)
    return open_spiel::down_cast<const Self*>(this);
  for (Self& child : *this) {
    if (const Self* node = child.FindNode(infostate_lookup)) {
      return node;
    }
  }
  return nullptr;
}
template<class Self>
std::string InfostateNode<Self>::ToString() const {
  if (!parent_) return "x";
  return absl::StrCat(parent_->ToString(), ",", incoming_index_);
}
template<class Self>
std::string InfostateNode<Self>::ComputeCertificate() const {
  if (type_ == kTerminalInfostateNode) return "{}";

  std::vector<std::string> certificates;
  for (InfostateNode& child : child_iterator()) {
    certificates.push_back(child.ComputeCertificate());
  }
  std::sort(certificates.begin(), certificates.end());

  std::string open, close;
  if (type_ == kDecisionInfostateNode) {
    open = "[";
    close = "]";
  } else if (type_ == kObservationInfostateNode) {
    open = "(";
    close = ")";
  }

  return absl::StrCat(
      open,
      absl::StrJoin(certificates.begin(), certificates.end(), ""),
      close);
}
template<class Self>
void InfostateNode<Self>::Rebalance(int target_depth, int current_depth) {
  SPIEL_DCHECK_LE(current_depth, target_depth);
  if (is_leaf_node() && target_depth != current_depth) {
    // Prepare the chain of dummy observations.
    std::unique_ptr<Self> node = Release();
    Self* node_parent = node->parent();
    int position_in_leaf_parent = node->incoming_index();
    std::unique_ptr<Self> chain_head =
        std::unique_ptr<Self>(new Self(
            /*tree=*/tree_, /*parent=*/nullptr,
            /*incoming_index=*/position_in_leaf_parent,
            kObservationInfostateNode,
            /*infostate_string=*/kFillerInfostate, /*terminal_utility=*/NAN,
            /*terminal_ch_reach_prob=*/NAN, /*originating_state=*/nullptr));
    Self* chain_tail = chain_head.get();
    for (int i = 1; i < target_depth - current_depth; ++i) {
      chain_tail = chain_tail->AddChild(
          std::unique_ptr<Self>(new Self(
              /*tree=*/tree_, /*parent=*/chain_tail,
              /*incoming_index=*/0, kObservationInfostateNode,
              /*infostate_string=*/kFillerInfostate, /*terminal_utility=*/NAN,
              /*terminal_ch_reach_prob=*/NAN,
              /*originating_state=*/nullptr)));
    }
    chain_tail->children_.push_back(nullptr);

    // First put the node to the chain. If we did it in reverse order,
    // i.e chain to parent and then node to the chain, the node would
    // become freed.
    node->SwapParent(std::move(node), /*target=*/chain_tail, 0);
    chain_head->SwapParent(std::move(chain_head), /*target=*/node_parent,
                           position_in_leaf_parent);
  }

  for (std::unique_ptr<Self>& child : children_) {
    child->Rebalance(target_depth, current_depth + 1);
  }
}

template<class Self>
std::unique_ptr<Self> InfostateNode<Self>::Release() {
  SPIEL_DCHECK_TRUE(parent_);
  SPIEL_DCHECK_TRUE(parent_->children_.at(incoming_index_).get() == this);
  return std::move(parent_->children_.at(incoming_index_));
}

template<class Self>
void InfostateNode<Self>::SwapParent(std::unique_ptr<Self> self, Self* target,
                                     int at_index) {
  // This node is still who it thinks it is :)
  SPIEL_DCHECK_TRUE(self.get() == this);
  target->children_.at(at_index) = std::move(self);
  this->parent_ = target;
  this->incoming_index_ = at_index;
}

template<class Node>
InfostateTree<Node>::InfostateTree(
    const std::vector<const State*>& start_states,
    const std::vector<float>& chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player acting_player,
    int max_move_ahead_limit, bool make_balanced)
    : player_(acting_player),
      infostate_observer_(std::move(infostate_observer)),
      root_(CreateRootNode()) {
  SPIEL_CHECK_FALSE(start_states.empty());
  SPIEL_CHECK_EQ(start_states.size(), chance_reach_probs.size());
  SPIEL_CHECK_GE(player_, 0);
  SPIEL_CHECK_LT(player_, start_states[0]->GetGame()->NumPlayers());
  SPIEL_CHECK_TRUE(infostate_observer_->HasString());

  int start_max_move_number = 0;
  for (const State* start_state : start_states) {
    start_max_move_number = std::max(start_max_move_number,
                                     start_state->MoveNumber());
  }

  for (int i = 0; i < start_states.size(); ++i) {
    RecursivelyBuildTree(
        &root_, /*depth=*/1, *start_states[i],
        start_max_move_number + max_move_ahead_limit,
        chance_reach_probs[i]);
  }
  if (make_balanced && !is_balanced()) Rebalance();
  nodes_at_depth_.resize(tree_height() + 1);
  CollectTreeStructure(mutable_root(), 0);
}

template<class Node>
InfostateTree<Node>::InfostateTree(const Game& game, Player acting_player,
                                   int max_move_limit, bool make_balanced)
    : InfostateTree({game.NewInitialState().get()}, /*chance_reach_probs=*/{1.},
                    game.MakeObserver(kInfoStateObsType, {}),
                    acting_player, max_move_limit, make_balanced) {}
template<class Node>
void InfostateTree<Node>::Rebalance() {
  root_.Rebalance(tree_height(), 0);
  is_tree_balanced_ = true;
}

template<class Node>
void InfostateTree<Node>::CollectTreeStructure(Node* node, int depth) {
  nodes_at_depth_[depth].push_back(node);
  for (Node& child : node->child_iterator())
    CollectTreeStructure(&child, depth + 1);
}

template<class Node>
InfostateTree<Node>::LeavesIterator::LeavesIterator(const InfostateTree* tree,
                                                    const Node* current)
    : tree_(tree), current_(current) {
  SPIEL_CHECK_TRUE(current_);
  SPIEL_CHECK_TRUE(current_->is_leaf_node() || current_->is_root_node());
}
template<class Node>
typename InfostateTree<Node>::LeavesIterator&
InfostateTree<Node>::LeavesIterator::operator++() {
  if (!current_->parent())
    SpielFatalError("All leaves have been iterated!");
  SPIEL_CHECK_TRUE(current_->is_leaf_node());
  int child_idx;
  do {  // Find some parent that was not fully traversed.
    SPIEL_DCHECK_LT(current_->incoming_index(),
                    current_->parent()->num_children());
    SPIEL_DCHECK_EQ(current_->parent()->child_at(current_->incoming_index()),
                    current_);
    child_idx = current_->incoming_index();
    current_ = current_->parent();
  } while (current_->parent()
      && child_idx + 1 == current_->num_children());
  // We traversed the whole tree and we got the root node.
  if (!current_->parent() && child_idx + 1 == current_->num_children())
    return *this;
  // Choose the next sibling node.
  current_ = current_->child_at(child_idx + 1);
  // Find the first leaf.
  while (!current_->is_leaf_node()) {
    current_ = current_->child_at(0);
  }
  return *this;
}
template<class Node>
bool InfostateTree<Node>::LeavesIterator::operator==(
    LeavesIterator other) const {
  return current_ == other.current_;
}
template<class Node>
bool InfostateTree<Node>::LeavesIterator::operator!=(
    LeavesIterator other) const { return !(*this == other); }
template<class Node>
const Node&
InfostateTree<Node>::LeavesIterator::operator*() const { return *current_; }

template<class Node>
typename InfostateTree<Node>::LeavesIterator
InfostateTree<Node>::LeavesIterator::begin() const { return *this; }
template<class Node>
typename InfostateTree<Node>::LeavesIterator
InfostateTree<Node>::LeavesIterator::end() const {
  return LeavesIterator(tree_, &(current_->tree().root()));
}
template<class Node>
typename InfostateTree<Node>::LeavesIterator
InfostateTree<Node>::leaves_iterator() const {
  // Find the first leaf.
  const Node* node = &root_;
  while (!node->is_leaf_node()) node = node->child_at(0);
  return LeavesIterator(this, node);
}
template<class Node>
int InfostateTree<Node>::CountLeaves() const {
  int cnt = 0;
  for (const Node& n : leaves_iterator()) cnt++;
  return cnt;
}
template<class Node>
int InfostateTree<Node>::CountLeafCorrespondingHistories() const {
  int cnt = 0;
  for (const Node& n : leaves_iterator())
    cnt += n.CorrespondingStates().size();
  return cnt;
}
template<class Node>
void InfostateTree<Node>::PrintStats() {
  std::cout << "Infostate tree for player " << player_ << ".\n"
            << "Tree height: " << tree_height_ << "\n"
            << "Root branching: " << root().num_children() << "\n"
            << "Number of leaves: " << CountLeaves() << "\n"
            << "Number of leaf corresponding states: "
            << CountLeafCorrespondingHistories() << "\n"
            << "Tree certificate: " << std::endl;
  std::cout << root().ComputeCertificate() << std::endl;
}

template<class Node>
Node InfostateTree<Node>::CreateRootNode() const {
  return Node(
      /*tree=*/*this, /*parent=*/nullptr, /*incoming_index=*/0,
      /*type=*/kObservationInfostateNode,
      /*infostate_string=*/kDummyRootNodeInfostate,
      /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN,
      /*originating_state=*/nullptr);
}
template<class Node>
std::unique_ptr<Node> InfostateTree<Node>::MakeNode(
    Node* parent, InfostateNodeType type, const std::string& infostate_string,
    double terminal_utility, double terminal_ch_reach_prob,
    const State* originating_state) {
  return std::make_unique<Node>(
      *this, parent, parent->num_children(), type,
      infostate_string, terminal_utility, terminal_ch_reach_prob,
      originating_state);
}
template<class Node>
void InfostateTree<Node>::UpdateLeafNode(
    Node* node, const State& state, int leaf_depth, double chance_reach_probs) {
  if (tree_height_ != -1 && is_tree_balanced_) {
    is_tree_balanced_ = tree_height_ == leaf_depth;
  }
  tree_height_ = std::max(tree_height_, leaf_depth);
  node->corresponding_states_.push_back(state.Clone());
  node->corresponding_ch_reaches_.push_back(chance_reach_probs);
}
template<class Node>
void InfostateTree<Node>::RecursivelyBuildTree(
    Node* parent, int depth, const State& state,
    int move_limit, double chance_reach_prob) {
  if (state.IsTerminal())
    return BuildTerminalNode(parent, depth, state, chance_reach_prob);
  else if (state.IsPlayerActing(player_))
    return BuildDecisionNode(parent, depth, state, move_limit,
                             chance_reach_prob);
  else
    return BuildObservationNode(parent, depth, state, move_limit,
                                chance_reach_prob);
}
template<class Node>
void InfostateTree<Node>::BuildTerminalNode(
    Node* parent, int depth,
    const State& state, double chance_reach_prob) {
  const double terminal_utility = state.Returns()[player_];
  Node* terminal_node = parent->AddChild(MakeNode(
      parent, kTerminalInfostateNode,
      infostate_observer_->StringFrom(state, player_), terminal_utility,
      chance_reach_prob, &state));
  UpdateLeafNode(terminal_node, state, depth, chance_reach_prob);
}
template<class Node>
void InfostateTree<Node>::BuildDecisionNode(
    Node* parent, int depth, const State& state,
    int move_limit, double chance_reach_prob) {
  SPIEL_DCHECK_EQ(parent->type(), kObservationInfostateNode);
  std::string info_state = infostate_observer_->StringFrom(state, player_);
  Node* decision_node = parent->GetChild(info_state);
  const bool is_leaf_node = state.MoveNumber() >= move_limit;

  if (decision_node) {
    // The decision node has been already constructed along with children
    // for each action: these are observation nodes.
    // Fetches the observation child and goes deeper recursively.
    SPIEL_DCHECK_EQ(decision_node->type(), kDecisionInfostateNode);

    if (is_leaf_node)  // Do not build deeper.
      return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);

    if (state.IsSimultaneousNode()) {
      const ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[player_].size(); ++i) {
        Node* observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(),
                        kObservationInfostateNode);

        for (Action flat_actions : action_view.fixed_action(player_, i)) {
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }
      }
    } else {
      std::vector<Action> legal_actions = state.LegalActions(player_);
      for (int i = 0; i < legal_actions.size(); ++i) {
        Node* observation_node = decision_node->child_at(i);
        SPIEL_DCHECK_EQ(observation_node->type(),
                        kObservationInfostateNode);
        std::unique_ptr<State> child = state.Child(legal_actions.at(i));
        RecursivelyBuildTree(observation_node, depth + 2, *child,
                             move_limit, chance_reach_prob);
      }
    }
  } else {  // The decision node was not found yet.
    decision_node = parent->AddChild(MakeNode(
        parent, kDecisionInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, &state));

    if (is_leaf_node)  // Do not build deeper.
      return UpdateLeafNode(decision_node, state, depth, chance_reach_prob);

    // Build observation nodes right away after the decision node.
    // This is because the player might be acting multiple times in a row:
    // each time it might get some observations that branch the infostate
    // tree.

    if (state.IsSimultaneousNode()) {
      ActionView action_view(state);
      for (int i = 0; i < action_view.legal_actions[player_].size(); ++i) {
        // We build a dummy observation node.
        // We can't ask for a proper infostate string or an originating state,
        // because such a thing is not properly defined after only a partial
        // application of actions for the sim move state
        // (We need to supply all the actions).
        Node* observation_node = decision_node->AddChild(MakeNode(
            decision_node, kObservationInfostateNode,
            /*infostate_string=*/kFillerInfostate,
            /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN,
            /*originating_state=*/nullptr));

        for (Action flat_actions : action_view.fixed_action(player_, i)) {
          // Only now we can advance the state, when we have all actions.
          std::unique_ptr<State> child = state.Child(flat_actions);
          RecursivelyBuildTree(observation_node, depth + 2, *child,
                               move_limit, chance_reach_prob);
        }

      }
    } else {  // Not a sim move node.
      for (Action a : state.LegalActions()) {
        std::unique_ptr<State> child = state.Child(a);
        Node* observation_node = decision_node->AddChild(MakeNode(
            decision_node, kObservationInfostateNode,
            infostate_observer_->StringFrom(*child, player_),
            /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN,
            child.get()));
        RecursivelyBuildTree(observation_node, depth + 2, *child,
                             move_limit, chance_reach_prob);
      }
    }
  }
}
template<class Node>
void InfostateTree<Node>::BuildObservationNode(
    Node* parent, int depth, const State& state,
    int move_limit, double chance_reach_prob) {
  SPIEL_DCHECK_TRUE(state.IsChanceNode() || !state.IsPlayerActing(player_));
  const bool is_leaf_node = state.MoveNumber() >= move_limit;
  const std::string info_state =
      infostate_observer_->StringFrom(state, player_);

  Node* observation_node = parent->GetChild(info_state);
  if (!observation_node) {
    observation_node = parent->AddChild(MakeNode(
        parent, kObservationInfostateNode, info_state,
        /*terminal_utility=*/NAN, /*chance_reach_prob=*/NAN, &state));
  }
  SPIEL_DCHECK_EQ(observation_node->type(), kObservationInfostateNode);

  if (is_leaf_node)  // Do not build deeper.
    return UpdateLeafNode(observation_node, state, depth, chance_reach_prob);

  if (state.IsChanceNode()) {
    for (std::pair<Action, double> action_prob : state.ChanceOutcomes()) {
      std::unique_ptr<State> child = state.Child(action_prob.first);
      RecursivelyBuildTree(observation_node, depth + 1, *child,
                           move_limit,
                           chance_reach_prob * action_prob.second);
    }
  } else {
    for (Action a : state.LegalActions()) {
      std::unique_ptr<State> child = state.Child(a);
      RecursivelyBuildTree(observation_node, depth + 1, *child,
                           move_limit, chance_reach_prob);
    }
  }
}

}  // namespace algorithms
}  // namespace open_spiel
