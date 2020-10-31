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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/action_view.h"

// This file contains a container data structure that builds an infostate tree
// for specified acting player, starting at some histories in the game.
//
// As infostate nodes may require to contain arbitrary values, they are
// implemented with curiously recurring template pattern (CRTP). The common
// usage for CFR is provided under a CFRTree and CFRNode respectively.
//
// The identification of infostates is based on strings from an information
// state observer.

namespace open_spiel {
namespace algorithms {

// We use the nomenclature from the [Predictive CFR] paper.
//
// In _decision nodes_, the acting player selects actions.
// The _observation nodes_ can correspond to State that is a chance node,
// opponent's node, but importantly, also to the acting player's node,
// as the player may have discovered something as a result of its action
// in the previous decision node. Additionally, we use _terminal nodes_,
// which correspond to a single State terminal history.
//
// The terminal nodes store player's utility as well as cumulative chance reach
// probability.
//
// [Predictive CFR] https://arxiv.org/pdf/2007.14358.pdf.
enum InfostateNodeType {
  kDecisionInfostateNode,
  kObservationInfostateNode,
  kTerminalInfostateNode
};

// Representing the game via infostates leads actually to a graph structure
// of a forest (a collection of trees). We trivially make it into a proper tree
// by introducing a "dummy" root node, which we set as an observation node.
// It could be interpreted as "the player observes the start of the game".
// This is the infostate string for this node.
constexpr char* kDummyRootNodeInfostate = "(dummy root)";

// Sometimes we need to create infostate nodes that do not have a corresponding
// game State, and therefore we cannot retrieve its string representation.
// This use case is for simultaneous move games or to enable rebalancing game
// trees.
constexpr char* kFillerInfostate = "(filler node)";

// Forward declarations.
template<class Node> class InfostateTree;
template<class Self> class InfostateNode;

template<class Self>
class InfostateNode {
 public:
  InfostateNode(
      const InfostateTree<Self>& tree, Self* parent, int incoming_index,
      InfostateNodeType type, const std::string& infostate_string,
      double terminal_utility, double terminal_ch_reach_prob,
      const State* originating_state);
  InfostateNode(InfostateNode&&) noexcept = default;
  virtual ~InfostateNode() = default;

  const InfostateTree<Self>& Tree() const { return tree_; }
  Self* Parent() const { return parent_; }
  int IncomingIndex() const { return incoming_index_; }
  const InfostateNodeType& Type() const { return type_; }
  bool IsLeafNode() const { return children_.empty(); }
  bool IsRootNode() const { return !parent_; }
  const std::string& InfostateString() const;
  bool HasInfostateString() const;
  double TerminalUtility() const;
  double TerminalChanceReachProb() const;
  absl::Span<const Action> LegalActions() const;
  const std::vector<std::unique_ptr<State>>& CorrespondingStates() const;
  const std::vector<double>& CorrespondingChanceReaches() const;
  Self* AddChild(std::unique_ptr<Self> child);
  Self* GetChild(const std::string& infostate_string) const;
  Self* ChildAt(int i) const { return children_.at(i).get(); }
  int NumChildren() const { return children_.size(); }
  const Self* FindNode(const std::string& infostate_lookup) const;

  // Intended only for debug purposes.
  std::string ToString() const;
  // Compute subtree certificate (string representation) for easy comparison.
  std::string ComputeCertificate() const;

  // Iterate over children and expose references to the children
  // (instead of unique_ptrs).
  class ChildIterator {
    int pos_;
    const std::vector<std::unique_ptr<Self>>& children_;
   public:
    ChildIterator(const std::vector<std::unique_ptr<Self>>& children,
                  int pos = 0) : pos_(pos), children_(children) {}
    ChildIterator& operator++() { pos_++; return *this; }
    bool operator==(ChildIterator other) const { return pos_ == other.pos_; }
    bool operator!=(ChildIterator other) const { return !(*this == other); }
    Self& operator*() { return *children_[pos_]; }
    ChildIterator begin() const { return *this; }
    ChildIterator end() const {
      return ChildIterator(children_, children_.size());
    }
  };
  ChildIterator child_iterator() const { return ChildIterator(children_); }

  // Make sure that the subtree ends at the requested target depth by inserting
  // dummy observation nodes with one outcome.
  void Rebalance(int target_depth, int current_depth);

 private:
  // Get the unique_ptr for this node. The usage is intended only for tree
  // balance manipulation.
  std::unique_ptr<Self> Release();

  // Change the parent of this node by inserting it at at index
  // of the new parent. The node at the existing position will be freed.
  // We pass the unique ptr of itself, because calling Release might be
  // undefined: the node we want to swap a parent for can be root of a subtree.
  void SwapParent(std::unique_ptr<Self> self, Self* target, int at_index);

 protected:
  // Needed for adding corresponding_states_ during tree traversal.
  friend class InfostateTree<Self>;

  // Reference to the tree that this node belongs to. This reference has a valid
  // lifetime, as all the nodes are recursively owned by their parents, and the
  // root is owned by the tree.
  const InfostateTree<Self>& tree_;
  // Pointer to the parent node.
  // This is not const so that we can change it when we SwapParent().
  Self* parent_;
  // Position of this node in the parent's children, i.e. it should hold that
  //   parent_->children_.at(incoming_index_).get() == this.
  // For decision nodes this corresponds also to the
  //   State::LegalActions(player_).at(incoming_index_)
  // This is not const so that we can change it when we SwapParent() for tree
  // manipulation.
  int incoming_index_;
  // Type of the node.
  const InfostateNodeType type_;
  // Identifier of the infostate.
  const std::string infostate_string_;
  // Utility of terminal state corresponding to a terminal infostate node.
  const double terminal_utility_;
  const double terminal_chn_reach_prob_;
  // Only for decision nodes.
  std::vector<Action> legal_actions_;
  // Children infostate nodes. Notice the node owns its children.
  std::vector<std::unique_ptr<Self>> children_;
  // Optionally store States that correspond to this infostate node.
  std::vector<std::unique_ptr<State>> corresponding_states_;
  std::vector<double> corresponding_ch_reaches_;
};

template<class Node>
class InfostateTree final {
 public:
  // Creates an infostate tree for a player based on the initial state
  // of the game, up to some move limit.
  InfostateTree(const Game& game, Player acting_player,
                int max_move_limit = 1000, bool make_balanced = true);

  // Creates an infostate tree for a player based on some start states,
  // up to some move limit from the deepest start state.
  InfostateTree(
      absl::Span<const State*> start_states,
      absl::Span<const float> chance_reach_probs,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_move_ahead_limit = 1000, bool make_balanced = true);

  const Node& Root() const { return root_; }
  Node* MutableRoot() { return &root_; }
  Player GetPlayer() const { return player_; }
  const Observer& GetObserver() const { return *infostate_observer_; }
  int TreeHeight() const { return tree_height_; }
  bool IsBalanced() const { return is_tree_balanced_; }

  // Makes sure that all tree leaves are at the same height.
  // It inserts a linked list of dummy observation nodes with appropriate length
  // to balance all the leaves. In the worst case this makes the tree about 2x
  // as large (in the number of nodes).
  void Rebalance();

  // Iterate over all leaves.
  class LeavesIterator {
    const InfostateTree* tree_;
    const Node* current_;
   public:
    LeavesIterator(const InfostateTree* tree, const Node* current);
    LeavesIterator& operator++();
    bool operator==(LeavesIterator other) const;
    bool operator!=(LeavesIterator other) const;
    const Node& operator*() const;
    LeavesIterator begin() const;
    LeavesIterator end() const;
  };
  LeavesIterator leaves_iterator() const;
  // Expensive. Use only for debugging.
  int CountLeaves() const;
  int CountLeafCorrespondingHistories() const;
  void PrintStats();

 private:
  const Player player_;
  const std::shared_ptr<Observer> infostate_observer_;
  Node root_;

  // A value that helps to determine if the tree is balanced.
  int tree_height_ = -1;
  // We call a tree balanced if all leaves are in the same depth.
  bool is_tree_balanced_ = true;

  Node CreateRootNode() const;

  // Utility function whenever we create a new node for the tree.
  std::unique_ptr<Node> MakeNode(
      Node* parent, InfostateNodeType type, const std::string& infostate_string,
      double terminal_utility, double terminal_ch_reach_prob,
      const State* originating_state);

  // Track and update information about tree balance.
  void UpdateLeafNode(Node* node, const State& state,
                      int leaf_depth, double chance_reach_probs);

  // Build the tree.
  void RecursivelyBuildTree(Node* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob);
  void BuildTerminalNode(Node* parent, int depth, const State& state,
                         double chance_reach_prob);
  void BuildDecisionNode(Node* parent, int depth, const State& state,
                         int move_limit, double chance_reach_prob);
  void BuildObservationNode(Node* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob);
};

// Provide convenient types for usage in CFR-based algorithms.
class CFRNode;
using CFRTree = InfostateTree<CFRNode>;

class CFRNode : public InfostateNode</*Self=*/CFRNode> {
 public:
  CFRInfoStateValues values_;  // TODO: use just floats.
  std::vector<Action> terminal_history_;
  CFRNode(const CFRTree& tree, CFRNode* parent, int incoming_index,
          InfostateNodeType type, const std::string& infostate_string,
          double terminal_utility, double terminal_chn_reach_prob,
          const State* originating_state) :
      InfostateNode<CFRNode>(
          tree, parent, incoming_index, type, infostate_string,
          terminal_utility, terminal_chn_reach_prob, originating_state)  {
    SPIEL_DCHECK_TRUE(
        !(originating_state && type == kDecisionInfostateNode)
            || originating_state->IsPlayerActing(tree.GetPlayer()));
    if (originating_state) {
      if (type_ == kDecisionInfostateNode) {
        values_ = CFRInfoStateValues(
            originating_state->LegalActions(tree.GetPlayer()));
      }
      if (type_ == kTerminalInfostateNode) {
        terminal_history_ = originating_state->History();
      }
    }
  }

  // Provide a convenient operator to access the values.
  CFRInfoStateValues* operator->() {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return &values_;
  }
  // Provide getters as well.
  const CFRInfoStateValues& values() const {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return values_;
  }
  CFRInfoStateValues& values() {
    SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
    return values_;
  }
  absl::Span<const Action> TerminalHistory() const {
    SPIEL_DCHECK_EQ(type_, kTerminalInfostateNode);
    return absl::MakeSpan(terminal_history_);
  }
};

void CollectInfostateLookupTable(
    const CFRNode& node,
    std::unordered_map<std::string, const CFRInfoStateValues*>* out);

}  // namespace algorithms
}  // namespace open_spiel

// Template implementation.
#include "open_spiel/algorithms/infostate_tree.hpp"

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
