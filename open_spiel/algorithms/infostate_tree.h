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

#include <memory>
#include <string>
#include <unordered_map>
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
class InfostateTree;
class InfostateNode;

class InfostateNode {
 public:
  InfostateNode(
      const InfostateTree& tree, InfostateNode* parent, int incoming_index,
      InfostateNodeType type, const std::string& infostate_string,
      double terminal_utility, double terminal_ch_reach_prob,
      const State* originating_state);
  InfostateNode(InfostateNode&&) noexcept = default;
  virtual ~InfostateNode() = default;

  const InfostateTree& tree() const { return tree_; }
  InfostateNode* parent() const { return parent_; }
  int incoming_index() const { return incoming_index_; }
  const InfostateNodeType& type() const { return type_; }
  bool is_leaf_node() const { return children_.empty(); }
  bool is_root_node() const { return !parent_; }
  const std::string& infostate_string() const;
  bool has_infostate_string() const;
  double terminal_utility() const;
  double terminal_chance_reach_prob() const;
  const std::vector<Action>& legal_actions() const;
  const std::vector<std::unique_ptr<State>>& corresponding_states() const;
  const std::vector<double>& corresponding_chance_reach_probs() const;

  InfostateNode* child_at(int i) const { return children_.at(i).get(); }
  int num_children() const { return children_.size(); }
  InfostateNode* AddChild(std::unique_ptr<InfostateNode> child);
  InfostateNode* GetChild(const std::string& infostate_string) const;
  const InfostateNode* FindNode(const std::string& infostate_lookup) const;
  const std::vector<Action>& TerminalHistory() const;

  // Intended only for debug purposes.
  std::string ToString() const;
  // Compute subtree certificate (string representation) for easy comparison.
  std::string ComputeCertificate() const;

  // Iterate over children and expose references to the children
  // (instead of unique_ptrs).
  class ChildIterator {
    int pos_;
    const std::vector<std::unique_ptr<InfostateNode>>& children_;
   public:
    ChildIterator(const std::vector<std::unique_ptr<InfostateNode>>& children,
                  int pos = 0) : pos_(pos), children_(children) {}
    ChildIterator& operator++() { pos_++; return *this; }
    bool operator==(ChildIterator other) const { return pos_ == other.pos_; }
    bool operator!=(ChildIterator other) const { return !(*this == other); }
    InfostateNode& operator*() { return *children_[pos_]; }
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
  std::unique_ptr<InfostateNode> Release();

  // Change the parent of this node by inserting it at at index
  // of the new parent. The node at the existing position will be freed.
  // We pass the unique ptr of itself, because calling Release might be
  // undefined: the node we want to swap a parent for can be root of a subtree.
  void SwapParent(std::unique_ptr<InfostateNode> self, InfostateNode* target, int at_index);

 protected:
  // Needed for adding corresponding_states() during tree traversal.
  friend class InfostateTree;

  // Reference to the tree that this node belongs to. This reference has a valid
  // lifetime, as all the nodes are recursively owned by their parents, and the
  // root is owned by the tree.
  const InfostateTree& tree_;
  // Pointer to the parent node.
  // This is not const so that we can change it when we SwapParent().
  InfostateNode* parent_;
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
  // Cumulative product of chance probabilities leading up to a terminal node.
  const double terminal_chn_reach_prob_;
  // Stored only for decision nodes.
  std::vector<Action> legal_actions_;
  // Children infostate nodes. Notice the node owns its children.
  std::vector<std::unique_ptr<InfostateNode>> children_;
  // Store States that correspond to a leaf node.
  std::vector<std::unique_ptr<State>> corresponding_states_;
  // Store chance reach probs for States that correspond to a leaf node.
  std::vector<double> corresponding_ch_reaches_;

  std::vector<Action> terminal_history_;
};

class InfostateTree final {
 public:
  // Creates an infostate tree for a player based on the initial state
  // of the game, up to some move limit.
  InfostateTree(const Game& game, Player acting_player,
                int max_move_limit = 1000, bool make_balanced = true);

  // Creates an infostate tree for a player based on some start states,
  // up to some move limit from the deepest start state.
  InfostateTree(
      const std::vector<const State*>& start_states,
      const std::vector<float>& chance_reach_probs,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_move_ahead_limit = 1000, bool make_balanced = true);

  const InfostateNode& root() const { return root_; }
  InfostateNode* mutable_root() { return &root_; }
  Player acting_player() const { return player_; }
  int tree_height() const { return tree_height_; }
  bool is_balanced() const { return is_tree_balanced_; }

  // Makes sure that all tree leaves are at the same height.
  // It inserts a linked list of dummy observation nodes with appropriate length
  // to balance all the leaves. In the worst case this makes the tree about 2x
  // as large (in the number of nodes).
  void Rebalance();

  // Returns the branching factor of the root node.
  int root_branching_factor() const { return root_.num_children(); }
  // Returns cached pointers to leaf nodes of the CFR tree. Unlike the
  // CFRTree::leaves_iterator(), this does not need to recursively traverse
  // the tree.
  const std::vector<InfostateNode*>& leaf_nodes() const {
    return nodes_at_depth_.back();
  }
  // Returns the number of leaf nodes.
  int num_leaves() const {
    return nodes_at_depth_.back().size();
  }
  const std::vector<std::vector<InfostateNode*>>& nodes_at_depth() const {
    return nodes_at_depth_;
  }

  // TODO: remove leaves iterator.
  // Iterate over all leaves.
  class LeavesIterator {
    const InfostateTree* tree_;
    const InfostateNode* current_;
   public:
    LeavesIterator(const InfostateTree* tree, const InfostateNode* current);
    LeavesIterator& operator++();
    bool operator==(LeavesIterator other) const;
    bool operator!=(LeavesIterator other) const;
    const InfostateNode& operator*() const;
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
  InfostateNode root_;

  // A value that helps to determine if the tree is balanced.
  int tree_height_ = -1;
  // We call a tree balanced if all leaves are in the same depth.
  bool is_tree_balanced_ = true;
  // Tree structure information. Pointers are collected after rebalancing.
  std::vector<std::vector<InfostateNode*>> nodes_at_depth_;

  InfostateNode CreateRootNode() const;

  // Utility function whenever we create a new node for the tree.
  std::unique_ptr<InfostateNode> MakeNode(
      InfostateNode* parent, InfostateNodeType type, const std::string& infostate_string,
      double terminal_utility, double terminal_ch_reach_prob,
      const State* originating_state);

  // Track and update information about tree balance.
  void UpdateLeafNode(InfostateNode* node, const State& state,
                      int leaf_depth, double chance_reach_probs);

  // Build the tree.
  void RecursivelyBuildTree(InfostateNode* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob);
  void BuildTerminalNode(InfostateNode* parent, int depth, const State& state,
                         double chance_reach_prob);
  void BuildDecisionNode(InfostateNode* parent, int depth, const State& state,
                         int move_limit, double chance_reach_prob);
  void BuildObservationNode(InfostateNode* parent, int depth, const State& state,
                            int move_limit, double chance_reach_prob);

  void CollectTreeStructure(InfostateNode* node, int depth);
};

// A type for tables holding pointers to CFR values.
//
// It is similar to what CFRSolver uses, i.e. the InfoStateValuesTable.
// However, this table has pointers to the values, not the actual values,
// because they are stored within the infostate tree.
//
// It makes looking up the strategies / regrets for players easier to do.
using CFRInfoStateValuesPtrTable =
  std::unordered_map<std::string, CFRInfoStateValues*>;


}  // namespace algorithms
}  // namespace open_spiel


#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_TREE_H_
