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

// This file contains data structures used in imperfect information games.
// Specifically, we implement an infostate tree, a representation of how a game
// looks like from the perspective of an acting player.
//
// The infostate tree contains infostate nodes, which describe where the player
// is acting, getting observations or receiving terminal utilities and finishing
// the game. See `InfostateNodeType` for more details.
//
// As the tree can be constructed with a depth limit, we make a distinction
// between leaf nodes and non-leaf nodes. All terminal nodes are leaf nodes.
//
// The identification of infostates is based on strings from an information
// state observer, i.e. one that is constructed using kInfoStateObsType.
//
// As algorithms typically need to store information associated to the nodes
// of the tree, we provide following addressing mechanisms:
// - `DecisionId` refers to an infostate where the player acts.
// - `SequenceId` refers to a sequence of actions player has done so far.
// - `LeafId` refers to an infostate node which is a leaf.
// All of these ids can be used to get a pointer to the infostate node.
//
// To enable some very specific algorithmic optimizations we construct the trees
// "balanced" by default. We call a _balanced_ tree one which has all leaf nodes
// at the same depth. To ensure the tree is balanced, we may need to pad "dummy"
// observation nodes as prefixes for the (previously too shallow) leafs.
//
// [1]: Smoothing Techniques for Computing Nash Equilibria of Sequential Games

namespace open_spiel {
namespace algorithms {

// To categorize infostate nodes we use nomenclature from the [Predictive CFR]
// paper.
//
// - In _decision nodes_, the acting player selects actions.
// - The _observation nodes_ can correspond to State that is a chance node,
//   or opponent's node. Importantly they can correspond also to the acting
//   player's node, as the player may have discovered something as a result
//   of its action in the previous decision node.
// - Additionally, we use _terminal nodes_, which correspond to a single State
//   terminal history.
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
// It could be interpreted as "the player observes the start of the game" and
// it also corresponds to the empty sequence.
// This is the infostate string for this node.
constexpr char* kDummyRootNodeInfostate = "(dummy root)";

// Sometimes we need to create infostate nodes that do not have a corresponding
// game State, and therefore we cannot retrieve their string representations.
// This happens in simultaneous move games or if we rebalance game trees.
constexpr char* kFillerInfostate = "(filler node)";

// Forward declarations.
class InfostateTree;

namespace {

// FIXME I know anonymous namespaces in headers go against the google's
//        c++ guidelines, but I don't know how to do this without introducing
//        boilerplate code.

// An implementation detail - Not to be used directly.
// Creates indexing of specific infostate nodes.
//
// We use CRTP as it allows us to reuse the implementation for indexing various
// nodes in the tree. Most importantly it allows to make debug-time checks to
// make sure that we are using the ids on appropriate trees and we do not try
// to index opponents' trees.
template<class Self>
struct NodeId {
  size_t id = -1;
#ifndef NDEBUG  // Allow additional automatic debug-time checks.
  InfostateTree* tree = nullptr;
  explicit constexpr NodeId() {}
  NodeId(size_t id_value, InfostateTree* tree_ptr)
      : id(id_value), tree(tree_ptr) {}
  bool operator==(const Self& rhs) const {
    SPIEL_CHECK_EQ(tree, rhs.tree);
    return id == rhs.id;
  }
  bool BelongsToTree(InfostateTree* other) const { return tree == other; }
#else
  TreeNodeId<Self>(size_t id_value, InfostateTree*) : id(id_value) {}
  bool operator==(const Self & rhs) const { return id == rhs.id; }
  bool BelongsToTree(InfostateTree* other) const {
    SpielFatalError("Must not be called in release mode!");
  }
#endif
  bool operator!=(const Self& rhs) const { return !(rhs == *this); }
  std::ostream& operator<<(std::ostream& os) const {
    return os << typeid(Self).name() << '{' << id << '}';
  }
};

}  // namespace

// TODO docs
struct SequenceId : public NodeId<SequenceId> {
  using NodeId<SequenceId>::NodeId;
};
// TODO docs
struct DecisionId : public NodeId<DecisionId> {
  using NodeId<DecisionId>::NodeId;
};
constexpr DecisionId kUndefinedDecisionId = DecisionId();
// TODO docs
struct LeafId : public NodeId<LeafId> {
  using NodeId<LeafId>::NodeId;
};
constexpr LeafId kUndefinedLeafId = LeafId();

class InfostateNode;

// Creates an infostate tree for a player based on the initial state
// of the game, up to some move limit.
std::unique_ptr<InfostateTree> MakeInfostateTree(
    const Game& game, Player acting_player,
    int max_move_limit = 1000, bool make_balanced = true);

// Creates an infostate tree for a player based on some start states,
// up to some move limit from the deepest start state.
std::unique_ptr<InfostateTree> MakeInfostateTree(
    const std::vector<const State*>& start_states,
    const std::vector<float>& chance_reach_probs,
    std::shared_ptr<Observer> infostate_observer, Player acting_player,
    int max_move_ahead_limit = 1000, bool make_balanced = true);

class InfostateTree final {
  // Note that only MakeInfostateTree is allowed to call the constructor
  // to ensure the trees are always allocated on heap. We do this so that all
  // the collected pointers are valid throughout the tree's lifetime even if
  // they are moved around.
 private:
  InfostateTree(const Game& game, Player acting_player,
                int max_move_limit = 1000, bool make_balanced = true);
  InfostateTree(
      const std::vector<const State*>& start_states,
      const std::vector<float>& chance_reach_probs,
      std::shared_ptr<Observer> infostate_observer, Player acting_player,
      int max_move_ahead_limit = 1000, bool make_balanced = true);
  // Friend factories.
  friend std::unique_ptr<InfostateTree> MakeInfostateTree(
      const Game&, Player, int, bool);
  friend std::unique_ptr<InfostateTree> MakeInfostateTree(
      const std::vector<const State*>&, const std::vector<float>&,
      std::shared_ptr<Observer>, Player, int, bool);

 public:
  const InfostateNode& root() const { return *root_; }
  InfostateNode* mutable_root() { return root_.get(); }
  Player acting_player() const { return player_; }
  int tree_height() const { return tree_height_; }
  bool is_balanced() const { return is_tree_balanced_; }

  size_t num_decision_infostates() const { return decision_infostates_.size(); }
  size_t num_sequences() const { return sequences_.size(); }
  size_t num_leaves() const { return nodes_at_depth_.back().size(); }
  int root_branching_factor() const;

  // Returns cached pointers to leaf nodes of the CFR tree.
  const std::vector<InfostateNode*>& leaf_nodes() const {
    return nodes_at_depth_.back();
  }
  const std::vector<std::vector<InfostateNode*>>& nodes_at_depth() const {
    return nodes_at_depth_;
  }

  // For debugging.
  void PrintStats();

 private:
  const Player player_;
  const std::shared_ptr<Observer> infostate_observer_;
  std::unique_ptr<InfostateNode> root_;

  // A value that helps to determine if the tree is balanced.
  int tree_height_ = -1;
  // We call a tree balanced if all leaves are in the same depth.
  bool is_tree_balanced_ = true;

  std::vector<InfostateNode*> decision_infostates_;
  std::vector<InfostateNode*> sequences_;
  // Tree structure information. The last vector corresponds to the leaf nodes.
  std::vector<std::vector<InfostateNode*>> nodes_at_depth_;

  // Utility function whenever we create a new node for the tree.
  std::unique_ptr<InfostateNode> MakeNode(
      InfostateNode* parent, InfostateNodeType type,
      const std::string& infostate_string, double terminal_utility,
      double terminal_ch_reach_prob, const State* originating_state);
  std::unique_ptr<InfostateNode> MakeRootNode() const;

  // Makes sure that all tree leaves are at the same height.
  // It inserts a linked list of dummy observation nodes with appropriate length
  // to balance all the leaves. In the worst case this makes the tree about 2x
  // as large (in the number of nodes).
  void Rebalance();

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

// Iterate over a vector of unique pointers, but expose only the raw pointers.
template<class T>
class VecWithUniquePtrsIterator {
  int pos_;
  const std::vector<std::unique_ptr<T>>& vec_;
 public:
  explicit VecWithUniquePtrsIterator(
      const std::vector<std::unique_ptr<T>>& vec, int pos = 0)
      : pos_(pos), vec_(vec) {}
  VecWithUniquePtrsIterator& operator++() { pos_++; return *this; }
  bool operator==(VecWithUniquePtrsIterator other) const {
    return pos_ == other.pos_;
  }
  bool operator!=(VecWithUniquePtrsIterator other) const {
    return !(*this == other);
  }
  T* operator*() { return vec_[pos_].get(); }
  VecWithUniquePtrsIterator begin() const { return *this; }
  VecWithUniquePtrsIterator end() const {
    return VecWithUniquePtrsIterator(vec_, vec_.size());
  }
};

// TODO docs
class InfostateNode final {
 private:
  // Only InfostateTree is allowed to construct nodes.
  InfostateNode(
      const InfostateTree& tree, InfostateNode* parent, int incoming_index,
      InfostateNodeType type, const std::string& infostate_string,
      const DecisionId& decision_id, double terminal_utility,
      double terminal_ch_reach_prob, const State* originating_state);
  friend class InfostateTree;

 public:
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

  VecWithUniquePtrsIterator<InfostateNode> child_iterator() const {
    return VecWithUniquePtrsIterator(children_);
  }

 private:
  // Make sure that the subtree ends at the requested target depth by inserting
  // dummy observation nodes with one outcome.
  void Rebalance(int target_depth, int current_depth);

  // Get the unique_ptr for this node. The usage is intended only for tree
  // balance manipulation.
  std::unique_ptr<InfostateNode> Release();

  // Change the parent of this node by inserting it at at index
  // of the new parent. The node at the existing position will be freed.
  // We pass the unique ptr of itself, because calling Release might be
  // undefined: the node we want to swap a parent for can be root of a subtree.
  void SwapParent(std::unique_ptr<InfostateNode> self, InfostateNode* target, int at_index);

 protected:
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
  // Identifier if this node is a decision node.
  const DecisionId decision_id_;
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
