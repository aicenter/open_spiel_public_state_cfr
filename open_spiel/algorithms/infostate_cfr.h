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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This file provides a vectorized implementation of CFR.
// This is intended for advanced usage and the code may not be as readable
// as a more basic algorithm. See cfr.h for a basic implementation.
//
// This code uses a preconstructed infostate trees of each player. It updates
// all infostates at a tree depth at once. While the current implementation is
// okay in efficiency (~10x faster than CFRSolver), it could be further
// improved:
//
// - Use contiguous memory blocks for storing regrets/current/cumul
//   for each action.
// - Skip subtrees that have zero reach probabilities.
// - Use stack allocation: most trees are small. If they exceed the allocated
//   limit (1024? nodes), use something like a linked list with these large
//   memory blocks.
//
// More todos:
// - Implement CFR+, Predictive CFR+, as local regret minimizers.
// - Provide custom leaf evaluation (neural net)
//
// If you decide to make contributions to this code, please open up an issue
// on github first. Thank you!

namespace open_spiel {
namespace algorithms {

// A helper class that allows to propagate reach probs / cf values
// up and down the tree. It modifies the contents of CFRInfoStateValues saved
// in the supplied tree to save the player's regrets and strategy.
//
// To operate more efficiently, it caches the pointers to nodes at each depth.
// The tree structure must not change in order for this class to work properly!
//
// Before the calling the top-down and bottom-up passes, ranges and leaf values
// must be provided externally.
class InfostateTreeValuePropagator {
  // Tree structure information.
  std::vector<std::vector<CFRNode*>> nodes_at_depth;

  // Mutable values to keep track of.
  // These have the size of largest depth of the tree (i.e. leaf nodes).
  std::vector<float> reach_probs;
  std::vector<float> cf_values;

 public:
  // Construct the value propagator, so we can use vectorized top-down
  // and bottom-up passes.
  // We require the tree is balanced: we need to make sure that all terminals
  // are at the same depth so that we can propagate the computation of reach
  // probs / cf values in a single vector. This requirement is typically
  // satisfied by most domains already during the tree construction anyway.
  /*implicit*/ InfostateTreeValuePropagator(CFRTree* balanced_tree);

  // Make a top-down pass, using the current policy stored in the tree nodes.
  // This computes the reach_probs_ buffer for storing cumulative product
  // of reach probabilities for leaf nodes.
  // The starting values at depth 1 must be provided externally by writing
  // to range()
  void TopDown();

  // Make a bottom-up pass, starting with the current cf_values stored
  // in the buffer. This loopss over all depths from the bottom.
  // The leaf values must be provided externally by writing to leaves_cf_values().
  void BottomUp();

  // Returns the branching factor of the root node.
  int root_branching_factor() const {
    return nodes_at_depth[0][0]->num_children();
  }
  // Returns the (writable) range, i.e. the reach probabilities
  // of the infostate tree's root children.
  absl::Span<float> range() {
    return absl::MakeSpan(/*ptr=*/&reach_probs[0],
                          /*size=*/root_branching_factor());
  }
  // Returns the (read-only) counter-factual values of the infostate tree's
  // root children. These values are valid only after a BottomUp() pass is made.
  absl::Span<const float> values() const {
    return absl::MakeSpan(/*ptr=*/&cf_values[0],
                          /*size=*/root_branching_factor());
  }
  std::vector<float>& leaves_reach_probs() { return reach_probs; }
  const std::vector<float>& leaves_reach_probs() const { return reach_probs; }
  std::vector<float>& leaves_cf_values() { return cf_values; }
  const std::vector<float>& leaves_cf_values() const { return cf_values; }

  // Calculates the root cf value as weighted sum of the cf_values()
  // (the weights are the respective ranges). If the supplied range is empty,
  // it returns their sum (i.e. all weights have a value of 1).
  float RootCfValue(absl::Span<const float> range = {}) const;

  // Returns cached pointers to leaf nodes of the CFR tree. Unlike the
  // CFRTree::leaves_iterator(), this does not need to recursively traverse
  // the tree.
  const std::vector<CFRNode*>& leaf_nodes() const {
    return nodes_at_depth.back();
  }
  // Returns the number of leaf nodes.
  int num_leaves() const { return nodes_at_depth.back().size(); }
};

// Run vectorized CFR on the whole game or on specified trees.
class InfostateCFR {
 public:
  // Basic constructor for the whole game.
  explicit InfostateCFR(const Game& game);
  // Run CFR on the specified trees.
  explicit InfostateCFR(std::array<CFRTree, 2> cfr_trees);

  void RunSimultaneousIterations(int iterations);
  void RunAlternatingIterations(int iterations);

  // Computes the root value. If the algorithm is running on the full trees
  // of the game, this converges to the game value.
  float RootValue() const { return propagators_[0].RootCfValue(); }

  CFRInfoStateValuesPtrTable InfoStateValuesPtrTable();
  std::shared_ptr<Policy> AveragePolicy();

 private:
  void PrepareTerminals();
  void PrepareRootReachProbs();
  void PrepareRootReachProbs(Player pl);
  void EvaluateLeaves();
  void EvaluateLeaves(Player pl);
  float TerminalReachProbSum();

  // The trees which hold the strategies of the players.
  std::array<CFRTree, 2> trees_;
  // The propagators that make the top-down bottom-up passed on each iteration.
  std::array<InfostateTreeValuePropagator, 2> propagators_;
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> terminal_permutation_;
  // Chance reach probs.
  std::vector<float> terminal_ch_reaches_;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<float> terminal_values_;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
