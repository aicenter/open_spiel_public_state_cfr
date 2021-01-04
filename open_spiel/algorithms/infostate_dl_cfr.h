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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Depth-limited CFR is conceptually similar to what `infostate_cfr.h` does.
// Additionally it saves structure of public states in the leaves of the
// depth-limited infostate tree. It collects these into a set of
// `LeafPublicState`s. DL-CFR can be instantiated with custom leaf evaluators,
// which may need to save different structures associated to those leafs.
//
// Therefore the leaf evaluator provides these two methods:
//
// - CreateContext: takes a leaf public state and return any special context
//   representation needed for this leaf evaluator. These will be saved by
//   DL-CFR to provide them later as necessary.
// - EvaluatePublicState: receives a pointer to the leaf public state along with
//   a context. In the state the evaluator can find the current ranges, and it
//   should update the counterfactual values to continue the DL-CFR iterations.
//
// DL-CFR can be queried for `Policy` only in the depth-limited constructed
// parts of the tree.

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

struct LeafPublicState final {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const Observation public_tensor;
  // For each player, store a pointer to a leaf node for this public state,
  // within the depth-limited infostate tree. If needed, you can get access
  // to its `State`s via `CFRNode::CorrespondingStates()`.
  std::array<std::vector<const InfostateNode*>, 2> leaf_nodes;
  // For each player, store ranges for the top-most infostates.
  std::array<std::vector<double>, 2> ranges;
  // For each player, store counterfactual values for the top-most infostates.
  std::array<std::vector<double>, 2> values;
  // Position in the vector of DepthLimitedCFR::public_leaves_
  size_t public_id;

  explicit LeafPublicState(const Observation& public_observation)
      : public_tensor(public_observation) {}

  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const;
  // Debugging check: makes sure that the call to IsTerminal() is correct.
  bool IsConsistent() const;
};

void DebugPrintPublicFeatures(const std::vector<LeafPublicState>& states);

// Derived classes specify members as they need for their specific
// leaf evaluators.
//
// It is **strongly advised** to make the contexts immutable, or if the leaf
// evaluator mutates the context for the purposes of its computation, it should
// restore it back to the original. This is because it is beneficial to share
// the same context between multiple leaf evaluator implementations (for example
// the sequence-form oracle evaluators).
struct PublicStateContext {
  // Make sure PublicStateContext is polymorphic.
  virtual ~PublicStateContext() = default;
};

// Leaf evaluator can create appropriate contexts for later evaluation of public
// states. The derived classes should down_cast the context as needed.
// Ranges and values are saved within the public state.
class LeafEvaluator {
 public:
  virtual ~LeafEvaluator() = default;
  virtual std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& leaf_state) const { return nullptr; };
  virtual void ResetContext(PublicStateContext* context) const {}
  virtual void EvaluatePublicState(
      LeafPublicState* public_state, PublicStateContext* context) const = 0;
};

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicStateContext final : public PublicStateContext {
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> utilities;
  explicit TerminalPublicStateContext(const LeafPublicState& state);
};

class TerminalEvaluator final : public LeafEvaluator {
 public:
  std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& state) const override;
  void EvaluatePublicState(
      LeafPublicState* state, PublicStateContext* context) const override;
};

std::shared_ptr<LeafEvaluator> MakeTerminalEvaluator();

// -- DL CFR -------------------------------------------------------------------

// At least one evaluator must be specified: leaf_evaluator
// or terminal_evaluator.
class DepthLimitedCFR {
 public:
  DepthLimitedCFR(std::shared_ptr<const Game> game, int depth_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator);

  DepthLimitedCFR(std::shared_ptr<const Game> game,
                  std::vector<std::shared_ptr<InfostateTree>> depth_lim_trees,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer,
                  std::vector<BanditVector> bandits);

  void RunSimultaneousIterations(int iterations);
  void PrepareRootReachProbs();
  void EvaluateLeaves();

  void SetPlayerRanges(const std::array<std::vector<double>, 2>& ranges);
  double RootValue(Player pl = 0) const;
  std::array<absl::Span<const double>, 2> RootChildrenCfValues() const;
  void Reset();

  // Accessors.
  std::vector<std::shared_ptr<InfostateTree>>& trees() { return trees_; }
  std::vector<BanditVector>& bandits() { return bandits_; }
  std::vector<std::unique_ptr<PublicStateContext>>& contexts() {
    return contexts_;
  }
  std::vector<LeafPublicState>& public_leaves() { return public_leaves_; }
  std::vector<std::vector<double>>& reach_probs() { return reach_probs_; }
  std::vector<std::vector<double>>& cf_values() { return cf_values_; }

  // Trunk evaluation.
  std::shared_ptr<Policy> AveragePolicy();
  std::shared_ptr<Policy> CurrentPolicy();

 private:
  const std::shared_ptr<const Game> game_;
  std::vector<std::shared_ptr<InfostateTree>> trees_;
  const std::shared_ptr<Observer> public_observer_;
  const std::shared_ptr<const LeafEvaluator> leaf_evaluator_;
  const std::shared_ptr<const LeafEvaluator> terminal_evaluator_;
  std::array<std::vector<double>, 2> player_ranges_;

  // Allocated based on propagator / cfr tree construction.
  std::vector<LeafPublicState> public_leaves_;
  std::vector<std::unique_ptr<PublicStateContext>> contexts_;
  std::map<const InfostateNode*, int> leaf_positions_;

  // Mutable values to keep track of.
  // These have the size of largest depth of the tree (i.e. leaf nodes).
  std::vector<std::vector<double>> reach_probs_;
  std::vector<std::vector<double>> cf_values_;

  std::vector<BanditVector> bandits_;

  size_t num_iterations_ = 0;

  void PrepareLeafNodesForPublicStates();
  void PrepareRangesAndValuesForPublicStates();
  void CreateContexts();
  LeafPublicState* GetPublicLeaf(const Observation& public_observation);

  // Internal checks.
  bool DoStatesProduceEqualPublicObservations(
      const InfostateNode& node, absl::Span<float> expected_observation);
};


// -- CFR evaluator ------------------------------------------------------------

struct CFRContext : public PublicStateContext {
  std::unique_ptr<DepthLimitedCFR> dlcfr;
  explicit CFRContext(std::unique_ptr<DepthLimitedCFR> d)
      : dlcfr(std::move(d)) {}
};

struct CFREvaluator : public LeafEvaluator {
  std::shared_ptr<const Game> game;
  int depth_limit;
  std::shared_ptr<const LeafEvaluator> leaf_evaluator;
  std::shared_ptr<const LeafEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  bool reset_subgames_on_evaluation = true;
  int num_cfr_iterations = 1;
  std::string bandit_name = "PredictiveRegretMatchingPlus";

  CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
               std::shared_ptr<const LeafEvaluator> leaf_evaluator,
               std::shared_ptr<const LeafEvaluator> terminal_evaluator,
               std::shared_ptr<Observer> public_observer,
               std::shared_ptr<Observer> infostate_observer);

  std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& state) const override;
  void ResetContext(PublicStateContext* context) const override;
  void EvaluatePublicState(LeafPublicState* public_state,
                           PublicStateContext* context) const override;
};

// -- Range table --------------------------------------------------------------


template<class T>
struct BijectiveContainer {
  std::map<T, T> x2y;
  std::map<T, T> y2x;

  void put(std::pair<T, T> xy) {
    const T& x = xy.first;
    const T& y = xy.second;
    SPIEL_CHECK_TRUE(x2y.find(x) == x2y.end());
    SPIEL_CHECK_TRUE(y2x.find(y) == y2x.end());
    x2y[x] = y;
    y2x[y] = x;
  }
  const std::map<T, T>& tree_to_net() const { return x2y; }
  const std::map<T, T>& net_to_tree() const { return y2x; }

  size_t size() const {
    SPIEL_CHECK_EQ(x2y.size(), y2x.size());
    return x2y.size();
  }
};

struct RangeTable {
  // Bijection between ranges coming from infostate tree (x)
  // and the input position (y), called also hand, for each public state.
  // This is used for encoding NN inputs (resp. outputs).
  // Forward:  tree  -> input positions
  // Backward: output positions -> tree
  std::vector<BijectiveContainer<size_t>> bijections;

  // List all possible private observations ("hands") for each player.
  // Their vector indices represent the input position for each public state.
  std::vector<Observation> private_hands;

  RangeTable(int num_public_states) : bijections(num_public_states) {}
  size_t largest_range() const;
  size_t hand_index(const Observation& obs);
};

std::vector<RangeTable> CreateRangeTables(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves);

void DebugPrintRangeTables(const std::vector<dlcfr::RangeTable>& tables);

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
