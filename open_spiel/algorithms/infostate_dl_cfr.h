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
// - CreateContext: take a leaf public state and return any special context
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
  const std::vector<float> public_tensor;
  // For each player, store a pointer to a leaf node for this public state,
  // within the depth-limited infostate tree. If needed, you can get access
  // to its `State`s via `CFRNode::CorrespondingStates()`.
  std::array<std::vector<const InfostateNode*>, 2> leaf_nodes;
  // For each player, store ranges for the top-most infostates.
  std::array<std::vector<float>, 2> ranges;
  // For each player, store counterfactual values for the top-most infostates.
  std::array<std::vector<float>, 2> values;

  explicit LeafPublicState(absl::Span<float> tensor)
      : public_tensor(tensor.begin(), tensor.end()) {}

  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const;
  // Debugging check: makes sure that the call to IsTerminal() is correct.
  bool IsConsistent() const;
};

// Derived classes specify members as they need.
struct PublicStateContext {
  // Make sure PublicStateContext is polymorphic.
  virtual ~PublicStateContext() = default;
};

// Leaf evaluator can create appropriate contexts for later evaluation of public
// states. The derived classes should down_cast the context as needed.
// Ranges and values are saved within the public state.
struct LeafEvaluator {
  virtual std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& leaf_state) const { return nullptr; };
  virtual void EvaluatePublicState(
      LeafPublicState* public_state, PublicStateContext* context) const = 0;
};

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicStateContext : public PublicStateContext {
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<float> utilities;
  explicit TerminalPublicStateContext(const LeafPublicState& state);
};

struct TerminalEvaluator : public LeafEvaluator {
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
                  std::array<std::shared_ptr<InfostateTree>, 2> trees,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer);

  void RunSimultaneousIterations(int iterations);
  void SimultaneousTopDownEvaluate();

  void SetPlayerRanges(const std::array<std::vector<float>, 2>& ranges);
  float RootValue(Player pl = 0) const;
  std::array<absl::Span<const float>, 2> RootChildrenCfValues() const;

  std::array<const InfostateNode*, 2> Roots() const;
  std::array<std::shared_ptr<InfostateTree>, 2>& Trees();

  CFRInfoStateValuesPtrTable InfoStateValuesPtrTable();

  std::vector<std::unique_ptr<PublicStateContext>>& GetContexts();
  std::vector<LeafPublicState>& GetPublicLeaves();

  // Trunk evaluation.
  float CfBestResponse(Player responding_player) const;
  float TrunkExploitability() const;

 private:
  const std::shared_ptr<const Game> game_;
  std::array<std::shared_ptr<InfostateTree>, 2> trees_;
  const std::shared_ptr<Observer> public_observer_;
  const std::shared_ptr<const LeafEvaluator> leaf_evaluator_;
  const std::shared_ptr<const LeafEvaluator> terminal_evaluator_;
  std::array<std::vector<float>, 2> player_ranges_;

  // Allocated based on propagator / cfr tree construction.
  std::vector<LeafPublicState> public_leaves_;
  std::vector<std::unique_ptr<PublicStateContext>> contexts_;
  std::map<const InfostateNode*, int> leaf_positions_;

  // Mutable values to keep track of.
  // These have the size of largest depth of the tree (i.e. leaf nodes).
  std::array<std::vector<float>, 2> reach_probs_;
  std::array<std::vector<float>, 2> cf_values_;

  std::unordered_map<const InfostateNode*, CFRInfoStateValues> node_values_;

  void PrepareRootReachProbs();
  void PrepareLeafNodesForPublicStates();
  void PrepareRangesAndValuesForPublicStates();
  void CreateContexts();
  void EvaluateLeaves();
  LeafPublicState* GetPublicLeaf(absl::Span<float> public_tensor);
  float CfBestResponse(const InfostateNode& node, Player pl, int* leaf_index) const;

  // Internal checks.
  bool DoStatesProduceEqualPublicObservations(
      const InfostateNode& node, absl::Span<float> expected_observation);
};


// -- CFR evaluator ------------------------------------------------------------

struct CFRPublicState : public PublicStateContext {
  std::unique_ptr<DepthLimitedCFR> dlcfr;
  explicit CFRPublicState(std::unique_ptr<DepthLimitedCFR> d)
      : dlcfr(std::move(d)) {}
};

struct CFREvaluator : public LeafEvaluator {
  std::shared_ptr<const Game> game;
  int depth_limit;
  std::shared_ptr<const LeafEvaluator> leaf_evaluator;
  std::shared_ptr<const LeafEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  int num_cfr_iterations = 1;

  CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
               std::shared_ptr<const LeafEvaluator> leaf_evaluator,
               std::shared_ptr<const LeafEvaluator> terminal_evaluator,
               std::shared_ptr<Observer> public_observer,
               std::shared_ptr<Observer> infostate_observer);

  std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& state) const override;
  void EvaluatePublicState(LeafPublicState* public_state,
                           PublicStateContext* context) const override;
};

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
