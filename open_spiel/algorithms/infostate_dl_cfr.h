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

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Depth-limited CFR is conceptually similar to what `infostate_cfr.h` does.
// Additionally it saves structure of public states in the leaves of the
// depth-limited infostate tree. It collects these into a set of
// `LeafPublicState`s. DL-CFR can be instantiated with different leaf
// evaluators, which may need to save different structures for those leafs.
//
// Therefore the leaf evaluator has to provide two methods:
//
// - EncodeLeafPublicState: take "raw" leaf public state and return the
//   representation needed for this leaf evaluator. These will be saved by
//   DL-CFR to provide them later as necessary.
// - EvaluatePublicState: receives a pointer to the encoded state along with
//   ranges of both players, evaluates such public belief state, and returns
//   the counterfactual values to continue in the DL-CFR iterations.
//
// DL-CFR can be queried for `Policy` only in the DL constructed parts of the
// tree.

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

struct LeafPublicState {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const std::vector<float> public_tensor;

  // For each player, store a pointer to a leaf node for this public state,
  // within the depth-limited infostate tree. If needed, you can get access
  // to its `State`s via `CFRNode::CorrespondingStates()`.
  std::array<std::vector<const CFRNode*>, 2> leaf_nodes;

  LeafPublicState(absl::Span<float> tensor)
      : public_tensor(tensor.begin(), tensor.end()) {}

  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const {
    return leaf_nodes[0][0]->type() == kTerminalInfostateNode;
  }
  // Debugging check: makes sure that the call to IsTerminal() is correct.
  bool IsConsistent() const;
};

// Derived classes specify the members they need.
struct EncodedPublicState {
  // Make sure EncodedPublicState is polymorphic.
  virtual ~EncodedPublicState() = default;
};

// Leaf evaluator returns cf values for leaf public states. It receives their
// encoded representation for easier usage. The derived classes should down_cast
// the encoded states they receive with the pointer.
struct LeafEvaluator {
  virtual std::unique_ptr<EncodedPublicState> EncodeLeafPublicState(
      const LeafPublicState& leaf_state) const = 0;
  virtual std::array<absl::Span<const float>, 2> EvaluatePublicState(
      EncodedPublicState* public_state,
      std::array<absl::Span<const float>, 2> ranges) const = 0;
};

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicState : public EncodedPublicState {
  // Map from player 1 index (key) to player 0 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<float> utilities;
  // Store terminal cfvs so we can return a span to them when evaluating leaves.
  std::array<std::vector<float>, 2> cfvs;

  explicit TerminalPublicState(const LeafPublicState& state);
};

struct TerminalEvaluator : public LeafEvaluator {
  std::unique_ptr<EncodedPublicState> EncodeLeafPublicState(
      const LeafPublicState& state) const override;
  std::array<absl::Span<const float>, 2> EvaluatePublicState(
      EncodedPublicState* state,
      std::array<absl::Span<const float>, 2> ranges) const override;
};

std::shared_ptr<LeafEvaluator> MakeTerminalEvaluator();

// -- DL CFR -------------------------------------------------------------------

class DepthLimitedCFR {
 public:
  DepthLimitedCFR(std::shared_ptr<const Game> game, int depth_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator);

  DepthLimitedCFR(std::shared_ptr<const Game> game,
                  absl::Span<const State*> start_states,
                  absl::Span<const float> chance_reach_probs,
                  int max_move_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer,
                  const std::shared_ptr<Observer>& infostate_observer);

  DepthLimitedCFR(std::shared_ptr<const Game> game,
                  std::array<CFRTree, 2> trees,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer);

  void TrackPlayerRanges(std::array<absl::Span<const float>, 2> track_source);
  std::array<absl::Span<const float>, 2> RootChildrenCfValues() const;
  std::array<absl::Span<const float>, 2> RootChildren() const;

  void RunSimultaneousIterations(int iterations);
  void PrepareRootReachProbs();
  void EvaluateLeaves();

  std::array<const CFRNode*, 2> Roots() const {
    return {&trees_[0].root(), &trees_[1].root() };
  }
  float RootCfValue() const {
    return propagators_[0].RootCfValue(tracked_player_ranges_[0]);
  }

  std::unordered_map<std::string, CFRInfoStateValues const*>
    InfoStateValuesPtrTable() const;

  const std::vector<std::unique_ptr<EncodedPublicState>>&
    GetEncodedLeaves() const { return encoded_leaves_; }

 private:
  const std::shared_ptr<const Game> game_;
  std::array<CFRTree, 2> trees_;
  std::array<InfostateTreeValuePropagator, 2> propagators_;
  const std::shared_ptr<Observer> public_observer_;
  const std::shared_ptr<const LeafEvaluator> leaf_evaluator_;
  const std::shared_ptr<const LeafEvaluator> terminal_evaluator_;

  // Propagators need to have top-most reach probs updated externally.
  // We do that via tracked_player_ranges_, which represents an arbitrary span:
  // These could be reach probs of the parent subgame. However, tracking is not
  // always possible / desirable, so we also save a vector of ranges here
  // as well.
  const std::array<std::vector<float>, 2> player_ranges_;
  std::array<absl::Span<const float>, 2> tracked_player_ranges_;

  // Allocated based on propagator / cfr tree construction.
  std::vector<LeafPublicState> public_leaves_;
  std::vector<std::unique_ptr<EncodedPublicState>> encoded_leaves_;
  std::map<const CFRNode*, int> leaf_positions_;

  void PrepareLeafPublicStates();
  void EncodePublicStates();
  LeafPublicState* GetPublicLeaf(absl::Span<float> public_tensor);

  // Internal checks.
  bool DoStatesProduceEqualPublicObservations(
      const CFRNode& node, absl::Span<float> expected_observation);
};


// -- CFR evaluator ------------------------------------------------------------

struct CFRPublicState : public EncodedPublicState {
  std::unique_ptr<DepthLimitedCFR> dlcfr;
  explicit CFRPublicState(std::unique_ptr<DepthLimitedCFR> d)
      : dlcfr(std::move(d)) {};
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
               std::shared_ptr<Observer> infostate_observer)
      : game(std::move(game)), depth_limit(depth_limit),
        leaf_evaluator(std::move(leaf_evaluator)),
        terminal_evaluator(std::move(terminal_evaluator)),
        public_observer(std::move(public_observer)),
        infostate_observer(std::move(infostate_observer)) {
    SPIEL_CHECK_GT(depth_limit, 0);
  }

  std::unique_ptr<EncodedPublicState> EncodeLeafPublicState(
      const LeafPublicState& leaf_state) const override;
  std::array<absl::Span<const float>, 2> EvaluatePublicState(
      EncodedPublicState* public_state,
      std::array<absl::Span<const float>, 2>) const override;
};


}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
