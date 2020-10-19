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


namespace open_spiel {
namespace algorithms {
namespace dlcfr {

struct LeafPublicState {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const std::vector<float> public_tensor;

  // For each player, encode the position of a leaf node in the flattened
  // tree structure. This is used to identify where we should store the result
  // of leaf evaluation into each of player's tree propagators.
  std::array<std::vector<CFRNode const*>, 2> leaf_nodes;

  LeafPublicState(absl::Span<float> tensor)
      : public_tensor(tensor.begin(), tensor.end()) {}

  bool IsTerminal() const {
    // todo: checks
    return leaf_nodes[0][0]->Type() == kTerminalInfostateNode;
  }
};

struct EncodedPublicState {
  // Make sure EncodedPublicState is polymorphic.
  virtual ~EncodedPublicState() = default;
};

// Leaf evaluator returns cf values for leaf public states. It receives their
// encoded representation for easier usage. The derived classes should down_cast
// the encoded states they receive with the pointer.
class LeafEvaluator {
 public:
  virtual std::unique_ptr<EncodedPublicState> EncodePublicState(
      const LeafPublicState& state) const = 0;
  virtual std::array<absl::Span<const float>, 2> EvaluatePublicLeaf(
      EncodedPublicState*,
      const std::array<std::vector<double>, 2>& ranges) const = 0;
};

struct TerminalPublicState : public EncodedPublicState {
  // Map from player 1 index (key) to player 0 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> utilities;
  // Store terminal cfvs so we can return a span to them when evaluating leaves.
  std::array<std::vector<float>, 2> cfvs;

  explicit TerminalPublicState(const LeafPublicState& state);
};

class TerminalEvaluator : public LeafEvaluator {
 public:
  std::unique_ptr<EncodedPublicState> EncodePublicState(
      const LeafPublicState& state) const override;
  std::array<absl::Span<const float>, 2> EvaluatePublicLeaf(
      EncodedPublicState* state,
      const std::array<std::vector<double>, 2>& ranges) const override;
};

std::shared_ptr<LeafEvaluator> MakeTerminalEvaluator();

class DepthLimitedCFR {
 public:
  DepthLimitedCFR(std::shared_ptr<const Game> game,
                  absl::Span<const State*> start_states,
                  std::array<std::vector<double>, 2> player_ranges,
                  absl::Span<const double> chance_reach_probs,
                  int depth_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer,
                  const std::shared_ptr<Observer>& infostate_observer);

  DepthLimitedCFR(std::shared_ptr<const Game> game, int depth_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator);

  void RunSimultaneousIterations(int iterations);
  void RunAlternatingIterations(int iterations);
  void EvaluateLeaves();

  std::unordered_map<std::string, CFRInfoStateValues const*>
    InfoStateValuesPtrTable() const;


 private:
  const std::shared_ptr<const Game> game_;
  const std::shared_ptr<Observer> public_observer_;
  const std::array<std::vector<double>, 2> player_ranges_;
  const std::shared_ptr<const LeafEvaluator> leaf_evaluator_;
  const std::shared_ptr<const LeafEvaluator> terminal_evaluator_;

  std::array<InfostateTreeValuePropagator, 2> propagators_;

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

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
