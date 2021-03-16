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
#include "open_spiel/utils/data_structures.h"

// Depth-limited CFR is conceptually similar to what `infostate_cfr.h` does.
// Additionally it saves structure of public states in the root and leaves
// of the depth-limited infostate tree. It collects these into a set of
// `PublicState`s. DL-CFR can be instantiated with custom state evaluators,
// which may need to save different structures associated to these states.
//
// Therefore the state evaluator provides these two methods:
//
// - CreateContext: takes a public state and return any special context
//   representation needed for this state evaluator. These will be saved by
//   DL-CFR to provide them later as necessary.
// - EvaluatePublicState: receives a pointer to the public state along with
//   a context. In the state the evaluator can find the current beliefs, and it
//   should update the counterfactual values to continue the DL-CFR iterations.
//
// DL-CFR can be queried for `Policy` only in the depth-limited constructed
// parts of the tree.

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

// Public state in which the depth-limited subgame is rooted.
constexpr size_t kInitialPublicState = 0;

struct PublicState final {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const Observation public_tensor;
  // Position in the vector of DepthLimitedCFR::public_leaves_
  const size_t public_id;
  // For each player, store a pointer to the infostate nodes for this public
  // state, within the depth-limited infostate tree. If needed, you can get
  // access to underlying perfect-information `State`s
  // via `InfostateNode::corresponding_states()`.
  // These are assigned once, and no subsequent change should be made.
  /*const*/ std::array<std::vector<const InfostateNode*>, 2> infostate_nodes;
  // For each player, store beliefs for the top-most infostate nodes.
  std::array<std::vector<double>, 2> ranges;  // FIXME: rename to beliefs.
  // For each player, store counterfactual values for the top-most infostates.
  std::array<std::vector<double>, 2> values;

  explicit PublicState(const Observation& public_observation,
                       const size_t public_id)
      : public_tensor(public_observation), public_id(public_id) {}

  // Check if the public state if initial, i.e. the depth-limited subgame
  // is rooted in this public state.
  bool IsInitial() const { return false; /*FIXME public_id == kInitialPublicState;*/ }
  // Check if the public state if a leaf within the depth-limited subgame.
  bool IsLeaf() const { return true; /*FIXME public_id != kInitialPublicState;*/ }
  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const;
  // Debugging check: makes sure that the call to IsTerminal() is correct.
  bool IsConsistent() const;
};

void DebugPrintPublicFeatures(const std::vector<PublicState>& states);

// Derived classes specify members as they need for their specific
// public state evaluators.
//
// It is **strongly advised** to make the contexts immutable, or if the state
// evaluator mutates the context for the purposes of its computation, it should
// restore it back to the original. This is because it is beneficial to share
// the same context between multiple evaluator implementations (for example
// the sequence-form oracle evaluators). If mutation were to happen, each
// evaluator then must take this into account, making implementation more
// difficult.
struct PublicStateContext {
  // Make sure PublicStateContext is polymorphic.
  virtual ~PublicStateContext() = default;
};

// Public state evaluator can create appropriate contexts for later evaluation
// of public states. The derived classes should down_cast the context as needed.
// Beliefs and values are saved within the public state.
class PublicStateEvaluator {
 public:
  virtual ~PublicStateEvaluator() = default;
  virtual std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const { return nullptr; };
  virtual void ResetContext(PublicStateContext* context) const {}
  virtual void EvaluatePublicState(
      PublicState* public_state, PublicStateContext* context) const = 0;
};

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicStateContext final : public PublicStateContext {
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> utilities;
  explicit TerminalPublicStateContext(const PublicState& state);
};

class TerminalEvaluator final : public PublicStateEvaluator {
 public:
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(
      PublicState* state, PublicStateContext* context) const override;
};

std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator();
std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator();

// -- DL CFR -------------------------------------------------------------------

// At least one evaluator must be specified: nonterminal_evaluator
// or terminal_evaluator.
class DepthLimitedCFR {
 public:
  DepthLimitedCFR(std::shared_ptr<const Game> game, int depth_limit,
                  std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
                  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator);

  DepthLimitedCFR(std::shared_ptr<const Game> game,
                  std::vector<std::shared_ptr<InfostateTree>> depth_lim_trees,
                  std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
                  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer,
                  std::vector<BanditVector> bandits);

  void RunSimultaneousIterations(int iterations);
  void PrepareRootReachProbs();
  void EvaluateLeaves();
  void UpdateReachProbs();
  void UpdateTrunk();

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
  template<class T>
  std::vector<T*> contexts_as() {
    std::vector<T*> casted;
    casted.reserve(contexts_.size());
    for (int i = 0; i < contexts_.size(); ++i) {
      PublicState& state = public_states_[i];
      if (state.IsTerminal()) {
        casted.push_back(nullptr);
      } else {
        std::unique_ptr<PublicStateContext>& context = contexts_[i];
        casted.push_back(open_spiel::down_cast<T*>(context.get()));
      }
    }
    return casted;
  }
  std::vector<PublicState>& public_states() { return public_states_; }
  std::vector<std::vector<double>>& reach_probs() { return reach_probs_; }
  std::vector<std::vector<double>>& cf_values() { return cf_values_; }

  // Trunk evaluation.
  std::shared_ptr<Policy> AveragePolicy();
  std::shared_ptr<Policy> CurrentPolicy();
  size_t num_iterations_ = 0;
 private:
  const std::shared_ptr<const Game> game_;
  /*const*/ std::vector<std::shared_ptr<InfostateTree>> trees_;
  const std::shared_ptr<Observer> public_observer_;
  const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator_;
  const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator_;

  // Allocated based on propagator / cfr tree construction.
  // The first PublicState/PublicStateContext corresponds to the
  // root public state.
  /*const*/ std::vector<PublicState> public_states_;
  // Save evaluator-specific information for any public state.
  // If no information should be saved, a nullptr is used.
  /*const*/ std::vector<std::unique_ptr<PublicStateContext>> contexts_;
  // Store the position of a node within the reach_probs_ resp. cf_values_
  /*const*/ std::map<const InfostateNode*, int> node_positions_;

  // -- Mutable values to keep track of. --

  // These have the size at largest depth of the tree, i.e. the size of the
  // leaf infostate nodes.
  std::array<std::vector<double>, 2> player_ranges_;
  std::vector<std::vector<double>> reach_probs_;
  std::vector<std::vector<double>> cf_values_;
  std::vector<BanditVector> bandits_;

  void PrepareInfostateNodesForPublicStates();
  void PrepareRangesAndValuesForPublicStates();
  void CreateContexts();
  PublicState* GetPublicState(const Observation& public_observation);
  PublicState* GetPublicState(InfostateNode* node,
                              Observation& public_observation);

  // Internal checks.
  bool DoStatesProduceEqualPublicObservations(
      const InfostateNode& node, absl::Span<float> expected_observation);
};

// -- Dummy evaluator ----------------------------------------------------------

// Evaluator that does nothing.
struct DummyEvaluator : public algorithms::dlcfr::PublicStateEvaluator {
  std::unique_ptr<algorithms::dlcfr::PublicStateContext> CreateContext(
      const algorithms::dlcfr::PublicState& state) const override {
    return nullptr;
  };
  void ResetContext(
      algorithms::dlcfr::PublicStateContext* context) const override {};
  void EvaluatePublicState(
      algorithms::dlcfr::PublicState* public_state,
      algorithms::dlcfr::PublicStateContext* context) const override {};
};


// -- CFR evaluator ------------------------------------------------------------

struct CFRContext : public PublicStateContext {
  std::unique_ptr<DepthLimitedCFR> dlcfr;
  explicit CFRContext(std::unique_ptr<DepthLimitedCFR> d)
      : dlcfr(std::move(d)) {}
};

struct CFREvaluator : public PublicStateEvaluator {
  std::shared_ptr<const Game> game;
  int depth_limit;
  std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator;
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  bool reset_subgames_on_evaluation = true;
  int num_cfr_iterations = 100;
  std::string bandit_name = "RegretMatchingPlus";

  CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
               std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
               std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
               std::shared_ptr<Observer> public_observer,
               std::shared_ptr<Observer> infostate_observer);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void ResetContext(PublicStateContext* context) const override;
  void EvaluatePublicState(PublicState* state,
                           PublicStateContext* context) const override;
};


void PrintPublicStatesStats(const std::vector<PublicState>& public_leaves);

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_DL_CFR_H_
