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


#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game, int max_depth_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator
) :
    game_(std::move(game)),
    public_observer_(std::move(game_->MakeObserver(kPublicStateObsType, {}))),
    player_ranges_({std::vector<double>{1.}, std::vector<double>{1.}}),
    leaf_evaluator_(std::move(leaf_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    propagators_({std::make_unique<CFRTree>(*game_, 0, max_depth_limit),
                  std::make_unique<CFRTree>(*game_, 1, max_depth_limit)}) {
  SPIEL_CHECK_TRUE(terminal_evaluator_);
  PrepareLeafPublicStates();
  EncodePublicStates();
}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game,
    absl::Span<const State*> start_states,
    std::array<std::vector<double>, 2> player_ranges,
    absl::Span<const double> chance_reach_probs,
    int max_depth_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer,
    const std::shared_ptr<Observer>& infostate_observer
) :
    game_(std::move(game)),
    public_observer_(std::move(public_observer)),
    player_ranges_(std::move(player_ranges)),
    leaf_evaluator_(std::move(leaf_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    propagators_({
                     std::make_unique<CFRTree>(start_states, chance_reach_probs,
                                               infostate_observer, 0,
                                               max_depth_limit),
                     std::make_unique<CFRTree>(start_states, chance_reach_probs,
                                               infostate_observer, 1,
                                               max_depth_limit)
                 }) {
  SPIEL_CHECK_EQ(start_states.size(), player_ranges_[0].size());
  SPIEL_CHECK_EQ(start_states.size(), player_ranges_[1].size());
  SPIEL_CHECK_EQ(start_states.size(), chance_reach_probs.size());
  SPIEL_CHECK_TRUE(terminal_evaluator_);
  PrepareLeafPublicStates();
  EncodePublicStates();
}

void DepthLimitedCFR::PrepareLeafPublicStates() {
  Observation public_observation(*game_, public_observer_);

  for (int pl = 0; pl < 2; ++pl) {
    int leaf_position = 0;
    for (const CFRNode& leaf_node : propagators_[pl].tree->leaves_iterator()) {
// TODO: common parent branching invariant
//      // The tree is balanced, therefore we should be iterating over leaves
//      // in an ordered fashion.
//      const int tree_depth = propagators_[pl].depth_branching.size();
//      SPIEL_CHECK_EQ(propagators_[pl].depth_branching[tree_depth - 2].back())

      SPIEL_CHECK_FALSE(leaf_node.CorrespondingStates().empty());
      const std::unique_ptr<State>& some_state =
          leaf_node.CorrespondingStates()[0];
      public_observation.SetFrom(*some_state, kDefaultPlayerId);
      SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
          leaf_node, public_observation.Tensor()));
      LeafPublicState* leaf_state = GetPublicLeaf(public_observation.Tensor());
      leaf_state->leaf_nodes[pl].push_back(&leaf_node);
      leaf_positions_[&leaf_node] = leaf_position;
      leaf_position++;
    }

    SPIEL_CHECK_EQ(leaf_position, propagators_[pl].cf_values.size());
  }
}

void DepthLimitedCFR::EncodePublicStates() {
  for (const LeafPublicState& public_leaf : public_leaves_) {
    SPIEL_DCHECK_TRUE(public_leaf.IsConsistent());
    if (public_leaf.IsTerminal()) {
      encoded_leaves_.push_back(terminal_evaluator_->EncodePublicState(public_leaf));
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      encoded_leaves_.push_back(leaf_evaluator_->EncodePublicState(public_leaf));
    }
  }
}

LeafPublicState* DepthLimitedCFR::GetPublicLeaf(
    absl::Span<float> public_tensor) {
  for (LeafPublicState& state : public_leaves_) {
    if (state.public_tensor == public_tensor) return &state;
  }
  // None found: create and return the pointer.
  public_leaves_.emplace_back(public_tensor);
  return &public_leaves_.back();
}

bool DepthLimitedCFR::DoStatesProduceEqualPublicObservations(
    const CFRNode& node, absl::Span<float> expected_observation) {
  Observation public_observation(*game_, public_observer_);

  // Check that indeed all states produce the same public observations.
  for (const std::unique_ptr<State>& state : node.CorrespondingStates()) {
    public_observation.SetFrom(*state, kDefaultPlayerId);
    if (public_observation.Tensor() != expected_observation) return false;
  }
  return true;
}

std::shared_ptr<LeafEvaluator> MakeTerminalEvaluator() {
  return std::make_shared<TerminalEvaluator>();
}

TerminalPublicState::TerminalPublicState(const LeafPublicState& state) {
  auto& leaf_nodes = state.leaf_nodes;
  SPIEL_CHECK_EQ(leaf_nodes[0].size(), leaf_nodes[1].size());
  const int num_terminals = leaf_nodes[0].size();
  utilities.reserve(num_terminals);
  permutation.reserve(num_terminals);
  cfvs[0] = std::vector<float>(num_terminals);
  cfvs[1] = std::vector<float>(num_terminals);

  using History = absl::Span<const Action>;
  std::map<History, int> player1_map;
  for (int i = 0; i < num_terminals; ++i) {
    player1_map[leaf_nodes[1][i]->TerminalHistory()] = i;
  }
  SPIEL_CHECK_EQ(player1_map.size(), leaf_nodes[1].size());

  for (int i = 0; i < num_terminals; ++i) {
    const CFRNode const* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const CFRNode const* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const CFRNode const* leaf = leaf_nodes[0][i];
    const double v = leaf->TerminalValue();
    const double chn = leaf->TerminalChanceReachProb();
    utilities.push_back(v * chn);
    permutation.push_back(permutation_index);
  }
  SPIEL_DCHECK_EQ(
  // A quick check to see if the permutation is ok
  // by computing the arithmetic sum.
      std::accumulate(permutation.begin(),
                      permutation.end(), 0),
      num_terminals * (num_terminals - 1) / 2);
}

std::unique_ptr<EncodedPublicState> TerminalEvaluator::EncodePublicState(
    const LeafPublicState& state) const {
  return std::make_unique<TerminalPublicState>(state);
}

std::array<absl::Span<const float>, 2> TerminalEvaluator::EvaluatePublicLeaf(
    EncodedPublicState* state,
    const std::array<std::vector<double>, 2>& ranges) const {
  auto* terminal = open_spiel::down_cast<TerminalPublicState*>(state);

  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    terminal->cfvs[0][i] = terminal->utilities[i] * ranges[1][j];
    terminal->cfvs[1][j] = -terminal->utilities[i] * ranges[0][i];
  }

  return {
      absl::MakeSpan(terminal->cfvs[0]),
      absl::MakeSpan(terminal->cfvs[1])
  };
}

void DepthLimitedCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    propagators_[0].TopDown();
    propagators_[1].TopDown();

    EvaluateLeaves();

    propagators_[0].BottomUp();
    propagators_[1].BottomUp();
    SPIEL_DCHECK_TRUE(
        fabs(propagators_[0].cf_values[0] + propagators_[1].cf_values[0])
            < 1e-6);
  }
}

void DepthLimitedCFR::RunAlternatingIterations(int iterations) {
  // Warm up reach probs buffers.
  propagators_[0].TopDown();
  propagators_[1].TopDown();

  for (int t = 0; t < iterations; ++t) {
    for (int i = 0; i < 2; ++i) {
      propagators_[1 - i].TopDown();
      EvaluateLeaves();
      propagators_[i].BottomUp();
    }
  }
}

void DepthLimitedCFR::EvaluateLeaves() {
  SPIEL_CHECK_EQ(public_leaves_.size(), encoded_leaves_.size());
  for (int i = 0; i < public_leaves_.size(); ++i) {
    const LeafPublicState& public_leaf = public_leaves_[i];
    EncodedPublicState* encoded_leaf = encoded_leaves_[i].get();

    // 1. Prepare ranges
    auto ranges = std::array<std::vector<double>, 2>();
    for (int pl = 0; pl < 2; pl++) {
      const int num_infostate_leafs = public_leaf.leaf_nodes[pl].size();
      ranges[pl].reserve(num_infostate_leafs);
      for (const CFRNode* leaf_node : public_leaf.leaf_nodes[pl]) {
        const int leaf_position = leaf_positions_.at(leaf_node);
        SPIEL_DCHECK_GE(leaf_position, 0);
        SPIEL_DCHECK_LT(leaf_position, propagators_[pl].reach_probs.size());
        ranges[pl].push_back(propagators_[pl].reach_probs[leaf_position]);
      }
    }

    // 2. Evaluate: compute cfvs.
    std::array<absl::Span<const float>, 2> cfvs;
    if (public_leaf.IsTerminal()) {
      cfvs = terminal_evaluator_->EvaluatePublicLeaf(encoded_leaf, ranges);
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      cfvs = leaf_evaluator_->EvaluatePublicLeaf(encoded_leaf, ranges);
    }

    // 3. Update cfvs for propagators.
    for (int pl = 0; pl < 2; pl++) {
      const int num_infostate_leafs = public_leaf.leaf_nodes[pl].size();
      SPIEL_CHECK_EQ(cfvs[pl].size(), num_infostate_leafs);
      for (int l = 0; l < num_infostate_leafs; ++l) {
        const CFRNode* leaf_node = public_leaf.leaf_nodes[pl][l];
        const int leaf_position = leaf_positions_.at(leaf_node);
        SPIEL_DCHECK_GE(leaf_position, 0);
        SPIEL_DCHECK_LT(leaf_position, propagators_[pl].reach_probs.size());
        propagators_[pl].cf_values[leaf_position] = cfvs[pl][l];
      }
    }
  }
}

std::unordered_map<std::string, CFRInfoStateValues const*>
DepthLimitedCFR::InfoStateValuesPtrTable() const {
  std::unordered_map<std::string, CFRInfoStateValues const*> vec_ptable;
  CollectInfostateLookupTable(propagators_[0].tree->Root(), &vec_ptable);
  CollectInfostateLookupTable(propagators_[1].tree->Root(), &vec_ptable);
  return vec_ptable;
}

bool LeafPublicState::IsConsistent() const {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // Additionally, they should all be terminal or non-terminal.
  int num_terminals = 0, num_nonterminals = 0;

  for (int pl = 0; pl < 2; ++pl) {
    for (const CFRNode* node : leaf_nodes[pl]) {
      if (!node->IsLeafNode()) return false;
      if (node->Tree().GetPlayer() != pl) return false;
      if (node->Type() == kTerminalInfostateNode) num_terminals++;
      else num_nonterminals++;
    }
  }

  return !(num_terminals > 0 && num_nonterminals > 0);
}

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel
