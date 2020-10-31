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


#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game,
    std::array<CFRTree, 2> trees,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer
) :
    game_(std::move(game)),
    trees_(std::move(trees)),
    propagators_({&trees_[0], &trees_[1]}),
    public_observer_(std::move(public_observer)),
    leaf_evaluator_(std::move(leaf_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    player_ranges_({
       std::vector<float>(trees_[0].root().NumChildren(), 1.),
       std::vector<float>(trees_[1].root().NumChildren(), 1.)
    }),
    tracked_player_ranges_({ player_ranges_[0], player_ranges_[1] }) {
  SPIEL_CHECK_TRUE(terminal_evaluator_);
  PrepareLeafPublicStates();
  EncodePublicStates();
}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game, int max_depth_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator
) : DepthLimitedCFR(game,
                    { CFRTree(*game, 0, max_depth_limit),
                      CFRTree(*game, 1, max_depth_limit) },
                    std::move(leaf_evaluator),
                    std::move(terminal_evaluator),
                    game->MakeObserver(kPublicStateObsType, {})) {}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game,
    absl::Span<const State*> start_states,
    absl::Span<const float> chance_reach_probs,
    int max_move_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer,
    const std::shared_ptr<Observer>& infostate_observer
) : DepthLimitedCFR(std::move(game),
                    { CFRTree(start_states, chance_reach_probs,
                              infostate_observer, /*acting_player=*/0,
                              max_move_limit),
                      CFRTree(start_states, chance_reach_probs,
                              infostate_observer, /*acting_player=*/1,
                              max_move_limit) },
                    std::move(leaf_evaluator), std::move(terminal_evaluator),
                    std::move(public_observer)) {}


void FillStatesAndChanceRange(std::vector<const State*>* start_states,
                              std::vector<float>* chance_reach_probs,
                              absl::Span<const CFRNode* const> start_nodes) {
  // Collect pointers to starting states, along with their reach probs.
  // It's enough to do this just through one player, as the other player
  // has just a permutation of these states.
  for (const CFRNode* cfr_node : start_nodes) {
    SPIEL_CHECK_EQ(cfr_node->corresponding_states().size(),
                   cfr_node->corresponding_chance_reach_probs().size());
    for (int i = 0; i < cfr_node->corresponding_states().size(); ++i) {
      start_states->push_back(
          cfr_node->corresponding_states()[i].get());
      chance_reach_probs->push_back(
          cfr_node->corresponding_chance_reach_probs()[i]);
    }
  }
}

std::array<CFRTree, 2> CreateTrees(
    std::array<absl::Span<const CFRNode* const>, 2> start_nodes,
    const std::shared_ptr<Observer>& infostate_observer,
    int max_move_limit) {
  std::array<std::vector<const State*>, 2> start_states;
  std::array<std::vector<float>, 2> chance_reach_probs;

  FillStatesAndChanceRange(&start_states[0], &chance_reach_probs[0],
                           start_nodes[0]);
  FillStatesAndChanceRange(&start_states[1], &chance_reach_probs[1],
                           start_nodes[1]);

  return {
      CFRTree(absl::MakeSpan(start_states[0]),
              absl::MakeSpan(chance_reach_probs[0]),
              infostate_observer, /*acting_player=*/0, max_move_limit),
      CFRTree(absl::MakeSpan(start_states[1]),
              absl::MakeSpan(chance_reach_probs[1]),
              infostate_observer, /*acting_player=*/1, max_move_limit)
  };
}

void DepthLimitedCFR::PrepareLeafPublicStates() {
  Observation public_observation(*game_, public_observer_);

  for (int pl = 0; pl < 2; ++pl) {
    int leaf_position = 0;
    for (CFRNode* leaf_node : propagators_[pl].LeafNodes()) {
// TODO: common parent branching invariant
//      // The tree is balanced, therefore we should be iterating over leaves
//      // in an ordered fashion.
//      const int tree_depth = propagators_[pl].depth_branching.size();
//      SPIEL_CHECK_EQ(propagators_[pl].depth_branching[tree_depth - 2].back())

      SPIEL_CHECK_FALSE(leaf_node->corresponding_states().empty());
      const std::unique_ptr<State>& some_state =
          leaf_node->corresponding_states()[0];
      public_observation.SetFrom(*some_state, kDefaultPlayerId);
      SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
          *leaf_node, public_observation.Tensor()));
      LeafPublicState* leaf_state = GetPublicLeaf(public_observation.Tensor());
      leaf_state->leaf_nodes[pl].push_back(leaf_node);
      leaf_positions_[leaf_node] = leaf_position;
      leaf_position++;
    }

    SPIEL_CHECK_EQ(leaf_position, propagators_[pl].cf_values.size());
  }
}

void DepthLimitedCFR::EncodePublicStates() {
  for (const LeafPublicState& public_leaf : public_leaves_) {
    SPIEL_DCHECK_TRUE(public_leaf.IsConsistent());
    if (public_leaf.IsTerminal()) {
      encoded_leaves_.push_back(
          terminal_evaluator_->EncodeLeafPublicState(public_leaf));
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      encoded_leaves_.push_back(
          leaf_evaluator_->EncodeLeafPublicState(public_leaf));
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
  for (const std::unique_ptr<State>& state : node.corresponding_states()) {
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
    const double v = leaf->terminal_utility();
    const double chn = leaf->terminal_chance_reach_prob();
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

std::unique_ptr<EncodedPublicState> TerminalEvaluator::EncodeLeafPublicState(
    const LeafPublicState& state) const {
  return std::make_unique<TerminalPublicState>(state);
}

std::array<absl::Span<const float>, 2> TerminalEvaluator::EvaluatePublicState(
    EncodedPublicState* state,
    std::array<absl::Span<const float>, 2> ranges) const {
  auto* terminal = open_spiel::down_cast<TerminalPublicState*>(state);

  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    terminal->cfvs[0][i] = terminal->utilities[i] * ranges[1][j];
    terminal->cfvs[1][j] = -terminal->utilities[i] * ranges[0][i];
  }
  return {terminal->cfvs[0], terminal->cfvs[1]};
}

void DepthLimitedCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    PrepareRootReachProbs();
    propagators_[0].TopDown();
    propagators_[1].TopDown();

    EvaluateLeaves();
    propagators_[0].BottomUp();
    propagators_[1].BottomUp();
    SPIEL_DCHECK_FLOAT_NEAR(
        propagators_[0].RootCfValue(tracked_player_ranges_[0]),
        - propagators_[1].RootCfValue(tracked_player_ranges_[1]),
        1e-6);
  }
}

void DepthLimitedCFR::PrepareRootReachProbs() {
  for (int pl = 0; pl < 2; ++pl) {
    absl::Span<float> root_reaches = propagators_[pl].RootChildrenReachProbs();
    SPIEL_CHECK_EQ(tracked_player_ranges_[pl].size(), root_reaches.size());
    for (int i = 0; i < root_reaches.size(); ++i) {
      root_reaches[i] = tracked_player_ranges_[pl][i];
    }
  }
}

void DepthLimitedCFR::EvaluateLeaves() {
  SPIEL_CHECK_EQ(public_leaves_.size(), encoded_leaves_.size());
  for (int i = 0; i < public_leaves_.size(); ++i) {
    const LeafPublicState& public_leaf = public_leaves_[i];
    EncodedPublicState* encoded_leaf = encoded_leaves_[i].get();

    // TODO: similarly to tree Rebalance, we can do a "color-satisficing"
    //  permutation of the infostate tree, which will result into an ordering of
    //  CFRNodes such that they are properly grouped together in public states.
    //  Then we can drop the whole leaf_positions_ encoding and just use spans
    //  to communicate ranges and cfvs between the components efficiently.
    //  Maybe it will not work because we might need to permute decision nodes
    //  though...

    // 1. Prepare ranges
    auto vec_ranges = std::array<std::vector<float>, 2>();
    for (int pl = 0; pl < 2; pl++) {
      const int num_infostate_leafs = public_leaf.leaf_nodes[pl].size();
      vec_ranges[pl].reserve(num_infostate_leafs);
      for (const CFRNode* leaf_node : public_leaf.leaf_nodes[pl]) {
        const int leaf_position = leaf_positions_.at(leaf_node);
        SPIEL_DCHECK_GE(leaf_position, 0);
        SPIEL_DCHECK_LT(leaf_position, propagators_[pl].reach_probs.size());
        vec_ranges[pl].push_back(propagators_[pl].reach_probs[leaf_position]);
      }
    }

    // 2. Evaluate: compute cfvs.
    std::array<absl::Span<const float>, 2> cfvs;  // float due to neural nets
    std::array<absl::Span<const float>, 2> ranges = {vec_ranges[0],
                                                     vec_ranges[1]};
    if (public_leaf.IsTerminal()) {
      cfvs = terminal_evaluator_->EvaluatePublicState(encoded_leaf, ranges);
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      cfvs = leaf_evaluator_->EvaluatePublicState(encoded_leaf, ranges);
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
  CollectInfostateLookupTable(trees_[0].root(), &vec_ptable);
  CollectInfostateLookupTable(trees_[1].root(), &vec_ptable);
  return vec_ptable;
}
void DepthLimitedCFR::TrackPlayerRanges(
    std::array<absl::Span<const float>, 2> track_source) {
  tracked_player_ranges_ = track_source;
}

std::array<absl::Span<const float>, 2>
DepthLimitedCFR::RootChildrenCfValues() const {
  return {propagators_[0].RootChildrenCfValues()
         , propagators_[1].RootChildrenCfValues() };
}

bool LeafPublicState::IsConsistent() const {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // They should all be terminal or non-terminal.
  // The set of corresponding states must be the same across players.
  int num_terminals = 0, num_nonterminals = 0;

  using History = std::vector<Action>;
  std::unordered_set<History, absl::Hash<History>> state_histories;
  for (int pl = 0; pl < 2; ++pl) {
    for (const CFRNode* node : leaf_nodes[pl]) {
      if (!node->is_leaf_node()) return false;
      if (node->tree().acting_player() != pl) return false;
      if (node->type() == kTerminalInfostateNode) num_terminals++;
      else num_nonterminals++;

      for (const std::unique_ptr<State>& state : node->corresponding_states()) {
        const auto& h = state->History();
        if (pl == 0) {
          if (state_histories.find(h) != state_histories.end()) return false;
          state_histories.insert(h);
        } else {
          if (state_histories.find(h) == state_histories.end()) return false;
        }

        if (state->IsTerminal()) num_terminals++;
        else num_nonterminals++;
      }
    }
  }
  if (num_terminals > 0 && num_nonterminals > 0) return false;
  // We must count terminals twice (2 players).
  if (num_terminals % 2 != 0 && num_nonterminals % 2 != 0) return false;
  return true;
}

bool CheckChildPublicStateConsistency(
    const CFRPublicState& cfr_public_state, const LeafPublicState& leaf_state) {
  std::array<const CFRNode*, 2> roots = cfr_public_state.dlcfr->Roots();
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(leaf_state.leaf_nodes[pl].size(), roots[pl]->NumChildren());
    for (int i = 0; i < roots[pl]->NumChildren(); ++i) {
      const CFRNode& actual = *roots[pl]->ChildAt(i);
      const CFRNode& expected = *leaf_state.leaf_nodes[pl][i];
      SPIEL_CHECK_EQ(actual.infostate_string(), expected.infostate_string());
    }
  }
  return true;
}

std::unique_ptr<EncodedPublicState> CFREvaluator::EncodeLeafPublicState(
    const LeafPublicState& leaf_state) const {

  auto dlcfr = std::make_unique<DepthLimitedCFR>(
      game, CreateTrees({leaf_state.leaf_nodes[0], leaf_state.leaf_nodes[1]},
                        infostate_observer, depth_limit),
      leaf_evaluator, terminal_evaluator, public_observer);
  auto cfr_public_state = std::make_unique<CFRPublicState>(std::move(dlcfr));
  SPIEL_DCHECK_TRUE(
      CheckChildPublicStateConsistency(*cfr_public_state, leaf_state));
  return cfr_public_state;
}

std::array<absl::Span<const float>, 2> CFREvaluator::EvaluatePublicState(
    EncodedPublicState* public_state,
    std::array<absl::Span<const float>, 2> ranges) const {
  auto* cfr_state = open_spiel::down_cast<CFRPublicState*>(public_state);
  cfr_state->dlcfr->TrackPlayerRanges(ranges);
  cfr_state->dlcfr->RunSimultaneousIterations(num_cfr_iterations);
  return cfr_state->dlcfr->RootChildrenCfValues();
}

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel
