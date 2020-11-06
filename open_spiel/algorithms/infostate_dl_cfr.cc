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

std::unordered_map<const InfostateNode*, CFRInfoStateValues> CreateTable(
    const std::array<std::unique_ptr<InfostateTree>, 2>& trees) {
  std::unordered_map<const InfostateNode*, CFRInfoStateValues> map;
  for (int pl = 0; pl < 2; ++pl) {
    const std::vector<std::vector<InfostateNode*>>& nodes =
        trees[pl]->nodes_at_depths();
    for (int d = 0; d < nodes.size() - 1; ++d) {
      for (const InfostateNode* node : nodes[d]) {
        if (node->type() != kDecisionInfostateNode) continue;
        map[node] = CFRInfoStateValues(node->legal_actions());
      }
    }
  }
  return map;
}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game,
    std::array<std::unique_ptr<InfostateTree>, 2> trees,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer
) :
    game_(std::move(game)),
    trees_(std::move(trees)),
    cf_values_({
      std::vector<float>(trees_[0]->num_leaves(), 0.),
      std::vector<float>(trees_[1]->num_leaves(), 0.)
    }),
    reach_probs_({
      std::vector<float>(trees_[0]->num_leaves(), 0.),
      std::vector<float>(trees_[1]->num_leaves(), 0.)
    }),
    public_observer_(std::move(public_observer)),
    leaf_evaluator_(std::move(leaf_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    player_ranges_({
      std::vector<float>(trees_[0]->root_branching_factor(), 1.),
      std::vector<float>(trees_[1]->root_branching_factor(), 1.)
    }),
    node_values_(CreateTable(trees_)) {
  PrepareLeafNodesForPublicStates();
  PrepareRangesAndValuesForPublicStates();
  CreateContexts();
}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game, int max_depth_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator
) : DepthLimitedCFR(game,
                    {MakeInfostateTree(*game, 0, max_depth_limit),
                     MakeInfostateTree(*game, 1, max_depth_limit)},
                    std::move(leaf_evaluator),
                    std::move(terminal_evaluator),
                    game->MakeObserver(kPublicStateObsType, {})) {}

void FillStatesAndChanceRange(std::vector<const State*>* start_states,
                              std::vector<float>* chance_reach_probs,
                              absl::Span<const InfostateNode* const> start_nodes) {
  // Collect pointers to starting states, along with their reach probs.
  // It's enough to do this just through one player, as the other player
  // has just a permutation of these states.
  for (const InfostateNode* cfr_node : start_nodes) {
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

std::array<std::unique_ptr<InfostateTree>, 2> CreateTrees(
    std::array<absl::Span<const InfostateNode* const>, 2> start_nodes,
    const std::shared_ptr<Observer>& infostate_observer,
    int max_move_limit) {
  std::array<std::vector<const State*>, 2> start_states;
  std::array<std::vector<float>, 2> chance_reach_probs;

  FillStatesAndChanceRange(&start_states[0], &chance_reach_probs[0],
                           start_nodes[0]);
  FillStatesAndChanceRange(&start_states[1], &chance_reach_probs[1],
                           start_nodes[1]);

  return {
      MakeInfostateTree(start_states[0], chance_reach_probs[0],
                        infostate_observer, /*acting_player=*/0, max_move_limit),
      MakeInfostateTree(start_states[1], chance_reach_probs[1],
                        infostate_observer, /*acting_player=*/1, max_move_limit)
  };
}

void DepthLimitedCFR::PrepareLeafNodesForPublicStates() {
  Observation public_observation(*game_, public_observer_);

  for (int pl = 0; pl < 2; ++pl) {
    int leaf_position = 0;
    for (InfostateNode* leaf_node : trees_[pl]->leaf_nodes()) {
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

    SPIEL_CHECK_EQ(leaf_position, trees_[pl]->num_leaves());
  }
}

void DepthLimitedCFR::PrepareRangesAndValuesForPublicStates() {
  for (LeafPublicState& s : public_leaves_) {
    for (int pl = 0; pl < 2; ++pl) {
      const int num_leaves = s.leaf_nodes[pl].size();
      s.ranges[pl] = std::vector<float>(num_leaves, 0.);
      s.values[pl] = std::vector<float>(num_leaves, 0.);
    }
  }
}

void DepthLimitedCFR::CreateContexts() {
  for (const LeafPublicState& public_leaf : public_leaves_) {
    SPIEL_DCHECK_TRUE(public_leaf.IsConsistent());
    if (public_leaf.IsTerminal()) {
      contexts_.push_back(terminal_evaluator_->CreateContext(public_leaf));
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      contexts_.push_back(leaf_evaluator_->CreateContext(public_leaf));
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
    const InfostateNode& node, absl::Span<float> expected_observation) {
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

TerminalPublicStateContext::TerminalPublicStateContext(
    const LeafPublicState& state) {
  auto& leaf_nodes = state.leaf_nodes;
  SPIEL_CHECK_EQ(leaf_nodes[0].size(), leaf_nodes[1].size());
  const int num_terminals = leaf_nodes[0].size();
  utilities.reserve(num_terminals);
  permutation.reserve(num_terminals);

  using History = absl::Span<const Action>;
  std::map<History, int> player1_map;
  for (int i = 0; i < num_terminals; ++i) {
    player1_map[leaf_nodes[1][i]->TerminalHistory()] = i;
  }
  SPIEL_CHECK_EQ(player1_map.size(), leaf_nodes[1].size());

  for (int i = 0; i < num_terminals; ++i) {
    const InfostateNode const* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const InfostateNode const* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const InfostateNode const* leaf = leaf_nodes[0][i];
    const double v = leaf->terminal_utility();
    const double chn = leaf->terminal_chance_reach_prob();
    utilities.push_back(v * chn);
    permutation.push_back(permutation_index);
  }

  // A quick check to see if the permutation is ok
  // by computing the arithmetic sum.
  SPIEL_DCHECK_EQ(
      std::accumulate(permutation.begin(),
                      permutation.end(), 0),
      num_terminals * (num_terminals - 1) / 2);
}

std::unique_ptr<PublicStateContext> TerminalEvaluator::CreateContext(
    const LeafPublicState& state) const {
  return std::make_unique<TerminalPublicStateContext>(state);
}

void TerminalEvaluator::EvaluatePublicState(
    LeafPublicState* state, PublicStateContext* context) const {
  auto* terminal = open_spiel::down_cast<TerminalPublicStateContext*>(context);
  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    state->values[0][i] = terminal->utilities[i] * state->ranges[1][j];
    state->values[1][j] = -terminal->utilities[i] * state->ranges[0][i];
  }
}

void DepthLimitedCFR::SimultaneousTopDownEvaluate() {
  PrepareRootReachProbs();
  TopDown(trees_[0]->nodes_at_depths(), node_values_, absl::MakeSpan(reach_probs_[0]));
  TopDown(trees_[1]->nodes_at_depths(), node_values_, absl::MakeSpan(reach_probs_[1]));
  EvaluateLeaves();
}

void DepthLimitedCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    SimultaneousTopDownEvaluate();
    BottomUp(trees_[0]->nodes_at_depths(), node_values_, absl::MakeSpan(cf_values_[0]));
    BottomUp(trees_[1]->nodes_at_depths(), node_values_, absl::MakeSpan(cf_values_[1]));
    SPIEL_DCHECK_FLOAT_NEAR(RootValue(/*pl=*/0), -RootValue(/*pl=*/1), 1e-6);
  }
}

void DepthLimitedCFR::PrepareRootReachProbs() {
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(player_ranges_[pl].begin(), player_ranges_[pl].end(),
              reach_probs_[pl].begin());
  }
}

void DepthLimitedCFR::EvaluateLeaves() {
  SPIEL_CHECK_EQ(public_leaves_.size(), contexts_.size());
  for (int i = 0; i < public_leaves_.size(); ++i) {
    LeafPublicState& state = public_leaves_[i];
    PublicStateContext* context = contexts_[i].get();

    // 1. Prepare ranges
    for (int pl = 0; pl < 2; pl++) {
      const int num_leaves = state.leaf_nodes[pl].size();
      for (int j = 0; j < num_leaves; ++j) {
        const InfostateNode* leaf_node = state.leaf_nodes[pl][j];
        const int trunk_position = leaf_positions_.at(leaf_node);
        SPIEL_DCHECK_GE(trunk_position, 0);
        SPIEL_DCHECK_LT(trunk_position, trees_[pl]->num_leaves());
        // Copy range from the trunk to the leaf public state.
        state.ranges[pl][j] = reach_probs_[pl][trunk_position];
      }
    }

    // 2. Evaluate: compute cfvs.
    if (state.IsTerminal()) {
      SPIEL_CHECK_TRUE(terminal_evaluator_);
      terminal_evaluator_->EvaluatePublicState(&state, context);
    } else {
      SPIEL_CHECK_TRUE(leaf_evaluator_);
      leaf_evaluator_->EvaluatePublicState(&state, context);
    }

    // 3. Update cfvs for propagators.
    for (int pl = 0; pl < 2; pl++) {
      const int num_leaves = state.leaf_nodes[pl].size();
      for (int j = 0; j < num_leaves; ++j) {
        const InfostateNode* leaf_node = state.leaf_nodes[pl][j];
        const int trunk_position = leaf_positions_.at(leaf_node);
        SPIEL_DCHECK_GE(trunk_position, 0);
        SPIEL_DCHECK_LT(trunk_position, trees_[pl]->num_leaves());
        // Copy value from the leaf public state to the trunk.
        cf_values_[pl][trunk_position] = state.values[pl][j];
      }
    }
  }
}

CFRInfoStateValuesPtrTable DepthLimitedCFR::InfoStateValuesPtrTable() {
  CFRInfoStateValuesPtrTable vec_ptable;
  for (auto& [ptr, value] : node_values_) {
    vec_ptable[ptr->infostate_string()] = &value;
  }
  return vec_ptable;
}
void DepthLimitedCFR::SetPlayerRanges(
    const std::array<std::vector<float>, 2>& ranges) {
  player_ranges_ = ranges;
}
std::array<absl::Span<const float>, 2> DepthLimitedCFR::RootChildrenCfValues()
const {
  return {
      absl::MakeConstSpan(&cf_values_[0][0], trees_[0]->root_branching_factor()),
      absl::MakeConstSpan(&cf_values_[1][0], trees_[1]->root_branching_factor())
  };
}

bool LeafPublicState::IsConsistent() const {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // They should all be terminal or non-terminal.
  // The set of corresponding states must be the same across players.
  int num_terminals = 0, num_nonterminals = 0;

  using History = std::vector<Action>;
  std::unordered_set<History, absl::Hash<History>> state_histories;
  for (int pl = 0; pl < 2; ++pl) {
    for (const InfostateNode* node : leaf_nodes[pl]) {
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
bool LeafPublicState::IsTerminal() const {
  return leaf_nodes[0][0]->type() == kTerminalInfostateNode;
}

bool CheckChildPublicStateConsistency(
    const CFRPublicState& cfr_public_state, const LeafPublicState& leaf_state) {
  std::array<const InfostateNode*, 2> roots = cfr_public_state.dlcfr->Roots();
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(leaf_state.leaf_nodes[pl].size(), roots[pl]->num_children());
    for (int i = 0; i < roots[pl]->num_children(); ++i) {
      const InfostateNode& actual = *roots[pl]->child_at(i);
      const InfostateNode& expected = *leaf_state.leaf_nodes[pl][i];
      SPIEL_CHECK_EQ(actual.infostate_string(), expected.infostate_string());
    }
  }
  return true;
}

// -- CFR evaluator ------------------------------------------------------------

CFREvaluator::CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
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

std::unique_ptr<PublicStateContext> CFREvaluator::CreateContext(
    const LeafPublicState& state) const {
  auto dlcfr = std::make_unique<DepthLimitedCFR>(
      game, CreateTrees({state.leaf_nodes[0], state.leaf_nodes[1]},
                        infostate_observer, depth_limit),
      leaf_evaluator, terminal_evaluator, public_observer);
  auto cfr_public_state = std::make_unique<CFRPublicState>(std::move(dlcfr));
  SPIEL_DCHECK_TRUE(
      CheckChildPublicStateConsistency(*cfr_public_state, state));
  return cfr_public_state;
}

void CFREvaluator::EvaluatePublicState(LeafPublicState* public_state,
                                       PublicStateContext* context) const {
  auto* cfr_state = open_spiel::down_cast<CFRPublicState*>(context);
  DepthLimitedCFR* dlcfr = cfr_state->dlcfr.get();
  dlcfr->SetPlayerRanges(public_state->ranges);
  dlcfr->RunSimultaneousIterations(num_cfr_iterations);
  std::array<absl::Span<const float>, 2> resulting_values =
      dlcfr->RootChildrenCfValues();
  // Copy the results.
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(resulting_values[pl].begin(),
              resulting_values[pl].end(),
              public_state->values[pl].begin());
  }
}

// -- Counterfactul Best Response ----------------------------------------------

float DepthLimitedCFR::TrunkExploitability() const {
  return (CfBestResponse(0) + CfBestResponse(1)) / 2.;
}

float DepthLimitedCFR::CfBestResponse(
    const InfostateNode& node, Player pl, int* leaf_index) const {
  if (node.is_leaf_node()) {
    return cf_values_[pl][(*leaf_index)++];
  }
  if (node.type() == kObservationInfostateNode) {
    double sum_value = 0.;
    for (const InfostateNode* child : node.child_iterator()) {
      sum_value += CfBestResponse(*child, pl, leaf_index);
    }
    return sum_value;
  }
  SPIEL_CHECK_EQ(node.type(), kDecisionInfostateNode);
  double max_value = -std::numeric_limits<float>::infinity();
  for (const InfostateNode* child : node.child_iterator()) {
    max_value = std::fmax(CfBestResponse(*child, pl, leaf_index),
                          max_value);
  }
  return max_value;
}

float DepthLimitedCFR::CfBestResponse(Player responding_player) const {
  int leaf_index = 0;
  const InfostateNode& root_node = trees_[responding_player]->root();
  return CfBestResponse(root_node, responding_player, &leaf_index);
}
float DepthLimitedCFR::RootValue(Player pl) const {
  const int root_branching = trees_[pl]->root_branching_factor();
  return RootCfValue(
      root_branching, absl::MakeConstSpan(&cf_values_[pl][0], root_branching),
      player_ranges_[pl]);
}
std::array<const InfostateNode*, 2> DepthLimitedCFR::Roots() const {
  return {&trees_[0]->root(), &trees_[1]->root()};
}
std::array<std::unique_ptr<InfostateTree>, 2>& DepthLimitedCFR::Trees() { return trees_; }
std::vector<std::unique_ptr<PublicStateContext>>& DepthLimitedCFR::GetContexts() {
  return contexts_;
}
std::vector<LeafPublicState>& DepthLimitedCFR::GetPublicLeaves() {
  return public_leaves_;
}

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel
