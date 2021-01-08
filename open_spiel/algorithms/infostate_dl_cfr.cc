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
#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/utils/format_observation.h"

namespace open_spiel {
namespace algorithms {
namespace dlcfr {

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game,
    std::vector<std::shared_ptr<InfostateTree>> depth_lim_trees,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer,
    std::vector<BanditVector> bandits
) :
    game_(std::move(game)),
    trees_(std::move(depth_lim_trees)),
    public_observer_(std::move(public_observer)),
    leaf_evaluator_(std::move(leaf_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    player_ranges_({
      std::vector<double>(trees_[0]->root_branching_factor(), 1.),
      std::vector<double>(trees_[1]->root_branching_factor(), 1.)
    }),
    reach_probs_({
                     std::vector<double>(trees_[0]->num_leaves(), 0.),
                     std::vector<double>(trees_[1]->num_leaves(), 0.)
                 }),
    cf_values_({
                   std::vector<double>(trees_[0]->num_leaves(), 0.),
                   std::vector<double>(trees_[1]->num_leaves(), 0.)
               }),
    bandits_(std::move(bandits)) {
  SPIEL_CHECK_TRUE(public_observer_->HasTensor());
  PrepareLeafNodesForPublicStates();
  PrepareRangesAndValuesForPublicStates();
  CreateContexts();
}

DepthLimitedCFR::DepthLimitedCFR(
    std::shared_ptr<const Game> game, int max_depth_limit,
    std::shared_ptr<const LeafEvaluator> leaf_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator
) {
  auto trees = {MakeInfostateTree(*game, 0, max_depth_limit),
                MakeInfostateTree(*game, 1, max_depth_limit)};
  // TODO: fix.
  new(this) DepthLimitedCFR(game, trees,
                            std::move(leaf_evaluator),
                            std::move(terminal_evaluator),
                            game->MakeObserver(kPublicStateObsType, {}),
                            MakeBanditVectors(trees));
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
      LeafPublicState* leaf_state = GetPublicLeaf(public_observation);
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
      s.ranges[pl] = std::vector<double>(num_leaves, 0.);
      s.values[pl] = std::vector<double>(num_leaves, 0.);
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
    const Observation& public_observation) {
  for (LeafPublicState& state : public_leaves_) {
    if (state.public_tensor == public_observation) return &state;
  }
  // None found: create and return the pointer.
  public_leaves_.emplace_back(public_observation);
  LeafPublicState* state = &public_leaves_.back();
  state->public_id = public_leaves_.size() - 1;
  return state;
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

std::shared_ptr<Policy> DepthLimitedCFR::AveragePolicy() {
  return std::make_shared<BanditsAveragePolicy>(trees_, bandits_);
}
std::shared_ptr<Policy> DepthLimitedCFR::CurrentPolicy() {
  return std::make_shared<BanditsCurrentPolicy>(trees_, bandits_);
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
    const InfostateNode* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const InfostateNode* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const InfostateNode* leaf = leaf_nodes[0][i];
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
    state->values[1][j] = - terminal->utilities[i] * state->ranges[0][i];
  }
}

void DepthLimitedCFR::UpdateReachProbs() {
  PrepareRootReachProbs();
  for (int pl = 0; pl < 2; ++pl) {
    TopDown(*trees_[pl], bandits_[pl],
            absl::MakeSpan(reach_probs_[pl]), num_iterations_);
  }
}

void DepthLimitedCFR::UpdateTrunk() {
  for (int pl = 0; pl < 2; ++pl) {
    BottomUp(*trees_[pl], bandits_[pl], absl::MakeSpan(cf_values_[pl]));
  }
}

void DepthLimitedCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;
    UpdateReachProbs();
    EvaluateLeaves();
    UpdateTrunk();
//    SPIEL_DCHECK_FLOAT_NEAR(RootValue(/*pl=*/0), -RootValue(/*pl=*/1), 1e-6);
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

void DepthLimitedCFR::SetPlayerRanges(
    const std::array<std::vector<double>, 2>& ranges) {
  player_ranges_ = ranges;
}
std::array<absl::Span<const double>, 2> DepthLimitedCFR::RootChildrenCfValues()
const {
  return {
      absl::MakeConstSpan(&cf_values_[0][0], trees_[0]->root_branching_factor()),
      absl::MakeConstSpan(&cf_values_[1][0], trees_[1]->root_branching_factor())
  };
}

void DepthLimitedCFR::Reset() {
  // Reset trunk
  num_iterations_ = 0;
  for (int pl = 0; pl < 2; ++pl) {
    std::fill(cf_values_[pl].begin(), cf_values_[pl].end(), 0.);
    std::fill(reach_probs_[pl].begin(), reach_probs_[pl].end(), 0.);
    std::fill(player_ranges_[pl].begin(), player_ranges_[pl].end(), 1.);
  }
  for (BanditVector& bandits : bandits_) {
    for (DecisionId id : bandits.range()) {
      bandits[id]->Reset();
    }
  }
  // Reset subgames
  for (int i = 0; i < public_leaves_.size(); ++i) {
    LeafPublicState& state = public_leaves_[i];
    for (int pl = 0; pl < 2; ++pl) {
      std::fill(state.ranges[pl].begin(), state.ranges[pl].end(), 0.);
      std::fill(state.values[pl].begin(), state.values[pl].end(), 0.);
    }
    std::unique_ptr<PublicStateContext>& context = contexts_[i];
    if (!state.IsTerminal() && context && leaf_evaluator_) {
      leaf_evaluator_->ResetContext(context.get());
    }
  }
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

void DebugPrintPublicFeatures(const std::vector<LeafPublicState>& states) {
  std::cout << "# Public features:\n";
  for (int i = 0; i < states.size(); ++i) {
    std::cout << "#   states[" << i << "].public_tensor\n#     "
              << ObservationToString(states[i].public_tensor, "\n#     ")
              << "\n";
  }
}

bool CheckChildPublicStateConsistency(
    const CFRContext& cfr_public_state, const LeafPublicState& leaf_state) {
  auto trees = cfr_public_state.dlcfr->trees();
  for (int pl = 0; pl < 2; ++pl) {
    const InfostateNode& root = trees[pl]->root();
    SPIEL_CHECK_EQ(leaf_state.leaf_nodes[pl].size(), root.num_children());
    for (int i = 0; i < root.num_children(); ++i) {
      const InfostateNode& actual = *root.child_at(i);
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
  auto subgame_trees = std::vector{
      MakeInfostateTree(state.leaf_nodes[0], depth_limit),
      MakeInfostateTree(state.leaf_nodes[1], depth_limit)
  };
  auto subgame_bandits = MakeBanditVectors(subgame_trees, bandit_name);
  auto dlcfr = std::make_unique<DepthLimitedCFR>(
      game, subgame_trees, leaf_evaluator, terminal_evaluator, public_observer,
      std::move(subgame_bandits));
  auto cfr_public_state = std::make_unique<CFRContext>(std::move(dlcfr));
  SPIEL_DCHECK_TRUE(CheckChildPublicStateConsistency(*cfr_public_state, state));
  return cfr_public_state;
}

void CFREvaluator::ResetContext(PublicStateContext* context) const {
  auto* cfr_state = open_spiel::down_cast<CFRContext*>(context);
  cfr_state->dlcfr->Reset();
}

void CFREvaluator::EvaluatePublicState(LeafPublicState* public_state,
                                       PublicStateContext* context) const {
  auto* cfr_state = open_spiel::down_cast<CFRContext*>(context);
  DepthLimitedCFR* dlcfr = cfr_state->dlcfr.get();
  if (reset_subgames_on_evaluation) {
    dlcfr->Reset();
  }
  dlcfr->SetPlayerRanges(public_state->ranges);
  dlcfr->RunSimultaneousIterations(num_cfr_iterations);
  std::array<absl::Span<const double>, 2> resulting_values =
      dlcfr->RootChildrenCfValues();
  // Copy the results.
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(resulting_values[pl].begin(),
              resulting_values[pl].end(),
              public_state->values[pl].begin());
  }
}

double DepthLimitedCFR::RootValue(Player pl) const {
  const int root_branching = trees_[pl]->root_branching_factor();
  return RootCfValue(
      root_branching, absl::MakeConstSpan(&cf_values_[pl][0], root_branching),
      player_ranges_[pl]);
}

// -- Range table --------------------------------------------------------------

size_t RangeTable::largest_range() const { return private_hands.size(); }

size_t RangeTable::hand_index(const Observation& hand) {
  auto it = std::find(private_hands.begin(), private_hands.end(), hand);
  if (it == private_hands.end()) {
    private_hands.push_back(hand);
    return private_hands.size() - 1;
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

bool AllInfoStatesHaveDistinctHands(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    Player pl, const dlcfr::LeafPublicState& state) {
  const std::vector<const InfostateNode*>& info_states = state.leaf_nodes[pl];
  std::unordered_map<Observation, const InfostateNode*> hands_for_infostates;

  Observation hand(game, hand_observer);
  for (const InfostateNode* info_state : info_states) {
    const State& some_state = *info_state->corresponding_states().at(0);
    hand.SetFrom(some_state, pl);
    if (hands_for_infostates.find(hand) == hands_for_infostates.end()) {
      hands_for_infostates[hand] = info_state;
    } else {
      std::cerr << "Not all hands are unique in public state: \n"
                << ObservationToString(state.public_tensor) << "\n"
                << "Printing out the hands.\n-----\n";
      for (const auto& [hand, info_state] : hands_for_infostates) {
        std::cerr << "Infostate string: " << info_state->infostate_string() << "\n"
                  << "Hand observation: " << ObservationToString(hand) << "\n"
                  << "Some history in infostate: "
                  << info_state->corresponding_states()[0]->HistoryString() << "\n"
                  << "-----\n";
      }
      std::cerr << "Offending infostate: \n"
                << "Infostate string: " << info_state->infostate_string() << "\n"
                << "Hand observation: " << ObservationToString(hand) << "\n"
                << "Some history in infostate: "
                << info_state->corresponding_states()[0]->HistoryString() << "\n"
                << "-----\n";
      return false;
    }
  }
  return true;
}

bool AllStatesHaveSameHands(const Observation& expected_hand, Player player,
                            const std::vector<std::unique_ptr<State>>& states) {
  Observation actual_hand(expected_hand);
  for (const std::unique_ptr<State>& state : states) {
    actual_hand.SetFrom(*state, player);
    if (actual_hand != expected_hand) {
      return false;
    }
  }
  return true;
}

std::vector<RangeTable> CreateRangeTables(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves) {
  std::vector<RangeTable> tables{public_leaves.size(), public_leaves.size()};
  Observation hand(game, hand_observer);
  for (int state_idx = 0; state_idx < public_leaves.size(); ++state_idx) {
    const dlcfr::LeafPublicState& state = public_leaves[state_idx];
    // Terminal states are not handled by non-terminal leaf evaluators,
    // so we don't need to create range table for them.
    if (state.IsTerminal()) {
      continue;
    }

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_DCHECK_TRUE(  // Holds within public state.
          AllInfoStatesHaveDistinctHands(game, hand_observer, pl, state));

      for (int tree_idx = 0;
           tree_idx < state.leaf_nodes[pl].size(); ++tree_idx) {
        const InfostateNode* node = state.leaf_nodes[pl][tree_idx];
        const State& some_state = *node->corresponding_states().at(0);
        hand.SetFrom(some_state, pl);

        SPIEL_DCHECK_TRUE(  // Should hold for all states within an infostate.
            AllStatesHaveSameHands(hand, pl, node->corresponding_states()));

        size_t net_idx = tables[pl].hand_index(hand);
        tables[pl].bijections[state_idx].put({tree_idx, net_idx});
      }
    }
  }
  return tables;
}

void DebugPrintRangeTables(const std::vector<RangeTable>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# List of private hands for player " << pl << "\n";
    const RangeTable& table = tables[pl];
    for (int i = 0; i < table.private_hands.size(); ++i) {
      std::cout << "#   private_hand[" << i << "]:\n#      "
                << ObservationToString(table.private_hands[i], "\n#      ")
                << "\n";
    }

    std::cout << "# List of bijections (tree <-> net) for player "
              << pl << "\n";
    for (size_t i = 0; i < table.bijections.size(); ++i) {
      std::cout << "#  Public state " << i << "\n";
      const std::map<size_t, size_t>& tree_to_net =
          table.bijections[i].tree_to_net();
      for (auto&[key, val] : tree_to_net) {
        std::cout << "#   " << key << " -> " << val << "\n";
      }
    }
  }
}

}  // namespace dlcfr
}  // namespace algorithms
}  // namespace open_spiel
