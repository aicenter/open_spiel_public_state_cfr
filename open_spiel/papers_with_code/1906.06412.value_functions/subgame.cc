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
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/utils/format_observation.h"

namespace open_spiel {
namespace papers_with_code {

namespace {

void CheckConsistency(const PublicState& s) {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // They should all be terminal or non-terminal.
  // The set of corresponding states must be the same across players.
  int num_terminals = 0, num_nonterminals = 0;

  using History = std::vector<Action>;
  std::unordered_set<History, absl::Hash<History>> state_histories;
  for (int pl = 0; pl < 2; ++pl) {
    for (const algorithms::InfostateNode* node : s.nodes[pl]) {
      SPIEL_CHECK_TRUE(!s.IsLeaf() || node->is_leaf_node());
      SPIEL_CHECK_EQ(node->tree().acting_player(), pl);
      if (node->type() == algorithms::kTerminalInfostateNode) num_terminals++;
      else num_nonterminals++;

      for (const std::unique_ptr<State>& state : node->corresponding_states()) {
        const auto& h = state->History();
        if (pl == 0) {
          SPIEL_CHECK_TRUE(state_histories.find(h) == state_histories.end());
          state_histories.insert(h);
        } else {
          SPIEL_CHECK_TRUE(state_histories.find(h) != state_histories.end());
        }

        if (state->IsTerminal()) num_terminals++;
        else num_nonterminals++;
      }
    }
  }
  SPIEL_CHECK_FALSE(num_terminals > 0 && num_nonterminals > 0);
  // We must count terminals twice (2 players).
  SPIEL_CHECK_FALSE(num_terminals % 2 != 0 && num_nonterminals % 2 != 0);
  // All OK! Yay!
}

bool DoStatesProduceEqualPublicObservations(
    const Game& game, std::shared_ptr<Observer> public_observer,
    const algorithms::InfostateNode& node, absl::Span<float> expected_observation) {
  Observation public_observation(game, public_observer);

  // Check that indeed all states produce the same public observations.
  for (const std::unique_ptr<State>& state : node.corresponding_states()) {
    public_observation.SetFrom(*state, kDefaultPlayerId);
    if (public_observation.Tensor() != expected_observation) return false;
  }
  return true;
}

void MakeReachesAndValuesForPublicStates(std::vector<PublicState>& states) {
  for (PublicState& state : states) {
    for (int pl = 0; pl < 2; ++pl) {
      const int num_nodes = state.nodes[pl].size();
      state.beliefs[pl] = std::vector<double>(num_nodes, 0.);
      state.values[pl] = std::vector<double>(num_nodes, 0.);
    }
  }
}

}  // namespace

Subgame::Subgame(
    std::shared_ptr<const Game> game,
    std::vector<std::shared_ptr<algorithms::InfostateTree>> depth_lim_trees,
    std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
    std::shared_ptr<Observer> public_observer,
    std::vector<algorithms::BanditVector> bandits
) :
    game_(std::move(game)),
    trees_(std::move(depth_lim_trees)),
    public_observer_(std::move(public_observer)),
    nonterminal_evaluator_(std::move(nonterminal_evaluator)),
    terminal_evaluator_(std::move(terminal_evaluator)),
    beliefs_({
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
  SPIEL_CHECK_EQ(trees_[0]->storage_policy() & kDlCfrInfostateTreeStorage,
                 kDlCfrInfostateTreeStorage);
  SPIEL_CHECK_EQ(trees_[1]->storage_policy() & kDlCfrInfostateTreeStorage,
                 kDlCfrInfostateTreeStorage);
  PrepareInfostateNodesForPublicStates();
  PrepareReachesAndValuesForPublicStates();
  CreateContexts();
}

Subgame::Subgame(
    std::shared_ptr<const Game> game, int max_depth_limit,
    std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator
) {
  auto trees = {algorithms::MakeInfostateTree(*game, 0, max_depth_limit, kDlCfrInfostateTreeStorage),
                algorithms::MakeInfostateTree(*game, 1, max_depth_limit, kDlCfrInfostateTreeStorage)};
  // FIXME: fix.
  new(this) Subgame(game, trees,
                    std::move(nonterminal_evaluator),
                    std::move(terminal_evaluator),
                    game->MakeObserver(kPublicStateObsType, {}),
                    algorithms::MakeBanditVectors(trees));
}

void Subgame::PrepareInfostateNodesForPublicStates() {
  Observation public_observation(*game_, public_observer_);
  // Save node positions for initial (root) public state.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < trees_[pl]->root_branching_factor(); ++i) {
      algorithms::InfostateNode* root_node = trees_[pl]->root().child_at(i);
      SPIEL_CHECK_TRUE(root_node->is_root_child());
      PublicState* init_state = GetPublicState(public_observation,
                                               kInitialPublicState, root_node);
      init_state->nodes[pl].push_back(root_node);
      root_node_positions_[root_node] = i;  // TODO: do we need this?
    }
  }

  // Make sure we have built only one initial state:
  // the infostate trees are rooted in a single initial state.
  // While more initial states are possible, we don't do this as it would
  // complicate the code unnecessarily.
  SPIEL_CHECK_EQ(public_states_.size(), 1);

  // Save node positions for leaf public states.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < trees_[pl]->num_leaves(); ++i) {
      algorithms::InfostateNode* leaf_node = trees_[pl]->leaf_nodes()[i];
      SPIEL_CHECK_TRUE(leaf_node->is_leaf_node());
      PublicState* leaf_state = GetPublicState(public_observation,
                                               kLeafPublicState, leaf_node);
      leaf_state->nodes[pl].push_back(leaf_node);
      leaf_node_positions_[leaf_node] = i;
    }
  }
}

void Subgame::PrepareReachesAndValuesForPublicStates() {
  MakeReachesAndValuesForPublicStates(public_states_);
}

void Subgame::CreateContexts() {
  for (const PublicState& state : public_states_) {
    SPIEL_DCHECK(CheckConsistency(state));
    if (state.IsTerminal()) {
      contexts_.push_back(terminal_evaluator_->CreateContext(state));
    } else {
      contexts_.push_back(nonterminal_evaluator_
                          ? nonterminal_evaluator_->CreateContext(state)
                          : nullptr);
    }
  }
}

PublicState* Subgame::GetPublicState(Observation& public_observation,
                                     PublicStateType state_type,
                                     algorithms::InfostateNode* node) {
  SPIEL_CHECK_FALSE(node->corresponding_states().empty());
  const std::unique_ptr<State>& some_state = node->corresponding_states()[0];
  public_observation.SetFrom(*some_state, kDefaultPlayerId);
  SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
      *game_, public_observer_, *node, public_observation.Tensor()));
  PublicState* state = GetPublicState(public_observation, state_type);
  if(state->move_number == -1) {
    state->move_number = some_state->MoveNumber();
  } else {
    SPIEL_CHECK_EQ(state->move_number, some_state->MoveNumber());
  }
  return state;
}

PublicState* Subgame::GetPublicState(
    const Observation& public_observation,
    PublicStateType state_type) {
  for (PublicState& state : public_states_) {
    if (state.public_tensor == public_observation
        && state.state_type == state_type) return &state;
  }
  // None found: create and return the pointer.
  public_states_.emplace_back(public_observation,
                              state_type, public_states_.size());
  return &public_states_.back();
}

std::shared_ptr<Policy> Subgame::AveragePolicy() {
  return std::make_shared<algorithms::BanditsAveragePolicy>(trees_, bandits_);
}
std::shared_ptr<Policy> Subgame::CurrentPolicy() {
  return std::make_shared<algorithms::BanditsCurrentPolicy>(trees_, bandits_);
}

std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator() {
  return std::make_shared<TerminalEvaluator>();
}

std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator() {
  return std::make_shared<DummyEvaluator>();
}

TerminalPublicStateContext::TerminalPublicStateContext(
    const PublicState& state) {
  SPIEL_CHECK_TRUE(state.IsTerminal());
  auto& leaf_nodes = state.nodes;
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
    const algorithms::InfostateNode* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const algorithms::InfostateNode* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const algorithms::InfostateNode* leaf = leaf_nodes[0][i];
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
    const PublicState& state) const {
  return std::make_unique<TerminalPublicStateContext>(state);
}

void TerminalEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  auto* terminal = open_spiel::down_cast<TerminalPublicStateContext*>(context);
  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    state->values[0][i] = terminal->utilities[i] * state->beliefs[1][j];
    state->values[1][j] = - terminal->utilities[i] * state->beliefs[0][i];
  }
}

void Subgame::UpdateReachProbs() {
  PrepareRootReachProbs();
  for (int pl = 0; pl < 2; ++pl) {
    TopDown(*trees_[pl], bandits_[pl],
            absl::MakeSpan(reach_probs_[pl]), num_iterations_);
  }
}

void Subgame::UpdateTrunk() {
  for (int pl = 0; pl < 2; ++pl) {
    BottomUp(*trees_[pl], bandits_[pl], absl::MakeSpan(cf_values_[pl]));
  }
}

void Subgame::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;
    UpdateReachProbs();
    EvaluateLeaves();
    UpdateTrunk();
//    SPIEL_DCHECK_FLOAT_NEAR(RootValue(/*pl=*/0), -RootValue(/*pl=*/1), 1e-6);
  }
}

void Subgame::PrepareRootReachProbs() {
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(beliefs_[pl].begin(), beliefs_[pl].end(),
              reach_probs_[pl].begin());
  }
}

void Subgame::EvaluateLeaves() {
  SPIEL_CHECK_EQ(public_states_.size(), contexts_.size());
  for (int i = 0; i < public_states_.size(); ++i) {
    PublicState* state = &public_states_[i];
    if (!state->IsLeaf()) continue;
    PublicStateContext* context = contexts_[i].get();
    EvaluateLeaf(state, context);
  }
}

void Subgame::EvaluateLeaf(PublicState* state,
                           PublicStateContext* context) {
  SPIEL_CHECK_TRUE(state);
  SPIEL_CHECK_TRUE(state->IsLeaf());

  // 1. Prepare beliefs
  for (int pl = 0; pl < 2; pl++) {
    const int num_leaves = state->nodes[pl].size();
    for (int j = 0; j < num_leaves; ++j) {
      const algorithms::InfostateNode* leaf_node = state->nodes[pl][j];
      const int trunk_position = leaf_node_positions_.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, trees_[pl]->num_leaves());
      // Copy reach prob (player belief) from the trunk
      // to the leaf public state->
      state->beliefs[pl][j] = reach_probs_[pl][trunk_position];
    }
  }

  // 2. Evaluate: compute cfvs.
  if (state->IsTerminal()) {
    SPIEL_CHECK_TRUE(terminal_evaluator_);
    terminal_evaluator_->EvaluatePublicState(state, context);
  } else {
    SPIEL_CHECK_TRUE(nonterminal_evaluator_);
    nonterminal_evaluator_->EvaluatePublicState(state, context);
  }

  // 3. Update cfvs for propagators.
  for (int pl = 0; pl < 2; pl++) {
    const int num_leaves = state->nodes[pl].size();
    for (int j = 0; j < num_leaves; ++j) {
      const algorithms::InfostateNode* leaf_node = state->nodes[pl][j];
      const int trunk_position = leaf_node_positions_.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, trees_[pl]->num_leaves());
      // Copy value from the leaf public state to the trunk.
      cf_values_[pl][trunk_position] = state->values[pl][j];
    }
  }
}

// TODO: use initial state for this storage!
void Subgame::SetBeliefs(
    const std::array<std::vector<double>, 2>& beliefs) {
  beliefs_ = beliefs;
}
std::array<absl::Span<const double>, 2> Subgame::RootChildrenCfValues()
const {
  return {
      absl::MakeConstSpan(&cf_values_[0][0], trees_[0]->root_branching_factor()),
      absl::MakeConstSpan(&cf_values_[1][0], trees_[1]->root_branching_factor())
  };
}

void Subgame::Reset() {
  // Reset trunk
  num_iterations_ = 0;
  for (int pl = 0; pl < 2; ++pl) {
    std::fill(cf_values_[pl].begin(), cf_values_[pl].end(), 0.);
    std::fill(reach_probs_[pl].begin(), reach_probs_[pl].end(), 0.);
    std::fill(beliefs_[pl].begin(), beliefs_[pl].end(), 1.);
  }
  for (algorithms::BanditVector& bandits : bandits_) {
    for (algorithms::DecisionId id : bandits.range()) {
      bandits[id]->Reset();
    }
  }
  // Reset subgames
  for (int i = 0; i < public_states_.size(); ++i) {
    PublicState& state = public_states_[i];
    for (int pl = 0; pl < 2; ++pl) {
      std::fill(state.beliefs[pl].begin(), state.beliefs[pl].end(), 0.);
      std::fill(state.values[pl].begin(), state.values[pl].end(), 0.);
    }
    if (nonterminal_evaluator_.get()) {
      std::unique_ptr<PublicStateContext>& context = contexts_[i];
      if (!state.IsTerminal() && context.get()) {
        nonterminal_evaluator_->ResetContext(context.get());
      }
    }
  }
}

bool PublicState::IsTerminal() const {
  // A quick shortcut for checking if the state is terminal: we ensure
  // this indeed holds by calling CheckConsistency() in debug mode.
  SPIEL_DCHECK_FALSE(nodes[0].empty());
  SPIEL_DCHECK_TRUE(nodes[0][0]);
  return nodes[0][0]->type() == algorithms::kTerminalInfostateNode;
}
double PublicState::ReachProbability() const {
  TreeMap<State::PlayerAction, double> reach_map;

  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < nodes[pl].size(); ++i) {
      for (int k = 0; k < nodes[pl][i]->corresponding_states_size(); ++k) {
        State* s = nodes[pl][i]->corresponding_states()[k].get();
        const std::vector<State::PlayerAction>& h = s->FullHistory();
        if (pl == 0) {
          const double chn = nodes[pl][i]->corresponding_chance_reach_probs()[k];
          reach_map[h] = chn * beliefs[pl][i];
        } else {
          reach_map[h] *= beliefs[pl][i];
        }
      }
    }
  }

  return reach_map.fold_sum(0);
}

double PublicState::Value(int player) const {
  SPIEL_CHECK_EQ(beliefs[player].size(), values[player].size());
  double acc = 0.;
  for (int i = 0; i < beliefs[player].size(); ++i) {
    acc += beliefs[player][i] * values[player][i];
  }
  return acc;
}

void DebugPrintPublicFeatures(const std::vector<PublicState>& states) {
  std::cout << "# Public features:\n";
  for (int i = 0; i < states.size(); ++i) {
    std::cout << "#   states[" << i << "].public_tensor\n#     "
              << ObservationToString(states[i].public_tensor, "\n#     ")
              << "\n";
  }
}

void CheckChildPublicStateConsistency(
    const CFRContext& cfr_public_state, const PublicState& leaf_state) {
  SPIEL_CHECK_TRUE(leaf_state.IsLeaf());
  auto trees = cfr_public_state.dlcfr->trees();
  for (int pl = 0; pl < 2; ++pl) {
    const algorithms::InfostateNode& root = trees[pl]->root();
    SPIEL_CHECK_EQ(leaf_state.nodes[pl].size(), root.num_children());
    for (int i = 0; i < root.num_children(); ++i) {
      const algorithms::InfostateNode& actual = *root.child_at(i);
      const algorithms::InfostateNode& expected = *leaf_state.nodes[pl][i];
      SPIEL_CHECK_EQ(actual.infostate_string(), expected.infostate_string());
    }
  }
  // All OK.
}

// -- CFR evaluator ------------------------------------------------------------

CFREvaluator::CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
                           std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
                           std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
                           std::shared_ptr<Observer> public_observer,
                           std::shared_ptr<Observer> infostate_observer)
    : game(std::move(game)), depth_limit(depth_limit),
      nonterminal_evaluator(std::move(leaf_evaluator)),
      terminal_evaluator(std::move(terminal_evaluator)),
      public_observer(std::move(public_observer)),
      infostate_observer(std::move(infostate_observer)) {
  SPIEL_CHECK_GT(depth_limit, 0);
}

std::unique_ptr<PublicStateContext> CFREvaluator::CreateContext(
    const PublicState& state) const {
  if (!state.IsLeaf()) return nullptr;
  SPIEL_CHECK_TRUE(state.IsLeaf());

  auto subgame_trees = std::vector{
      MakeInfostateTree(state.nodes[0],
                        depth_limit, kDlCfrInfostateTreeStorage),
      MakeInfostateTree(state.nodes[1],
                        depth_limit, kDlCfrInfostateTreeStorage)
  };
  auto subgame_bandits = MakeBanditVectors(subgame_trees, bandit_name);
  auto dlcfr = std::make_unique<Subgame>(
      game, subgame_trees, nonterminal_evaluator, terminal_evaluator,
      public_observer, std::move(subgame_bandits));
  auto cfr_public_state = std::make_unique<CFRContext>(std::move(dlcfr));
  SPIEL_DCHECK(CheckChildPublicStateConsistency(*cfr_public_state, state));
  return cfr_public_state;
}

void CFREvaluator::ResetContext(PublicStateContext* context) const {
  auto* cfr_state = open_spiel::down_cast<CFRContext*>(context);
  cfr_state->dlcfr->Reset();
}

void CFREvaluator::EvaluatePublicState(PublicState* state,
                                       PublicStateContext* context) const {
  SPIEL_CHECK_TRUE(state->IsLeaf());
  auto* cfr_state = open_spiel::down_cast<CFRContext*>(context);
  Subgame* dlcfr = cfr_state->dlcfr.get();
  if (reset_subgames_on_evaluation) {
    dlcfr->Reset();
  }
  dlcfr->SetBeliefs(state->beliefs);
  dlcfr->RunSimultaneousIterations(num_cfr_iterations);
  std::array<absl::Span<const double>, 2> resulting_values =
      dlcfr->RootChildrenCfValues();
  // Copy the results.
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(resulting_values[pl].begin(), resulting_values[pl].end(),
              state->values[pl].begin());
  }
}

double Subgame::RootValue(Player pl) const {
  const int root_branching = trees_[pl]->root_branching_factor();
  return algorithms::RootCfValue(
      root_branching, absl::MakeConstSpan(&cf_values_[pl][0], root_branching),
      beliefs_[pl]);
}


void PrintPublicStatesStats(const std::vector<PublicState>& public_leaves) {
  for (const PublicState& state : public_leaves) {
    std::array<int, 2>
        num_nodes = { (int) state.nodes[0].size(),
                      (int) state.nodes[1].size() },
        largest_infostates = {-1, -1},
        smallest_infostates = {1000000, 1000000};
    int num_states = 0;
    for (int pl = 0; pl < 2; ++pl) {
      for (const algorithms::InfostateNode* node : state.nodes[pl]) {
        int size = node->corresponding_states_size();
        if (pl == 0) num_states += size;
        largest_infostates[pl] = std::max(largest_infostates[pl], size);
        smallest_infostates[pl] = std::min(smallest_infostates[pl], size);
      }
    }
    std::cout << "# Public state #"       << state.public_id
              << (state.IsTerminal() ? " (terminal)" : "")
              << "  states: "             << num_states
              << "  infostates: "         << num_nodes
              << "  largest infostate: "  << largest_infostates
              << "  smallest infostate: " << smallest_infostates << '\n';
  }
}

bool contains(std::vector<const algorithms::InfostateNode*>& xs, const algorithms::InfostateNode* x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

// TODO: optional plumbing of observers
std::unique_ptr<PublicStatesInGame> MakeAllPublicStates(const Game& game) {
  auto all = std::make_unique<PublicStatesInGame>();
  constexpr int store_all_states = algorithms::kStoreStatesInLeaves
                                 | algorithms::kStoreStatesInRoots
                                 | algorithms::kStoreStatesInBody;
  for (int pl = 0; pl < 2; ++pl) {
    all->infostate_trees.push_back(algorithms::MakeInfostateTree(
        game, pl, 1000, store_all_states));
  }
  std::shared_ptr<Observer> public_observer = game.MakeObserver(kPublicStateObsType, {});
  Observation public_observation(game, public_observer);
  for (int pl = 0; pl < 2; ++pl) {
    const std::vector<std::vector<algorithms::InfostateNode*>>& nodes_at_depths =
        all->infostate_trees[pl]->nodes_at_depths();
    for (int depth = 0; depth < nodes_at_depths.size(); ++depth) {
      for (algorithms::InfostateNode* node : nodes_at_depths[depth]) {
        // Some nodes may not have corresponding states, even though we
        // requested to save states at all the nodes (like root, or nodes added
        // due to  rebalancing)
        if (node->corresponding_states().empty()) continue;

        const std::unique_ptr<State>& some_state =
            node->corresponding_states()[0];
        public_observation.SetFrom(*some_state, kDefaultPlayerId);
        SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
            game, public_observer, *node, public_observation.Tensor()));
        PublicState* state = all->GetPublicState(public_observation);
        if(state->move_number == -1) {
          state->move_number = some_state->MoveNumber();
        } else {
          SPIEL_CHECK_EQ(state->move_number, some_state->MoveNumber());
        }
        SPIEL_DCHECK_FALSE(contains(state->nodes[pl], node->parent()));
        state->nodes[pl].push_back(node);
      }
    }
  }
  // Init.
  MakeReachesAndValuesForPublicStates(all->public_states);

  return all;
}

PublicState* PublicStatesInGame::GetPublicState(
    const Observation& public_observation) {
  for (PublicState& state : public_states) {
    if (state.public_tensor == public_observation
        && state.state_type == kInitialPublicState) {
      return &state;
    }
  }
  // None found: create and return the pointer.
  public_states.emplace_back(public_observation,
                             kInitialPublicState,
                             public_states.size());
  return &public_states.back();
}


}  // namespace papers_with_code
}  // namespace open_spiel
