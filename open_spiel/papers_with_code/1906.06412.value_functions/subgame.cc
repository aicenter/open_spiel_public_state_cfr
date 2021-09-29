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

void CheckConsistency(const PublicState &s) {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // They should all be terminal or non-terminal.
  // The set of corresponding states must be the same across players.
  int num_terminals = 0, num_nonterminals = 0;

  using History = std::vector<Action>;
  std::unordered_set<History, absl::Hash<History>> state_histories;
  for (int pl = 0; pl < 2; ++pl) {
    for (const algorithms::InfostateNode *node : s.nodes[pl]) {
      SPIEL_CHECK_TRUE(!s.IsLeaf() || node->is_leaf_node());
      SPIEL_CHECK_EQ(node->tree().acting_player(), pl);
      if (node->type() == algorithms::kTerminalInfostateNode) num_terminals++;
      else num_nonterminals++;

//      for (const std::unique_ptr<State> &state : node->corresponding_states()) {
//        std::unique_ptr<std::vector<Action>> h;
//        if (node->type() == algorithms::kTerminalInfostateNode) {
//          h = std::make_unique<std::vector<Action>>(node->TerminalHistory());
//        } else {
//          h = std::make_unique<std::vector<Action>>(state->History());
//        }
//        if (pl == 0) {
//          SPIEL_CHECK_TRUE(state_histories.find(*h) == state_histories.end());
//          state_histories.insert(*h);
//        } else {
//          SPIEL_CHECK_TRUE(state_histories.find(*h) != state_histories.end());
//        }
//
////        if (state->IsTerminal()) num_terminals++;
////        else num_nonterminals++;
//      }
    }
  }
  SPIEL_CHECK_FALSE(num_terminals > 0 && num_nonterminals > 0);
  // We must count terminals twice (2 players).
  SPIEL_CHECK_FALSE(num_terminals % 2 != 0 && num_nonterminals % 2 != 0);
  // All OK! Yay!
}

bool DoStatesProduceEqualPublicObservations(
    const Game &game, std::shared_ptr<Observer> public_observer,
    const algorithms::InfostateNode &node, absl::Span<float> expected_observation) {
  Observation public_observation(game, public_observer);

  // Check that indeed all states produce the same public observations.
  for (const std::unique_ptr<State> &state : node.corresponding_states()) {
    public_observation.SetFrom(*state, kDefaultPlayerId);
    if (public_observation.Tensor() != expected_observation) return false;
  }
  return true;
}

void CheckChildPublicStateConsistency(
    const CFRContext &cfr_public_state, const PublicState &leaf_state) {
  SPIEL_CHECK_TRUE(leaf_state.IsLeaf());
  auto trees = cfr_public_state.dlcfr->subgame()->trees;
  for (int pl = 0; pl < 2; ++pl) {
    const algorithms::InfostateNode &root = trees[pl]->root();
    SPIEL_CHECK_EQ(leaf_state.nodes[pl].size(), root.num_children());
    for (int i = 0; i < root.num_children(); ++i) {
      const algorithms::InfostateNode &actual = *root.child_at(i);
      const algorithms::InfostateNode &expected = *leaf_state.nodes[pl][i];
      SPIEL_CHECK_EQ(actual.infostate_string(), expected.infostate_string());
    }
  }
  // All OK.
}

void MakeReachesAndValuesForPublicStates(std::vector<PublicState> &states) {
  for (PublicState &state : states) {
    for (int pl = 0; pl < 2; ++pl) {
      const int num_nodes = state.nodes[pl].size();
      state.beliefs[pl] = std::vector<double>(num_nodes, 1.);
      state.values[pl] = std::vector<double>(num_nodes, 0.);
      state.average_values[pl] = std::vector<double>(num_nodes, 0.);
    }
  }
}

std::vector<std::unique_ptr<PublicStateContext>> MakeContexts(
    std::shared_ptr<Subgame> subgame,
    const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator) {
  std::vector<std::unique_ptr<PublicStateContext>> contexts;
  contexts.reserve(subgame->public_states.size());
  for (const PublicState &state : subgame->public_states) {
    SPIEL_DCHECK(CheckConsistency(state));
    if (state.IsTerminal()) {
      contexts.push_back(terminal_evaluator->CreateContext(state));
    } else {
      contexts.push_back(nonterminal_evaluator
                         ? nonterminal_evaluator->CreateContext(state)
                         : nullptr);
    }
  }
  return contexts;
}

}  // namespace


// -- Public state -------------------------------------------------------------

PublicState::PublicState(const Observation &public_observation,
                         const PublicStateType state_type,
                         const size_t public_id)
    : public_tensor(public_observation), state_type(state_type),
      public_id(public_id) {}

bool PublicState::IsTerminal() const {
  // A quick shortcut for checking if the state is terminal: we ensure
  // this indeed holds by calling CheckConsistency() in debug mode.
  // TODO: find which player has non empty nodes and call type
  SPIEL_DCHECK_FALSE(nodes[0].empty());
  SPIEL_DCHECK_TRUE(nodes[0][0]);
  return nodes[0][0]->type() == algorithms::kTerminalInfostateNode;
}

double PublicState::ReachProbability() const {
  TreeMap<State::PlayerAction, double> reach_map;

  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < nodes[pl].size(); ++i) {
      for (int k = 0; k < nodes[pl][i]->corresponding_states_size(); ++k) {
        State *s = nodes[pl][i]->corresponding_states()[k].get();
        const std::vector<State::PlayerAction> &h = s->FullHistory();
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

bool PublicState::IsReachableByPlayer(int player) const {
  for (const double &belief : beliefs[player]) {
    if (belief > 0) return true;
  }
  return false;
}

double PublicState::CurrentValue(int player) const {
  SPIEL_CHECK_EQ(beliefs[player].size(), values[player].size());
  double acc = 0.;
  for (int i = 0; i < beliefs[player].size(); ++i) {
    acc += beliefs[player][i] * values[player][i];
  }
  return acc;
}

double PublicState::AverageValue(int player) const {
  SPIEL_CHECK_EQ(beliefs[player].size(), average_values[player].size());
  double acc = 0.;
  for (int i = 0; i < beliefs[player].size(); ++i) {
    acc += beliefs[player][i] * average_values[player][i];
  }
  return acc;
}

void PublicState::SetBeliefs(
    const std::array<std::vector<double>, 2> &new_beliefs) {
  SPIEL_CHECK_EQ(new_beliefs[0].size(), nodes[0].size());
  SPIEL_CHECK_EQ(new_beliefs[1].size(), nodes[1].size());
  beliefs = new_beliefs;
}

std::unordered_map<std::string, double> PublicState::InfostateAvgValues(
    Player player) const {
  std::unordered_map<std::string, double> CFVs;
  for (int j = 0; j < nodes[player].size(); j++) {
    std::string infostate_string = nodes[player][j]->infostate_string();
    double cfv = average_values[player][j];
    CFVs.emplace(infostate_string, cfv);
  }
  return CFVs;
}

// -- Subgame ------------------------------------------------------------------

Subgame::Subgame(
    std::shared_ptr<const Game> game,
    std::shared_ptr<Observer> a_public_observer,
    std::vector<std::shared_ptr<algorithms::InfostateTree>> depth_lim_trees) :
    game(std::move(game)),
    public_observer(std::move(a_public_observer)),
    trees(std::move(depth_lim_trees)) {
  SPIEL_CHECK_TRUE(public_observer->HasTensor());
  SPIEL_CHECK_EQ(trees[0]->storage_policy() & kDlCfrInfostateTreeStorage,
                 kDlCfrInfostateTreeStorage);
  SPIEL_CHECK_EQ(trees[1]->storage_policy() & kDlCfrInfostateTreeStorage,
                 kDlCfrInfostateTreeStorage);
  MakePublicStates();
  MakeBeliefsAndValues();
}

Subgame::Subgame(std::shared_ptr<const Game> game, int max_moves)
    : Subgame(game, game->MakeObserver(kPublicStateObsType, {}),
              algorithms::MakeInfostateTrees(*game, max_moves,
                                             kDlCfrInfostateTreeStorage)) {}

void Subgame::MakePublicStates() {
  Observation public_observation(*game, public_observer);
  // Save nodes for initial (root) public state.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < trees[pl]->root_branching_factor(); ++i) {
      algorithms::InfostateNode *root_node = trees[pl]->root().child_at(i);
      SPIEL_CHECK_TRUE(root_node->is_root_child());
      PublicState *init_state = GetPublicState(
          public_observation, kInitialPublicState, root_node);
      init_state->nodes[pl].push_back(root_node);
      init_state->nodes_positions[root_node] = i;
    }
  }

  // Make sure we have built only one initial state:
  // the infostate trees are rooted in a single initial state.
  // While more initial states are possible, we don't do this as it would
  // complicate the code unnecessarily.
  SPIEL_CHECK_FALSE(public_states.empty());
  SPIEL_CHECK_EQ(public_states.size(), 1);

  // Save node positions for leaf public states.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < trees[pl]->num_leaves(); ++i) {
      algorithms::InfostateNode *leaf_node = trees[pl]->leaf_nodes()[i];
      SPIEL_CHECK_TRUE(leaf_node->is_leaf_node());
      PublicState *leaf_state = GetPublicState(public_observation,
                                               kLeafPublicState, leaf_node);
      leaf_state->nodes[pl].push_back(leaf_node);
      leaf_state->nodes_positions[leaf_node] = i;
    }
  }
}

PublicState *Subgame::GetPublicState(Observation &public_observation,
                                     PublicStateType state_type,
                                     algorithms::InfostateNode *node) {
  SPIEL_CHECK_FALSE(node->corresponding_states().empty());
  const std::unique_ptr<State> &some_state = node->corresponding_states()[0];
  public_observation.SetFrom(*some_state, kDefaultPlayerId);
  SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
      *game, public_observer, *node, public_observation.Tensor()));
  PublicState *state = GetPublicState(public_observation, state_type);
  if (state->move_number == -1) {
    state->move_number = some_state->MoveNumber();
  } else {
//    SPIEL_CHECK_EQ(state->move_number, some_state->MoveNumber());
  }
  return state;
}

PublicState *Subgame::GetPublicState(const Observation &public_observation,
                                     PublicStateType state_type) {
  for (PublicState &state : public_states) {
    if (state.public_tensor == public_observation
        && state.state_type == state_type)
      return &state;
  }
  // None found: create and return the pointer.
  public_states.emplace_back(public_observation,
                             state_type, public_states.size());
  public_states.back().trees = trees;
  return &public_states.back();
}

void Subgame::MakeBeliefsAndValues() {
  MakeReachesAndValuesForPublicStates(public_states);
}

PublicState *Subgame::PickRandomLeaf(std::mt19937 &rnd_gen) {
  // Pick some valid public state.
  // Loop until we find one. There should be always one -- or perhaps
  // the subgame is too deep, getting into terminal states only.
  int num_states = public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);
  int pick_public_state;
  PublicState *state = nullptr;
  while (!state || state->IsTerminal() || state->IsInitial()) {
    pick_public_state = public_state_dist(rnd_gen);
    state = &public_states[pick_public_state];
  }
  return state;
}

std::unique_ptr<Subgame> MakeSubgame(const PublicState &state,
                                     std::shared_ptr<const Game> game,
                                     std::shared_ptr<Observer> public_observer,
                                     int custom_move_ahead_limit) {
  if (!game) {
    SPIEL_CHECK_FALSE(state.nodes[0].empty());
    const algorithms::InfostateNode *node = state.nodes[0][0];
    SPIEL_CHECK_TRUE(node);
    SPIEL_CHECK_FALSE(node->corresponding_states().empty());
    const State *a_state = node->corresponding_states()[0].get();
    SPIEL_CHECK_TRUE(a_state);
    game = a_state->GetGame();
  }
  if (!public_observer) {
    public_observer = game->MakeObserver(kPublicStateObsType, {});
  }

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(state.nodes[pl],
                                      custom_move_ahead_limit, kDlCfrInfostateTreeStorage
    ));
  }
  auto out = std::make_unique<Subgame>(game, public_observer, trees);
  out->initial_state().SetBeliefs(state.beliefs);
  return out;
}

std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator(algorithms::PokerData poker_data, std::vector<int> cards) {
  return std::make_shared<PokerTerminalEvaluator>(std::move(poker_data), std::move(cards));
}

std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator() {
  return std::make_shared<TerminalEvaluator>();
}

std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator() {
  return std::make_shared<DummyEvaluator>();
}

std::shared_ptr<PublicStateEvaluator> MakeApproxOracleEvaluator(
    std::shared_ptr<const Game> game, int cfr_iterations) {

  std::shared_ptr<const PublicStateEvaluator> dummy_evaluator =
      MakeDummyEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();

  return std::make_shared<CFREvaluator>(game, algorithms::kNoMoveAheadLimit,
                                        dummy_evaluator, terminal_evaluator,
                                        public_observer, infostate_observer,
                                        cfr_iterations);
}

PokerTerminalPublicStateContext::PokerTerminalPublicStateContext(
    const PublicState &state) {
  SPIEL_CHECK_TRUE(state.IsTerminal());
  auto &leaf_nodes = state.nodes;
  SPIEL_CHECK_EQ(leaf_nodes[0].size(), leaf_nodes[1].size());

  fold_state_ = (state.nodes[0][0]->corresponding_states()[0]->History().back() == universal_poker::kFold);

  const int num_terminals = leaf_nodes[0].size();
  utilities_.reserve(num_terminals);

  for (int i = 0; i < num_terminals; ++i) {
    const algorithms::InfostateNode *leaf = leaf_nodes[0][i];
    const double v = leaf->terminal_utility();
    const double chn = leaf->terminal_chance_reach_prob();
    utilities_.push_back(v * chn);
  }
}

int PokerTerminalEvaluator::ConvertToFullPokerCard(int card, const algorithms::PokerData &poker_data) {
  int rank = (int) (card / poker_data.num_suits_);
  int suit = card % poker_data.num_suits_;
  return rank * 4 + suit;
}

PokerTerminalEvaluator::PokerTerminalEvaluator(
    algorithms::PokerData poker_data, std::vector<int> board_cards) : poker_data_(std::move(poker_data)) {
  board_cards.insert(board_cards.begin(), 1);
  board_cards.insert(board_cards.begin(), 0);
  std::vector<int> hand_strength;
  hand_strength.reserve(poker_data_.num_hands_);
  for(int i = 0; i < board_cards.size();i++) {
    board_cards[i] = ConvertToFullPokerCard(board_cards[i], poker_data_);
  }
  for (int card_one = 0; card_one < poker_data_.num_cards_ - 1; card_one++) {
    board_cards[0] = ConvertToFullPokerCard(card_one, poker_data_);;
    for (int card_two = card_one + 1; card_two < poker_data_.num_cards_; card_two++) {
      board_cards[1] = ConvertToFullPokerCard(card_two, poker_data_);;
      universal_poker::logic::CardSet cards = universal_poker::logic::CardSet(board_cards);
      hand_strength.push_back(cards.RankCards());
    }
  }
  std::vector<size_t> idx(hand_strength.size());
  iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&hand_strength](size_t i1, size_t i2) { return hand_strength[i1] < hand_strength[i2]; });
  std::vector<std::vector<int>> sorted_with_ties;
  int current_hand_strength = -1;
  for (int hand_index : idx) {
    if (hand_strength[hand_index] > current_hand_strength) {
      sorted_with_ties.push_back({hand_index});
      current_hand_strength = hand_strength[hand_index];
    } else {
      sorted_with_ties.back().push_back(hand_index);
    }
  }
  ordered_hands_ = sorted_with_ties;
}

std::unique_ptr<PublicStateContext> PokerTerminalEvaluator::CreateContext(
    const PublicState &state) const {
  return std::make_unique<PokerTerminalPublicStateContext>(state);
}

void PokerTerminalEvaluator::EvaluatePublicState(
    PublicState *state, PublicStateContext *context) const {
  auto *terminal_context = open_spiel::down_cast<PokerTerminalPublicStateContext *>(context);
  if (terminal_context->fold_state_) {
    // Fold state
    std::vector<double> beliefs(2, 0);
    for (int hand_index = 0; hand_index < state->beliefs[0].size(); hand_index++) {
      beliefs[0] += state->beliefs[0][hand_index];
      beliefs[1] += state->beliefs[1][hand_index];
    }
    for (int hand_index = 0; hand_index < state->beliefs[0].size(); hand_index++) {
      state->values[0][hand_index] = terminal_context->utilities_[hand_index] * beliefs[1];
      state->values[1][hand_index] = -terminal_context->utilities_[hand_index] * beliefs[0];
    }
  } else {
    // Showdown state
    std::vector<double> beliefs(2, 0);
    std::vector<double> current_beliefs(2, 0);
    for (const std::vector<int> &hand_indexes : ordered_hands_) {
      // collect beliefs
      current_beliefs[0] = current_beliefs[1] = 0;
      for (int hand_index : hand_indexes) {
        current_beliefs[0] += state->beliefs[0][hand_index];
        current_beliefs[1] += state->beliefs[1][hand_index];
      }
      // compute value
      for (int hand_index : hand_indexes) {
        state->values[0][hand_index] =
            terminal_context->utilities_[hand_index] * (2 * beliefs[1] + current_beliefs[1] - 1);
        state->values[1][hand_index] =
            terminal_context->utilities_[hand_index] * (2 * beliefs[0] + current_beliefs[0] - 1);
      }
      beliefs[0] += current_beliefs[0];
      beliefs[1] += current_beliefs[1];
    }
  }
}

TerminalPublicStateContext::TerminalPublicStateContext(
    const PublicState &state) {
  SPIEL_CHECK_TRUE(state.IsTerminal());
  auto &leaf_nodes = state.nodes;
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
    const algorithms::InfostateNode *a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const algorithms::InfostateNode *b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const algorithms::InfostateNode *leaf = leaf_nodes[0][i];
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
    const PublicState &state) const {
  return std::make_unique<TerminalPublicStateContext>(state);
}

void TerminalEvaluator::EvaluatePublicState(
    PublicState *state, PublicStateContext *context) const {
  auto *terminal = open_spiel::down_cast<TerminalPublicStateContext *>(context);
  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    state->values[0][i] = terminal->utilities[i] * state->beliefs[1][j];
    state->values[1][j] = -terminal->utilities[i] * state->beliefs[0][i];
  }
}

SubgameSolver::SubgameSolver(
    std::shared_ptr<Subgame> subgame,
    const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
    const std::shared_ptr<std::mt19937> rnd_gen,
    const std::string &bandit_name,
    PolicySelection save_values_policy,
    bool safe_resolving,
    bool beliefs_for_average,
    double noisy_values
) : subgame_(subgame),
    nonterminal_evaluator_(nonterminal_evaluator),
    terminal_evaluator_(terminal_evaluator),
    rnd_gen_(rnd_gen),
    safe_resolving_(safe_resolving),
    beliefs_for_average_(beliefs_for_average),
    noisy_values_(noisy_values),
    bandits_(algorithms::MakeBanditVectors(subgame_->trees, bandit_name)),
    reach_probs_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                  std::vector<double>(subgame_->trees[1]->num_leaves())}),
    cf_values_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                std::vector<double>(subgame_->trees[1]->num_leaves())}),
    contexts_(MakeContexts(subgame, nonterminal_evaluator, terminal_evaluator)),
    num_iterations_(0),
    init_save_values_(save_values_policy) {}

std::shared_ptr<Policy> SubgameSolver::AveragePolicy() {
  return std::make_shared<algorithms::BanditsAveragePolicy>(subgame()->trees,
                                                            bandits_);
}

std::shared_ptr<Policy> SubgameSolver::CurrentPolicy() {
  return std::make_shared<algorithms::BanditsCurrentPolicy>(subgame()->trees,
                                                            bandits_);
}

void SubgameSolver::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;

    // 1. Prepare initial reach probs, based on beliefs in initial state.
    std::array<std::vector<double>, 2> &beliefs = initial_state().beliefs;
//    SPIEL_DCHECK_TRUE(initial_state().IsReachableBySomePlayer());
    for (int pl = 0; pl < 2; ++pl) {
      std::copy(beliefs[pl].begin(), beliefs[pl].end(),
                reach_probs_[pl].begin());
    }

    // 2. Compute reach probs to the terminals.
    for (int pl = 0; pl < 2; ++pl) {
      TopDownCurrent(*subgame_->trees[pl], bandits_[pl],
                     absl::MakeSpan(reach_probs_[pl]), num_iterations_);
    }

    // Optionally instead of current reach probs, use average reach probs.
    // This corresponds to using CFR-AVE in [Appendix E, 1].
    //
    // [1] Combining Deep Reinforcement Learning and Search
    //     for Imperfect-Information Games
    //     Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong
    if (beliefs_for_average_) {
      // 1. Prepare initial reach probs, based on beliefs in initial state.
      for (int pl = 0; pl < 2; ++pl) {
        std::copy(beliefs[pl].begin(), beliefs[pl].end(),
                  reach_probs_[pl].begin());
      }
      // 2. Compute reach probs of avg strategy to the terminals.
      for (int pl = 0; pl < 2; ++pl) {
        TopDownAverage(*subgame_->trees[pl], bandits_[pl],
                       absl::MakeSpan(reach_probs_[pl]));
      }
    }

    // 3. Evaluate leaves using current reach probs.
    EvaluateLeaves();

    // 4. Propagate updated values up the tree.
    for (int pl = 0; pl < 2; ++pl) {
      BottomUp(*subgame_->trees[pl], bandits_[pl],
               absl::MakeSpan(cf_values_[pl]));
    }
    // Holds for oracle values, but not for the ones coming from NN (not yet).
//    SPIEL_DCHECK_FLOAT_NEAR(initial_state().Value(0),
//                            -initial_state().Value(1), 1e-6);

    if (init_save_values_ == PolicySelection::kAveragePolicy) {
      IncrementallyAverageValuesInState(&initial_state());
    }
  }

  if (init_save_values_ == PolicySelection::kCurrentPolicy) {
    CopyCurrentValuesToInitialState();
  }
}

void SubgameSolver::EvaluateLeaves() {
  SPIEL_CHECK_EQ(subgame()->public_states.size(), contexts_.size());
  for (int i = 0; i < subgame()->public_states.size(); ++i) {
    PublicState *state = &subgame()->public_states[i];
    if (!state->IsLeaf()) continue;
    PublicStateContext *context = contexts_[i].get();
    EvaluateLeaf(state, context);
  }
}

void SubgameSolver::EvaluateLeaf(PublicState *state,
                                 PublicStateContext *context) {
  SPIEL_CHECK_TRUE(state);
  SPIEL_CHECK_TRUE(state->IsLeaf());

  // 1. Prepare beliefs
  for (int pl = 0; pl < 2; pl++) {
    const int num_leaves = state->nodes[pl].size();
    for (int j = 0; j < num_leaves; ++j) {
      const algorithms::InfostateNode *leaf_node = state->nodes[pl][j];
      const int trunk_position = state->nodes_positions.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, subgame()->trees[pl]->num_leaves());
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
      const algorithms::InfostateNode *leaf_node = state->nodes[pl][j];
      const int trunk_position = state->nodes_positions.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, subgame()->trees[pl]->num_leaves());
      // Copy value from the leaf public state to the trunk.
      double noise = 0.;
      if (noisy_values_ > 0) {
        SPIEL_CHECK_TRUE(rnd_gen_);
        std::normal_distribution<double> dist(0, noisy_values_);
        noise = dist(*rnd_gen_);
      }
      cf_values_[pl][trunk_position] = state->values[pl][j] + noise;
    }
  }

  // 4. Incrementally update average CFVs.
  if (safe_resolving_) {
    for (int pl = 0; pl < 2; pl++) {
      const int num_leaves = state->nodes[pl].size();
      for (int j = 0; j < num_leaves; ++j) {
        state->average_values[pl][j] +=
            (state->values[pl][j] - state->average_values[pl][j])
                / num_iterations_;
      }
    }
  }
}

void SubgameSolver::Reset() {
  // Reset trunk
  num_iterations_ = 0;
  for (int pl = 0; pl < 2; ++pl) {
    std::fill(cf_values_[pl].begin(), cf_values_[pl].end(), 0.);
    std::fill(reach_probs_[pl].begin(), reach_probs_[pl].end(), 0.);
  }
  for (algorithms::BanditVector &bandits : bandits_) {
    for (algorithms::DecisionId id : bandits.range()) {
      bandits[id]->Reset();
    }
  }
  // Reset subgames
  for (int i = 0; i < subgame_->public_states.size(); ++i) {
    PublicState &state = subgame_->public_states[i];
    for (int pl = 0; pl < 2; ++pl) {
      // Conserve beliefs for initial state.
      if (!state.IsInitial()) {
        std::fill(state.beliefs[pl].begin(), state.beliefs[pl].end(), 0.);
      }
      std::fill(state.values[pl].begin(), state.values[pl].end(), 0.);
      std::fill(state.average_values[pl].begin(),
                state.average_values[pl].end(), 0.);
    }
    if (nonterminal_evaluator_.get()) {
      std::unique_ptr<PublicStateContext> &context = contexts_[i];
      if (!state.IsTerminal() && context.get()) {
        nonterminal_evaluator_->ResetContext(context.get());
      }
    }
  }
}

void SubgameSolver::CopyCurrentValuesToInitialState() {
  for (int pl = 0; pl < 2; ++pl) {
    int branching = subgame()->trees[pl]->root_branching_factor();
    SPIEL_CHECK_EQ(initial_state().values[pl].size(), branching);
    std::copy(cf_values_[pl].begin(), cf_values_[pl].begin() + branching,
              initial_state().values[pl].begin());
  }
}

void SubgameSolver::IncrementallyAverageValuesInState(PublicState *state) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < state->values[pl].size(); ++i) {
      state->values[pl][i] +=
          (cf_values_[pl][i] - state->values[pl][i]) / num_iterations_;
    }
  }
}


// -- CFR evaluator ------------------------------------------------------------

CFREvaluator::CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
                           std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
                           std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
                           std::shared_ptr<Observer> public_observer,
                           std::shared_ptr<Observer> infostate_observer,
                           int cfr_iterations)
    : game(std::move(game)), depth_limit(depth_limit),
      nonterminal_evaluator(std::move(leaf_evaluator)),
      terminal_evaluator(std::move(terminal_evaluator)),
      public_observer(std::move(public_observer)),
      infostate_observer(std::move(infostate_observer)),
      num_cfr_iterations(cfr_iterations) {
  SPIEL_CHECK_GT(depth_limit, 0);
}

std::unique_ptr<PublicStateContext> CFREvaluator::CreateContext(
    const PublicState &state) const {
  if (!state.IsLeaf()) return nullptr;
  SPIEL_CHECK_TRUE(state.IsLeaf());

  auto subgame_trees = std::vector{
      MakeInfostateTree(state.nodes[0],
                        depth_limit, kDlCfrInfostateTreeStorage),
      MakeInfostateTree(state.nodes[1],
                        depth_limit, kDlCfrInfostateTreeStorage)
  };
  auto subgame = std::make_shared<Subgame>(
      game, public_observer, subgame_trees);
  auto solver = std::make_unique<SubgameSolver>(
      subgame, nonterminal_evaluator, terminal_evaluator, /*rnd_gen=*/nullptr,
      bandit_name, save_values_policy);
  auto cfr_public_state = std::make_unique<CFRContext>(std::move(solver));
  SPIEL_DCHECK(CheckChildPublicStateConsistency(*cfr_public_state, state));
  return cfr_public_state;
}

void CFREvaluator::ResetContext(PublicStateContext *context) const {
  auto *cfr_state = open_spiel::down_cast<CFRContext *>(context);
  cfr_state->dlcfr->Reset();
}

void CFREvaluator::EvaluatePublicState(PublicState *state,
                                       PublicStateContext *context) const {
  SPIEL_CHECK_TRUE(state->IsLeaf());
  auto *cfr_context = open_spiel::down_cast<CFRContext *>(context);
  SubgameSolver *solver = cfr_context->dlcfr.get();
  // We pretty much always should. This only to support special test cases.
  if (reset_subgames_on_evaluation) {
    solver->Reset();
  }
  solver->initial_state().SetBeliefs(state->beliefs);
  solver->RunSimultaneousIterations(num_cfr_iterations);
  auto &resulting_values = solver->initial_state().values;
  // Copy the results.
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(resulting_values[pl].begin(), resulting_values[pl].end(),
              state->values[pl].begin());
  }
//  for (const algorithms::InfostateNode* node : state->nodes[0]) {
//    std::cout << node->infostate_string() << "\n";
//  }
//  std::cout << state->public_id << " "
//            << " " << " beliefs: " << state->beliefs[0]
//            << " " << " values: " << state->values[0]  << "\n";
}

void PrintPublicStatesStats(const std::vector<PublicState> &public_leaves) {
  for (const PublicState &state : public_leaves) {
    std::array<int, 2>
        num_nodes = {(int) state.nodes[0].size(),
                     (int) state.nodes[1].size()},
        largest_infostates = {-1, -1},
        smallest_infostates = {1000000, 1000000};
    int num_states = 0;
    for (int pl = 0; pl < 2; ++pl) {
      for (const algorithms::InfostateNode *node : state.nodes[pl]) {
        int size = node->corresponding_states_size();
        if (pl == 0) num_states += size;
        largest_infostates[pl] = std::max(largest_infostates[pl], size);
        smallest_infostates[pl] = std::min(smallest_infostates[pl], size);
      }
    }
    std::cout << "# Public state #" << state.public_id
              << (state.IsTerminal() ? " (terminal)" : "")
              << "  states: " << num_states
              << "  infostates: " << num_nodes
              << "  largest infostate: " << largest_infostates
              << "  smallest infostate: " << smallest_infostates << '\n';
  }
}

bool contains(std::vector<const algorithms::InfostateNode *> &xs,
              const algorithms::InfostateNode *x) {
  return std::find(xs.begin(), xs.end(), x) != xs.end();
}

// TODO: optional plumbing of observers
std::unique_ptr<PublicStatesInGame> MakeAllPublicStates(const Game &game) {
  auto all = std::make_unique<PublicStatesInGame>();
  constexpr int store_all_states = algorithms::kStoreStatesInLeaves
      | algorithms::kStoreStatesInRoots
      | algorithms::kStoreStatesInBody;
  for (int pl = 0; pl < 2; ++pl) {
    all->infostate_trees.push_back(algorithms::MakeInfostateTree(
        game, pl, algorithms::kNoMoveAheadLimit, store_all_states));
  }
  std::shared_ptr<Observer> public_observer =
      game.MakeObserver(kPublicStateObsType, {});
  Observation public_observation(game, public_observer);
  for (int pl = 0; pl < 2; ++pl) {
    const std::vector<std::vector<algorithms::InfostateNode *>> &nodes_at_depths =
        all->infostate_trees[pl]->nodes_at_depths();
    for (int depth = 0; depth < nodes_at_depths.size(); ++depth) {
      for (algorithms::InfostateNode *node : nodes_at_depths[depth]) {
        // Some nodes may not have corresponding states, even though we
        // requested to save states at all the nodes (like root, or nodes added
        // due to  rebalancing)
        if (node->corresponding_states().empty()) continue;

        const std::unique_ptr<State> &some_state =
            node->corresponding_states()[0];
        public_observation.SetFrom(*some_state, kDefaultPlayerId);
        SPIEL_DCHECK_TRUE(DoStatesProduceEqualPublicObservations(
            game, public_observer, *node, public_observation.Tensor()));
        PublicState *state = all->GetPublicState(public_observation);
        if (state->move_number == -1) {
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

PublicState *PublicStatesInGame::GetPublicState(
    const Observation &public_observation) {
  for (PublicState &state : public_states) {
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

PolicySelection GetSaveValuesPolicy(const std::string &s) {
  if (s == "current") return PolicySelection::kCurrentPolicy;
  if (s == "average") return PolicySelection::kAveragePolicy;
  SpielFatalError("Exhausted pattern match for PolicySelection");
}

}  // namespace papers_with_code
}  // namespace open_spiel
