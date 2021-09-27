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
        std::unique_ptr<std::vector<Action>> h;
        if (node->type() == algorithms::kTerminalInfostateNode) {
          h = std::make_unique<std::vector<Action>>(node->TerminalHistory());
        } else {
          h = std::make_unique<std::vector<Action>>(state->History());
        }
        if (pl == 0) {
          SPIEL_CHECK_TRUE(state_histories.find(*h) == state_histories.end());
          state_histories.insert(*h);
        } else {
          SPIEL_CHECK_TRUE(state_histories.find(*h) != state_histories.end());
        }

//        if (state->IsTerminal()) num_terminals++;
//        else num_nonterminals++;
      }
    }
  }
  SPIEL_CHECK_FALSE(num_terminals > 0 && num_nonterminals > 0);
  // We must count terminals twice (2 players).
  SPIEL_CHECK_FALSE(num_terminals % 2 != 0 && num_nonterminals % 2 != 0);
  // All OK! Yay!
}

void MakeReachesAndValuesForPublicStates(std::vector<PublicState>& states) {
  for (PublicState& state : states) {
    for (int pl = 0; pl < 2; ++pl) {
      const int num_nodes = state.nodes[pl].size();
      state.beliefs[pl] = std::vector<double>(num_nodes, 1.);
      state.values[pl] = std::vector<double>(num_nodes, 0.);
      state.average_values[pl] = std::vector<double>(num_nodes, 0.);
    }
  }
}

}  // namespace

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

// -- Public state -------------------------------------------------------------

PublicState::PublicState(const Observation& public_observation,
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

bool PublicState::IsReachableByPlayer(int player) const {
  for (const double& belief : beliefs[player]) {
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
    const std::array<std::vector<double>, 2>& new_beliefs) {
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
      algorithms::InfostateNode* root_node = trees[pl]->root().child_at(i);
      SPIEL_CHECK_TRUE(root_node->is_root_child());
      PublicState* init_state = GetPublicState(
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
      algorithms::InfostateNode* leaf_node = trees[pl]->leaf_nodes()[i];
      SPIEL_CHECK_TRUE(leaf_node->is_leaf_node());
      PublicState* leaf_state = GetPublicState(public_observation,
                                               kLeafPublicState, leaf_node);
      leaf_state->nodes[pl].push_back(leaf_node);
      leaf_state->nodes_positions[leaf_node] = i;
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
      *game, public_observer, *node, public_observation.Tensor()));
  PublicState* state = GetPublicState(public_observation, state_type);
  if (state->move_number == -1) {
    state->move_number = some_state->MoveNumber();
  } else {
    SPIEL_CHECK_EQ(state->move_number, some_state->MoveNumber());
  }
  return state;
}

PublicState* Subgame::GetPublicState(const Observation& public_observation,
                                     PublicStateType state_type) {
  for (PublicState& state : public_states) {
    if (state.public_tensor == public_observation
        && state.state_type == state_type) return &state;
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

PublicState* Subgame::PickRandomLeaf(std::mt19937& rnd_gen) {
  // Pick some valid public state.
  // Loop until we find one. There should be always one -- or perhaps
  // the subgame is too deep, getting into terminal states only.
  int num_states = public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);
  int pick_public_state;
  PublicState* state = nullptr;
  while (!state || state->IsTerminal() || state->IsInitial()) {
    pick_public_state = public_state_dist(rnd_gen);
    state = &public_states[pick_public_state];
  }
  return state;
}

std::unique_ptr<Subgame> MakeSubgame(const PublicState& state,
                                     std::shared_ptr<const Game> game,
                                     std::shared_ptr<Observer> public_observer,
                                     int custom_move_ahead_limit) {
  if (!game) {
    SPIEL_CHECK_FALSE(state.nodes[0].empty());
    const algorithms::InfostateNode* node = state.nodes[0][0];
    SPIEL_CHECK_TRUE(node);
    SPIEL_CHECK_FALSE(node->corresponding_states().empty());
    const State* a_state = node->corresponding_states()[0].get();
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

}  // namespace papers_with_code
}  // namespace open_spiel
