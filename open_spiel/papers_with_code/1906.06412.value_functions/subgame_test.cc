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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

#include <cmath>
#include <iostream>
#include <utility>
#include <absl/strings/str_replace.h>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/cfr.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void CheckInfostatePolicy(
    const std::string &infostate, const Policy &a, const Policy &b) {
  ActionsAndProbs vec_policy = a.GetStatePolicy(infostate);
  ActionsAndProbs str_policy = b.GetStatePolicy(infostate);
  SPIEL_CHECK_EQ(vec_policy.size(), str_policy.size());
  for (int j = 0; j < vec_policy.size(); ++j) {
    SPIEL_CHECK_EQ(vec_policy[j].first, str_policy[j].first);
    SPIEL_CHECK_FLOAT_NEAR(vec_policy[j].second, str_policy[j].second, 1e-6);
  }
}

void CheckIterationConsistency(const Policy &a, const Policy &b,
                               const algorithms::InfostateTree &tree) {
  for (algorithms::DecisionId id : tree.AllDecisionIds()) {
    CheckInfostatePolicy(tree.decision_infostate(id)->infostate_string(), a, b);
  }
}

void TestTerminalEvaluatorHasSameIterations(const std::string &game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int cfr_iterations = 10;

  algorithms::InfostateCFR vec_solver(*game);

  // We use only the terminal evaluator.
  std::shared_ptr<PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
      MakeDummyEvaluator();
  auto subgame = std::make_shared<Subgame>(game, algorithms::kNoMoveAheadLimit);
  SubgameSolver dl_solver(subgame, nonterminal_evaluator, terminal_evaluator,
      /*rnd_gen=*/nullptr, "RegretMatching");

  std::shared_ptr<Policy> vec_avg = vec_solver.AveragePolicy();
  std::shared_ptr<Policy> dl_avg = dl_solver.AveragePolicy();
  std::shared_ptr<Policy> vec_cur = vec_solver.CurrentPolicy();
  std::shared_ptr<Policy> dl_cur = dl_solver.CurrentPolicy();

  for (int i = 0; i < cfr_iterations; ++i) {
    vec_solver.RunSimultaneousIterations(1);
    dl_solver.RunSimultaneousIterations(1);
    for (int pl = 0; pl < 2; ++pl) {
      CheckIterationConsistency(*vec_avg, *dl_avg, *vec_solver.trees()[pl]);
      CheckIterationConsistency(*vec_cur, *dl_cur, *vec_solver.trees()[pl]);
    }
  }
}

std::unique_ptr<SubgameSolver> MakeRecursiveDepthLimitedCFR(
    std::shared_ptr<const Game> game, int trunk_depth_limit,
    int subgame_depth_limit) {
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
      MakeDummyEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  // Recursive leaf evaluator.

  auto leaf_evaluator = std::make_shared<CFREvaluator>(
      game, subgame_depth_limit, nonterminal_evaluator,
      terminal_evaluator, public_observer, infostate_observer);
  leaf_evaluator->reset_subgames_on_evaluation = false;  // Needed for test !
  leaf_evaluator->bandit_name = "RegretMatching";
  leaf_evaluator->nonterminal_evaluator = leaf_evaluator;
  leaf_evaluator->num_cfr_iterations = 1;
  leaf_evaluator->save_values_policy = PolicySelection::kCurrentPolicy;

  // Builds the root leaf public states so that we can call the recursive
  // evaluator.
  auto subgame = std::make_shared<Subgame>(game, trunk_depth_limit);
  return std::make_unique<SubgameSolver>(subgame, leaf_evaluator,
                                         terminal_evaluator,
      /*rnd_gen=*/nullptr,
                                         "RegretMatching",
                                         PolicySelection::kCurrentPolicy);
}

void TestRecursiveDepthLimitedSolving(const std::string &game_name) {
  // If we make 1 iterations in each of the recursive subgames, it is the same
  // as if we were running CFR in the whole game. Thus we can check that we
  // compute the same regrets as the original implementation.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int trunk_iterations = 10;

  for (int trunk_depth_limit = 0; trunk_depth_limit < game->MaxMoveNumber();
       ++trunk_depth_limit) {
    for (int subgame_depth_limit = 1; subgame_depth_limit < 4;
         ++subgame_depth_limit) {

      algorithms::InfostateCFR vec_solver(*game);
      std::unique_ptr<SubgameSolver> dl_solver = MakeRecursiveDepthLimitedCFR(
          game, trunk_depth_limit, subgame_depth_limit);

      std::shared_ptr<Policy> vec_avg = vec_solver.AveragePolicy();
      std::shared_ptr<Policy> dl_avg = dl_solver->AveragePolicy();
      std::shared_ptr<Policy> vec_cur = vec_solver.CurrentPolicy();
      std::shared_ptr<Policy> dl_cur = dl_solver->CurrentPolicy();
      auto trees = dl_solver->subgame()->trees;

      for (int j = 0; j < trunk_iterations; ++j) {
        vec_solver.RunSimultaneousIterations(1);
        dl_solver->RunSimultaneousIterations(1);
        SPIEL_CHECK_FLOAT_NEAR(vec_solver.RootValue(),
                               dl_solver->initial_state().CurrentValue(), 1e-6);
        for (int pl = 0; pl < 2; ++pl) {
          CheckIterationConsistency(*vec_avg, *dl_avg, *trees[pl]);
          CheckIterationConsistency(*vec_cur, *dl_cur, *trees[pl]);
        }
      }
    }
  }
}

void TestMakeAllPublicStates(const std::string &game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);

  for (PublicState &s : all->public_states) {
// Debug print:
//    std::cout << "----" << std::endl;
//    std::cout << "Obs: " << s.public_tensor.Tensor() << "\n";
//    for (int pl = 0; pl < 2; ++pl) {
//      std::cout << "Nodes " << pl << ":\n";
//      for (const InfostateNode* node : s.nodes[pl]) {
//        std::cout << "  " << node->TreePath() << "\n";
//        std::cout << "  States:\n";
//        for (const std::unique_ptr<State> & state
//            : node->corresponding_states()) {
//          std::cout << "    " << state->HistoryString() << "\n";
//        }
//      }
//    }

    using History = std::vector<Action>;
    std::unordered_set<History, absl::Hash<History>> state_histories;
    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_FALSE(s.nodes[pl].empty());
      SPIEL_CHECK_FALSE(s.nodes[0][0]->corresponding_states().empty());
      State *a_state = s.nodes[0][0]->corresponding_states()[0].get();

      for (const algorithms::InfostateNode *node : s.nodes[pl]) {
        SPIEL_CHECK_FALSE(node->corresponding_states().empty());
        SPIEL_CHECK_EQ(node->tree().acting_player(), pl);

        for (const std::unique_ptr<State> &state
            : node->corresponding_states()) {
          const auto &h = state->History();
          SPIEL_CHECK_EQ(a_state->MoveNumber(), state->MoveNumber());
          if (pl == 0) {
            SPIEL_CHECK_TRUE(state_histories.find(h) == state_histories.end());
            state_histories.insert(h);
          } else {
            SPIEL_CHECK_TRUE(state_histories.find(h) != state_histories.end());
          }
        }
      }
    }
  }
}

void TestFullPokerCardConversion() {
  std::vector<std::pair<int, int> > test_inputs = {{2, 6}, {3, 9}, {4, 11}};
  for (std::pair<int, int> game_spec : test_inputs) {
    int num_suits = game_spec.first;
    int num_ranks = game_spec.second;
    SPIEL_CHECK_LE(num_suits, 4);
    SPIEL_CHECK_LE(num_ranks, 13);
    std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                       "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks="
        + std::to_string(num_ranks)
        + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";

    std::shared_ptr<const Game> game = LoadGame(name);

    std::unique_ptr<State> initial_state = game->NewInitialState();

    algorithms::PokerData poker_data = algorithms::PokerData(*initial_state);

    std::vector<int> new_card(1, 0);

    for (int card = 0; card < num_ranks * num_suits; card++) {
      std::unique_ptr<State> child = initial_state->Child(card);

      auto poker_state = down_cast<const open_spiel::universal_poker::UniversalPokerState &>(*child);

      universal_poker::logic::CardSet card_set_from_state = universal_poker::logic::CardSet();

      card_set_from_state.AddCard(poker_state.acpc_state().hole_cards(0, 0));

      new_card[0] = ConvertToFullPokerCard(card, poker_data);

      universal_poker::logic::CardSet card_set_from_convert = universal_poker::logic::CardSet(new_card);

      SPIEL_CHECK_EQ(card_set_from_state.ToString(), card_set_from_convert.ToString());
    }
  }
}

void UpdateChanceReaches(std::vector<double> &chance_reaches,
                         const algorithms::PokerData &poker_data,
                         const std::vector<int> &cards) {
  chance_reaches;
  for (int card : cards) {
    for (int hand_index : poker_data.card_to_hands_.at(card)) {
      chance_reaches[hand_index] = 0;
    }
  }
  double mag = 0;
  for (double mag_part : chance_reaches) {
    mag += mag_part;
  }
  for (double &chance_reach : chance_reaches) {
    chance_reach /= mag;
  }
}

struct TestInputs {
  TestInputs(std::pair<int, int> game_specification_in, std::vector<int> board_in, std::vector<double> beliefs_in) :
      game_specification(std::move(game_specification_in)),
      board(std::move(board_in)),
      beliefs(std::move(beliefs_in)) {};

  std::pair<int, int> game_specification;
  std::vector<int> board;
  std::vector<double> beliefs;
};

void PokerTerminalEvaluatorTest() {
  std::vector<TestInputs> test_configurations;
  // Adding first test configuration
  std::vector<double> beliefs_in(66, 0);
  std::iota(beliefs_in.begin(), beliefs_in.end(), 1);
  for(double & i : beliefs_in) {
    i /= 66;
  }
  std::vector<int> board_in = {0, 2, 7, 9, 11};
  std::pair<int, int> game_specification_in = {2, 6};
  test_configurations.emplace_back(game_specification_in, board_in, beliefs_in);
  // Second test configuration
  std::vector<double> beliefs_in_one(120, 0);
  beliefs_in_one[37] = 1;
  std::vector<int> board_in_one = {3, 5, 8, 10, 13};
  std::pair<int, int> game_specification_in_one = {2, 8};
  test_configurations.emplace_back(game_specification_in_one, board_in_one, beliefs_in_one);
  // Third test configuration
  std::vector<double> beliefs_in_two(45, 0);
  std::iota(beliefs_in_two.begin(), beliefs_in_two.end(), 1);
  for (double &i : beliefs_in_two) {
    i /= 45;
  }
  std::vector<int> board_in_two = {0, 2, 7, 9, 6};
  std::pair<int, int> game_specification_in_two = {2, 5};
  test_configurations.emplace_back(game_specification_in_two, board_in_two, beliefs_in_two);
  // More test configurations can be added
  for (const TestInputs &test_inputs : test_configurations) {
    int num_suits = test_inputs.game_specification.first;
    int num_ranks = test_inputs.game_specification.second;
    SPIEL_CHECK_LE(num_suits, 4);
    SPIEL_CHECK_LE(num_ranks, 13);
    SPIEL_CHECK_EQ(test_inputs.beliefs.size(), (num_suits * num_ranks) * (num_suits * num_ranks - 1) / 2);
    std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                       "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks="
        + std::to_string(num_ranks)
        + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";

    std::shared_ptr<const Game> game = LoadGame(name);

    std::unique_ptr<State> state = game->NewInitialState();

    std::vector<int> board_cards = test_inputs.board;
    for (int board_card : board_cards) {
      SPIEL_CHECK_LE(board_card, num_suits * num_ranks);
    }

    //Poker specific part to construct the subgame

    // Deal 4 cards
    state->ApplyAction(0);
    state->ApplyAction(1);
    state->ApplyAction(2);
    state->ApplyAction(3);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal 3 board cards (Flop)
    state->ApplyAction(board_cards[0]);
    state->ApplyAction(board_cards[1]);
    state->ApplyAction(board_cards[2]);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal board card (Turn)
    state->ApplyAction(board_cards[3]);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal board card (River)
    state->ApplyAction(board_cards[4]);

    algorithms::PokerData poker_data = algorithms::PokerData(*state);

    std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);
    UpdateChanceReaches(chance_reaches, poker_data, board_cards);

    std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
    std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

    std::vector<std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
        state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage, board_cards);

    auto poker_specific_subgame = std::make_shared<Subgame>(game, public_observer, trees);

    std::shared_ptr<const PublicStateEvaluator>
        poker_terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data, board_cards);

    std::shared_ptr<const PublicStateEvaluator>
        poker_terminal_evaluator_q = std::make_shared<const PokerTerminalEvaluatorQuadratic>(poker_data, board_cards);

    // General subgame construction
    std::vector<std::unique_ptr<State>> starting_states;
    std::unique_ptr<State> initial_state = game->NewInitialState();

    for (Action action : initial_state->LegalActions()) {
      if (std::find(board_cards.begin(), board_cards.end(), action) != board_cards.end()) {
        continue;
      }
      auto child_one = initial_state->Child(action);
      for (Action action_one : child_one->LegalActions()) {
        if (std::find(board_cards.begin(), board_cards.end(), action_one) != board_cards.end()) {
          continue;
        }
        auto child_two = child_one->Child(action_one);
        for (Action action_two : child_two->LegalActions()) {
          if (std::find(board_cards.begin(), board_cards.end(), action_two) != board_cards.end()) {
            continue;
          }
          auto child_three = child_two->Child(action_two);
          for (Action action_three : child_three->LegalActions()) {
            if (std::find(board_cards.begin(), board_cards.end(), action_three) != board_cards.end()) {
              continue;
            }
            auto child_four = child_three->Child(action_three);
            // BothCall
            child_four->ApplyAction(1);
            child_four->ApplyAction(1);

            // Deal 3 board cards (Flop)
            child_four->ApplyAction(board_cards[0]);
            child_four->ApplyAction(board_cards[1]);
            child_four->ApplyAction(board_cards[2]);

            // BothCall
            child_four->ApplyAction(1);
            child_four->ApplyAction(1);

            // Deal board card (Turn)
            child_four->ApplyAction(board_cards[3]);

            // BothCall
            child_four->ApplyAction(1);
            child_four->ApplyAction(1);

            // Deal board card (River)
            child_four->ApplyAction(board_cards[4]);
            starting_states.push_back(std::move(child_four));
          }
        }
      }
    }

    std::vector<double> general_chance_reaches(starting_states.size(), 1. / starting_states.size());

    std::vector<std::shared_ptr<algorithms::InfostateTree>>
        trees_general = algorithms::MakeInfostateTrees(
        starting_states, general_chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

    auto general_subgame = std::make_shared<Subgame>(game, public_observer, trees_general);

    auto general_terminal_evaluator = MakeTerminalEvaluator();

    SPIEL_CHECK_EQ(general_subgame->public_states.size(), poker_specific_subgame->public_states.size());
    for (int i = 0; i < general_subgame->public_states.size(); i++) {
      auto general_public_state = &general_subgame->public_states[i];
      if (!general_public_state->IsTerminal()) {
        continue;
      }
      auto poker_public_state = &poker_specific_subgame->public_states[i];
      std::vector<double> beliefs = test_inputs.beliefs;
      SPIEL_CHECK_EQ(poker_public_state->public_tensor.Tensor(), general_public_state->public_tensor.Tensor());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string_one = poker_public_state->nodes[0][node_index]->infostate_string();
        std::string infostate_string_two = poker_public_state->nodes[1][node_index]->infostate_string();
        poker_public_state->beliefs[0][node_index] = beliefs[node_index];
        poker_public_state->beliefs[1][node_index] = beliefs[node_index];
        for (int general_node_index = 0; general_node_index < general_public_state->nodes[0].size();
             general_node_index++) {
          if (general_public_state->nodes[0][general_node_index]->infostate_string() == infostate_string_one) {
            general_public_state->beliefs[0][general_node_index] = beliefs[node_index];
          }
          if (general_public_state->nodes[1][general_node_index]->infostate_string() == infostate_string_two) {
            general_public_state->beliefs[1][general_node_index] = beliefs[node_index];
          }
        }
      }
      auto general_context = general_terminal_evaluator->CreateContext(*general_public_state);
      auto poker_context = poker_terminal_evaluator->CreateContext(*poker_public_state);
      auto poker_context_q = poker_terminal_evaluator_q->CreateContext(*poker_public_state);
      general_terminal_evaluator->EvaluatePublicState(general_public_state, general_context.get());
      poker_terminal_evaluator->EvaluatePublicState(poker_public_state, poker_context.get());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string = poker_public_state->nodes[0][node_index]->infostate_string();
        double general_value = 0;
        for (int general_node_index = 0; general_node_index < general_public_state->nodes[0].size();
             general_node_index++) {
          if (general_public_state->nodes[0][general_node_index]->infostate_string() == infostate_string) {
            general_value += general_public_state->values[0][general_node_index];
          }
        }
        SPIEL_CHECK_FLOAT_NEAR(poker_public_state->values[0][node_index], general_value, 0.000001);
      }
      double general_sum = 0;
      double poker_sum = 0;
      for (double poker_value : poker_public_state->values[0]) {
        poker_sum += poker_value;
      }
      for (double general_value : general_public_state->values[0]) {
        general_sum += general_value;
      }
      SPIEL_CHECK_FLOAT_NEAR(poker_sum, general_sum, 0.001);
      poker_terminal_evaluator_q->EvaluatePublicState(poker_public_state, poker_context_q.get());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string = poker_public_state->nodes[0][node_index]->infostate_string();
        double general_value = 0;
        for (int general_node_index = 0; general_node_index < general_public_state->nodes[0].size();
             general_node_index++) {
          if (general_public_state->nodes[0][general_node_index]->infostate_string() == infostate_string) {
            general_value += general_public_state->values[0][general_node_index];
          }
        }
        SPIEL_CHECK_FLOAT_NEAR(poker_public_state->values[0][node_index], general_value, 0.000001);
      }
      general_sum = 0;
      poker_sum = 0;
      for (double poker_value : poker_public_state->values[0]) {
        poker_sum += poker_value;
      }
      for (double general_value : general_public_state->values[0]) {
        general_sum += general_value;
      }
      SPIEL_CHECK_FLOAT_NEAR(poker_sum, general_sum, 0.001);
    }
  }
}

void GeneralPokerTerminalEvaluatorTest() {
  std::vector<TestInputs> test_configurations;
  // Adding first test configuration
  std::vector<double> beliefs_in(66, 0);
  std::iota(beliefs_in.begin(), beliefs_in.end(), 1);
  for (double &i : beliefs_in) {
    i /= 66;
  }
  std::vector<int> board_in = {0, 2, 7, 9, 11};
  std::pair<int, int> game_specification_in = {2, 6};
  test_configurations.emplace_back(game_specification_in, board_in, beliefs_in);
  // Second test configuration
  std::vector<double> beliefs_in_one(120, 0);
  beliefs_in_one[37] = 1;
  std::vector<int> board_in_one = {3, 5, 8, 10, 13};
  std::pair<int, int> game_specification_in_one = {2, 8};
  test_configurations.emplace_back(game_specification_in_one, board_in_one, beliefs_in_one);
  // More test configurations can be added
  for (const TestInputs &test_inputs : test_configurations) {
    int num_suits = test_inputs.game_specification.first;
    int num_ranks = test_inputs.game_specification.second;
    SPIEL_CHECK_LE(num_suits, 4);
    SPIEL_CHECK_LE(num_ranks, 13);
    SPIEL_CHECK_EQ(test_inputs.beliefs.size(), (num_suits * num_ranks) * (num_suits * num_ranks - 1) / 2);
    std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                       "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks="
        + std::to_string(num_ranks)
        + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";

    std::shared_ptr<const Game> game = LoadGame(name);

    std::unique_ptr<State> state = game->NewInitialState();

    std::vector<int> board_cards = test_inputs.board;
    for (int board_card : board_cards) {
      SPIEL_CHECK_LE(board_card, num_suits * num_ranks);
    }

    //Poker specific part to construct the subgame

    // Deal 4 cards
    state->ApplyAction(0);
    state->ApplyAction(1);
    state->ApplyAction(2);
    state->ApplyAction(3);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal 3 board cards (Flop)
    state->ApplyAction(board_cards[0]);
    state->ApplyAction(board_cards[1]);
    state->ApplyAction(board_cards[2]);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal board card (Turn)
    state->ApplyAction(board_cards[3]);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal board card (River)
    state->ApplyAction(board_cards[4]);

    algorithms::PokerData poker_data = algorithms::PokerData(*state);

    std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);
    UpdateChanceReaches(chance_reaches, poker_data, board_cards);

    std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
    std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

    std::vector<std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
        state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage, board_cards);

    auto poker_specific_subgame = std::make_shared<Subgame>(game, public_observer, trees);

    std::shared_ptr<const PublicStateEvaluator>
        general_poker_terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

    std::shared_ptr<const PublicStateEvaluator>
        poker_terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data, board_cards);

    for (int i = 0; i < poker_specific_subgame->public_states.size(); i++) {
      auto poker_public_state = &poker_specific_subgame->public_states[i];
      if (!poker_public_state->IsTerminal()) {
        continue;
      }
      std::vector<double> beliefs = test_inputs.beliefs;
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string_one = poker_public_state->nodes[0][node_index]->infostate_string();
        std::string infostate_string_two = poker_public_state->nodes[1][node_index]->infostate_string();
        poker_public_state->beliefs[0][node_index] = beliefs[node_index];
        poker_public_state->beliefs[1][node_index] = beliefs[node_index];
      }
      std::vector<double> general_poker_values;
      std::vector<double> poker_values;
      auto poker_context = poker_terminal_evaluator->CreateContext(*poker_public_state);
      auto general_poker_context = general_poker_terminal_evaluator->CreateContext(*poker_public_state);
      general_poker_terminal_evaluator->EvaluatePublicState(poker_public_state, general_poker_context.get());
      general_poker_values.reserve(poker_public_state->nodes[0].size());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        general_poker_values.push_back(poker_public_state->values[0][node_index]);
      }
      poker_terminal_evaluator->EvaluatePublicState(poker_public_state, poker_context.get());
      poker_values.reserve(poker_public_state->nodes[0].size());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        poker_values.push_back(poker_public_state->values[0][node_index]);
      }
      double general_poker_sum = 0;
      double poker_sum = 0;
      for (int node_index = 0; node_index < poker_values.size(); node_index++) {
        general_poker_sum += general_poker_values[node_index];
        poker_sum += poker_values[node_index];
//        std::cout << "General: " << general_poker_values[node_index] << " River: " << poker_values[node_index] << "\n";
        SPIEL_CHECK_FLOAT_NEAR(poker_values[node_index], general_poker_values[node_index], 0.001);
      }
//      std::cout << general_poker_sum << " " << poker_sum << "\n";
      SPIEL_CHECK_FLOAT_NEAR(poker_sum, general_poker_sum, 0.001);
    }
  }
}

void GeneralPokerTerminalEvaluatorTurnTest() {
  std::vector<TestInputs> test_configurations;
  // Adding first test configuration
  std::vector<double> beliefs_in(45, 0);
  std::iota(beliefs_in.begin(), beliefs_in.end(), 1);
  for (double &i : beliefs_in) {
    i /= 45;
  }
  std::vector<int> board_in = {0, 2, 7, 9};
  std::pair<int, int> game_specification_in = {2, 5};
  test_configurations.emplace_back(game_specification_in, board_in, beliefs_in);
  // Second test configuration
  std::vector<double> beliefs_in_one(45, 0);
  beliefs_in_one[37] = 1;
  std::vector<int> board_in_one = {1, 3, 6, 7};
  std::pair<int, int> game_specification_in_one = {2, 5};
  test_configurations.emplace_back(game_specification_in_one, board_in_one, beliefs_in_one);
  // Third test configuration
  std::vector<double> beliefs_in_two(36, 0);
  std::iota(beliefs_in_two.begin(), beliefs_in_two.end(), 1);
  for (double &i : beliefs_in_two) {
    i /= 36;
  }
  std::vector<int> board_in_two = {0, 1, 5, 8};
  std::pair<int, int> game_specification_in_two = {3, 3};
  test_configurations.emplace_back(game_specification_in_two, board_in_two, beliefs_in_two);
  // More test configurations can be added
  int test = 0;
  for (const TestInputs &test_inputs : test_configurations) {
    int num_suits = test_inputs.game_specification.first;
    int num_ranks = test_inputs.game_specification.second;
    SPIEL_CHECK_LE(num_suits, 4);
    SPIEL_CHECK_LE(num_ranks, 13);
    SPIEL_CHECK_EQ(test_inputs.beliefs.size(), (num_suits * num_ranks) * (num_suits * num_ranks - 1) / 2);
    std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                       "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks="
        + std::to_string(num_ranks)
        + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";

    std::shared_ptr<const Game> game = LoadGame(name);

    std::unique_ptr<State> state = game->NewInitialState();

    std::vector<int> board_cards = test_inputs.board;
    for (int board_card : board_cards) {
      SPIEL_CHECK_LE(board_card, num_suits * num_ranks);
    }

    //Poker specific part to construct the subgame

    // Deal 4 cards
    state->ApplyAction(0);
    state->ApplyAction(1);
    state->ApplyAction(2);
    state->ApplyAction(3);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal 3 board cards (Flop)
    state->ApplyAction(board_cards[0]);
    state->ApplyAction(board_cards[1]);
    state->ApplyAction(board_cards[2]);

    // BothCall
    state->ApplyAction(1);
    state->ApplyAction(1);

    // Deal board card (Turn)
    state->ApplyAction(board_cards[3]);

    algorithms::PokerData poker_data = algorithms::PokerData(*state);

    std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);
    UpdateChanceReaches(chance_reaches, poker_data, board_cards);

    std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
    std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

    std::vector<std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
        state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage, board_cards);

    auto poker_specific_subgame = std::make_shared<Subgame>(game, public_observer, trees);

    std::shared_ptr<const PublicStateEvaluator>
        poker_terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

    // General subgame construction
    std::vector<std::unique_ptr<State>> starting_states;
    std::unique_ptr<State> initial_state = game->NewInitialState();

    for (Action action : initial_state->LegalActions()) {
      if (std::find(board_cards.begin(), board_cards.end(), action) != board_cards.end()) {
        continue;
      }
      auto child_one = initial_state->Child(action);
      for (Action action_one : child_one->LegalActions()) {
        if (std::find(board_cards.begin(), board_cards.end(), action_one) != board_cards.end()) {
          continue;
        }
        auto child_two = child_one->Child(action_one);
        for (Action action_two : child_two->LegalActions()) {
          if (std::find(board_cards.begin(), board_cards.end(), action_two) != board_cards.end()) {
            continue;
          }
          auto child_three = child_two->Child(action_two);
          for (Action action_three : child_three->LegalActions()) {
            if (std::find(board_cards.begin(), board_cards.end(), action_three) != board_cards.end()) {
              continue;
            }
            auto child_four = child_three->Child(action_three);
            // BothCall
            child_four->ApplyAction(1);
            child_four->ApplyAction(1);

            // Deal 3 board cards (Flop)
            child_four->ApplyAction(board_cards[0]);
            child_four->ApplyAction(board_cards[1]);
            child_four->ApplyAction(board_cards[2]);

            // BothCall
            child_four->ApplyAction(1);
            child_four->ApplyAction(1);

            // Deal board card (Turn)
            child_four->ApplyAction(board_cards[3]);
            starting_states.push_back(std::move(child_four));
          }
        }
      }
    }

    std::vector<double> general_chance_reaches(starting_states.size(), 1. / starting_states.size());

    std::vector<std::shared_ptr<algorithms::InfostateTree>>
        trees_general = algorithms::MakeInfostateTrees(
        starting_states, general_chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

    auto general_subgame = std::make_shared<Subgame>(game, public_observer, trees_general);

    auto general_terminal_evaluator = MakeTerminalEvaluator();

    SPIEL_CHECK_EQ(general_subgame->public_states.size(), poker_specific_subgame->public_states.size());
    for (int i = 0; i < general_subgame->public_states.size(); i++) {
      auto general_public_state = &general_subgame->public_states[i];
      if (!general_public_state->IsTerminal()) {
        continue;
      }
      PublicState *poker_public_state = nullptr;
      for (int j = 0; j < poker_specific_subgame->public_states.size(); j++) {
        poker_public_state = &poker_specific_subgame->public_states[j];
        if (poker_public_state->public_tensor.Tensor() == general_public_state->public_tensor.Tensor()) {
          break;
        }
      }
      SPIEL_CHECK_TRUE(poker_public_state);
      std::vector<double> beliefs = test_inputs.beliefs;
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string_one = poker_public_state->nodes[0][node_index]->infostate_string();
        std::string infostate_string_two = poker_public_state->nodes[1][node_index]->infostate_string();
        poker_public_state->beliefs[0][node_index] = beliefs[node_index];
        poker_public_state->beliefs[1][node_index] = beliefs[node_index];
        for (int general_node_index = 0; general_node_index < general_public_state->nodes[0].size();
             general_node_index++) {
          if (general_public_state->nodes[0][general_node_index]->infostate_string() == infostate_string_one) {
            general_public_state->beliefs[0][general_node_index] = beliefs[node_index];
          }
          if (general_public_state->nodes[1][general_node_index]->infostate_string() == infostate_string_two) {
            general_public_state->beliefs[1][general_node_index] = beliefs[node_index];
          }
        }
      }
      auto general_context = general_terminal_evaluator->CreateContext(*general_public_state);
      auto poker_context = poker_terminal_evaluator->CreateContext(*poker_public_state);
      general_terminal_evaluator->EvaluatePublicState(general_public_state, general_context.get());
      poker_terminal_evaluator->EvaluatePublicState(poker_public_state, poker_context.get());
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        std::string infostate_string = poker_public_state->nodes[0][node_index]->infostate_string();
        double general_value = 0;
        for (int general_node_index = 0; general_node_index < general_public_state->nodes[0].size();
             general_node_index++) {
          if (general_public_state->nodes[0][general_node_index]->infostate_string() == infostate_string) {
            general_value += general_public_state->values[0][general_node_index];
          }
        }
        SPIEL_CHECK_FLOAT_NEAR(poker_public_state->values[0][node_index], general_value, 0.000001);
      }
      double general_sum = 0;
      double poker_sum = 0;
      for (double poker_value : poker_public_state->values[0]) {
        poker_sum += poker_value;
      }
      for (double general_value : general_public_state->values[0]) {
        general_sum += general_value;
      }
      SPIEL_CHECK_FLOAT_NEAR(poker_sum, general_sum, 0.001);
    }
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char **argv) {
//  std::vector<std::string> test_games = {
//      "kuhn_poker",
//      "leduc_poker",
//      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)",
//  };

//  for (const std::string &game_name : test_games) {
//    open_spiel::papers_with_code::TestTerminalEvaluatorHasSameIterations(
//        game_name);
//    open_spiel::papers_with_code::TestRecursiveDepthLimitedSolving(game_name);
//    open_spiel::papers_with_code::TestMakeAllPublicStates(game_name);
//  }

  open_spiel::papers_with_code::TestFullPokerCardConversion();

  open_spiel::papers_with_code::PokerTerminalEvaluatorTest();

  open_spiel::papers_with_code::GeneralPokerTerminalEvaluatorTest();

  open_spiel::papers_with_code::GeneralPokerTerminalEvaluatorTurnTest();
}
