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
#include "algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "subgame.h"
#include "algorithms/cfr.h"
#include "algorithms/best_response.h"
#include "infostate_tree_br.h"
#include "libratus_endgame_values.h"
#include "turn_poker_net.h"

#include <iostream>

namespace open_spiel {
namespace papers_with_code {
namespace {

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

std::string CreatePokerCertificateFromNode(State &state, Player player) {
  if (state.IsTerminal()) {
    return "{}";
  } else if (state.IsPlayerActing(player)) {
    std::string ret = "[";
    for (Action action : state.LegalActions()) {
      ret.append(CreatePokerCertificateFromNode(*state.Child(action), player));
    }
    ret.append("]");
    return ret;
  } else {
    std::string ret = "(";
    for (Action action : state.LegalActions()) {
      ret.append(CreatePokerCertificateFromNode(*state.Child(action), player));
    }
    ret.append(")");
    return ret;
  }
}

std::string CreateDepthPokerCertificateFromNode(State &state, Player player, int depth) {
  std::string ret;
  for (int i = 0; i < depth; i++) {
    ret.append(" ");
  }
  if (state.IsTerminal()) {
    ret.append("{}\n");
  } else if (state.IsPlayerActing(player)) {
    ret.append("[\n");
    for (Action action : state.LegalActions()) {
      ret.append(CreateDepthPokerCertificateFromNode(*state.Child(action), player, depth + 1));
    }
    for (int i = 0; i < depth; i++) {
      ret.append(" ");
    }
    ret.append("]\n");
  } else {
    ret.append("(\n");
    for (Action action : state.LegalActions()) {
      ret.append(CreateDepthPokerCertificateFromNode(*state.Child(action), player, depth + 1));
    }
    for (int i = 0; i < depth; i++) {
      ret.append(" ");
    }
    ret.append(")\n");
  }
  return ret;
}

std::pair<int, int> UniversalPokerTurnTest(int iterations) {
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
//  std::string name = "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
//                     "firstPlayer=2 1 1 "
//                     "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
//                     "1 1,stack=20000 20000,bettingAbstraction=fcpa)";
  std::shared_ptr<const Game> game = LoadGame(name);

  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<int> cards = {4, 31, 10, 15, 20};

  // Deal 4 cards
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(3);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal 3 board cards (Flop)
  state->ApplyAction(cards[0]);
  state->ApplyAction(cards[1]);
  state->ApplyAction(cards[2]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (Turn)
  state->ApplyAction(cards[3]);

  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

  std::vector<std::shared_ptr<algorithms::InfostateTree>>
      trees = algorithms::MakePokerInfostateTrees(state, std::vector<double>(1326, 1. / 1326),
                                                  infostate_observer, 1000, kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  algorithms::PokerData poker_data = algorithms::PokerData(*state);

  std::shared_ptr<const PublicStateEvaluator>
      terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorQuadratic>(poker_data, cards);

  SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                       std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFRPokerSpecificQuadratic(int iterations) {
//  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
//                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
//                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
  std::string name = "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
                     "firstPlayer=2 1 1 "
                     "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,stack=20000 20000,bettingAbstraction=fcpa)";
  std::shared_ptr<const Game> game = LoadGame(name);

  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<int> cards = {4, 31, 10, 15, 20};

  // Deal 4 cards
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(3);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal 3 board cards (Flop)
  state->ApplyAction(cards[0]);
  state->ApplyAction(cards[1]);
  state->ApplyAction(cards[2]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (Turn)
  state->ApplyAction(cards[3]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (River)
  state->ApplyAction(cards[4]);

  universal_poker::logic::CardSet card_set(cards);
  std::cout << card_set.ToString() << "\n";


  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

  std::vector<double> chance_reaches(1326, 1. / 1326);

  algorithms::PokerData poker_data = algorithms::PokerData(*state);

  UpdateChanceReaches(chance_reaches, poker_data, cards);

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakePokerInfostateTrees(state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator>
      terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorQuadratic>(poker_data, cards);

  SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                       std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFRPokerSpecificLinear(int iterations) {
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
//  std::string name = "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
//                     "firstPlayer=2 1 1 "
//                     "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
//                     "1 1,stack=20000 20000,bettingAbstraction=fcpa)";
  std::shared_ptr<const Game> game = LoadGame(name);

  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<int> cards = {4, 31, 10, 15, 20};

  // Deal 4 cards
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(3);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal 3 board cards (Flop)
  state->ApplyAction(cards[0]);
  state->ApplyAction(cards[1]);
  state->ApplyAction(cards[2]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (Turn)
  state->ApplyAction(cards[3]);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (River)
  state->ApplyAction(cards[4]);

  auto start = std::chrono::high_resolution_clock::now();
  std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

  std::vector<double> chance_reaches(1326, 1. / 1326);

  algorithms::PokerData poker_data = algorithms::PokerData(*state);

  UpdateChanceReaches(chance_reaches, poker_data, cards);

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakePokerInfostateTrees(state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator>
      terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data, cards);

  SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                       std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int player = 0; player < 2; player++) {
    SubgameSolver best_response = SubgameSolver(out, nullptr, terminal_evaluator,
                                                std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
    int opponent = 1 - player;
    auto fixed_policy = solver.AveragePolicy();
    algorithms::BanditVector &opponent_bandits = best_response.bandits()[opponent];
    for (algorithms::DecisionId id : opponent_bandits.range()) {
      algorithms::InfostateNode *node = out->trees[opponent]->decision_infostate(id);
      std::string infostate = node->infostate_string();
      ActionsAndProbs infostate_policy = fixed_policy->GetStatePolicy(infostate);
      std::vector<double> probs = GetProbs(infostate_policy);
      auto fixable_bandit = std::make_unique<algorithms::bandits::FixableStrategy>(probs);
      opponent_bandits[id] = std::move(fixable_bandit);
    }
    best_response.RunSimultaneousIterations(100);
    auto cf_values = best_response.GetCfValues();
    std::cout << cf_values[0][0] << ";" << cf_values[1][0] << "\n";
    std::cout << best_response.RootValues() << "\n";
  }
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

void SmallUniversalPokerTrunkTest() {
  int num_suits = 2;
  int num_ranks = 6;
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks=" + std::to_string(num_ranks)
      + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
  std::shared_ptr<const Game> game = LoadGame(name);

  std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});

  // General part
//  std::vector<std::unique_ptr<State>> starting_states;
//  std::unique_ptr<State> initial_state = game->NewInitialState();
//  std::vector<double> chance_reaches(11880, 1. / 11880);
//
//  for (Action action : initial_state->LegalActions()) {
//    auto child_one = initial_state->Child(action);
//    for (Action action_one : child_one->LegalActions()) {
//      auto child_two = child_one->Child(action_one);
//      for (Action action_two : child_two->LegalActions()) {
//        auto child_three = child_two->Child(action_two);
//        for (Action action_three : child_three->LegalActions()) {
//          auto child_four = child_three->Child(action_three);
//          starting_states.push_back(std::move(child_four));
//        }
//      }
//    }
//  }
//
//
//  std::vector<std::shared_ptr<algorithms::InfostateTree>>
//      tree_general = algorithms::MakeInfostateTrees(starting_states,
//                                                    chance_reaches,
//                                                    infostate_observer,
//                                                    2,
//                                                    kDlCfrInfostateTreeStorage);
//
//  std::cout << tree_general[0]->root().MakeCertificate() << "\n";
//  std::cout << tree_general[1]->root().MakeCertificate() << "\n";

  // Poker specific part
  std::unique_ptr<State> state = game->NewInitialState();
  // Deal 4 cards
  state->ApplyAction(0);
  state->ApplyAction(1);
  state->ApplyAction(2);
  state->ApplyAction(3);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal 3 board cards (Flop)
  state->ApplyAction(4);
  state->ApplyAction(5);
  state->ApplyAction(6);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (Turn)
  state->ApplyAction(7);

  // BothCall
  state->ApplyAction(1);
  state->ApplyAction(1);

  // Deal board card (River)
  state->ApplyAction(8);

  std::vector<std::shared_ptr<algorithms::InfostateTree>>
      tree_poker = algorithms::MakePokerInfostateTrees(state, std::vector<double>(66, 1. / 66),
                                                       infostate_observer, 1000, kDlCfrInfostateTreeStorage);

  std::unordered_map<char, char> reverse_brackets = {
      {'(', ')'},
      {')', '('},
      {']', '['},
      {'[', ']'},
      {'{', '}'},
      {'}', '{'}
  };

  std::string to_reverse;
  std::cout << tree_poker[0]->root().children()[0]->MakeCertificate() << "\n";
  to_reverse = CreatePokerCertificateFromNode(*state, 0);
  std::cout << to_reverse << "\n";
  for (int i = to_reverse.size() - 1; i >= 0; i--) {
    std::cout << reverse_brackets[to_reverse[i]];
  }
  std::cout << tree_poker[1]->root().children()[0]->MakeCertificate() << "\n";
  to_reverse = CreatePokerCertificateFromNode(*state, 1);
  std::cout << to_reverse << "\n";
  std::reverse(to_reverse.begin(), to_reverse.end());
  std::cout << to_reverse << "\n";
}

void TestSameInfostates() {
  std::vector<std::pair<int, int> > test_inputs = {{2, 6}};
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

    std::unique_ptr<State> state = game->NewInitialState();

    std::vector<int> board_cards = {0, 2, 7, 9, 11};
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
        state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

    auto poker_specific_subgame = std::make_shared<Subgame>(game, public_observer, trees);

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

    SPIEL_CHECK_EQ(general_subgame->public_states.size(), poker_specific_subgame->public_states.size());
    for (int i = 0; i < general_subgame->public_states.size(); i++) {
      auto general_public_state = &general_subgame->public_states[i];
      if (!general_public_state->IsTerminal()) {
        continue;
      }
      auto poker_public_state = &poker_specific_subgame->public_states[i];
      std::vector<std::set<std::string>> general_infostates(2, std::set<std::string>());
      std::vector<std::set<std::string>> poker_infostates(2, std::set<std::string>());
      SPIEL_CHECK_EQ(general_public_state->nodes[0].size(), general_public_state->nodes[1].size());
      for (int node_index = 0; node_index < general_public_state->nodes[0].size(); node_index++) {
        for (Player player = 0; player < 2; player++) {
          general_infostates[player].insert(general_public_state->nodes[player][node_index]->infostate_string());
        }
      }
      for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
        for (Player player = 0; player < 2; player++) {
          if (poker_public_state->nodes[player][node_index]->terminal_chance_reach_prob() > 0) {
            poker_infostates[player].insert(poker_public_state->nodes[player][node_index]->infostate_string());
          }
        }
      }
      SPIEL_CHECK_TRUE(poker_infostates == general_infostates);
    }
  }
}

std::pair<int, int> SolverLeducCFRInfostate(int iterations) {
  std::string name = "leduc_poker";

  std::shared_ptr<const Game> game = LoadGame(name);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::InfostateCFR solver(*game);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  solver.RunSimultaneousIterations(iterations);
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> SolverLeducCFREfg(int iterations) {
  std::string name = "leduc_poker";

  std::shared_ptr<const Game> game = LoadGame(name);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> SolverLeducCFREfgHashed(int iterations) {
  std::string name = "leduc_poker";

  std::shared_ptr<const Game> game = LoadGame(name);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFREfg(int iterations) {
  std::mt19937 rng(0);

  int pot_size = 200;
  std::string board_cards = "9s7c5s4h3c";

  std::vector<double> uniform_reaches;
  uniform_reaches.reserve(2 * universal_poker::kSubgameUniqueHands);
  for (int i = 0; i < 2 * universal_poker::kSubgameUniqueHands; ++i) {
    uniform_reaches.push_back(1. / (2 * universal_poker::kSubgameUniqueHands));
  }
  std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(rng, pot_size, board_cards, uniform_reaches);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

std::pair<int, int> UniversalPokerRiverCFREfgHashed(int iterations) {
  std::mt19937 rng(0);

  int pot_size = 200;
  std::string board_cards = "9s7c5s4h3c";

  std::vector<double> uniform_reaches;
  uniform_reaches.reserve(2 * universal_poker::kSubgameUniqueHands);
  for (int i = 0; i < 2 * universal_poker::kSubgameUniqueHands; ++i) {
    uniform_reaches.push_back(1. / (2 * universal_poker::kSubgameUniqueHands));
  }
  std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(rng, pot_size, board_cards, uniform_reaches);

  auto start = std::chrono::high_resolution_clock::now();
  algorithms::CFRSolverBase solver(*game, false, false, true, true);
  auto end = std::chrono::high_resolution_clock::now();
  auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    solver.EvaluateAndUpdatePolicy();
  }
  end = std::chrono::high_resolution_clock::now();
  auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  return std::pair<int, int>(setup_duration.count(), run_duration.count());
}

void MeasureTime(int runs, int iterations, std::pair<int, int> (*f)(int)) {
  std::vector<std::pair<int, int>> collected_times;
  collected_times.reserve(runs);
  for (int i = 0; i < runs; i++) {
    collected_times.push_back(f(iterations));
  }
  std::pair<int, int> cumulative_times(0, 0);
  for (auto &time_pair : collected_times) {
    cumulative_times.first += time_pair.first;
    cumulative_times.second += time_pair.second;
  }
  std::cout << "Average setup time: " << cumulative_times.first / runs << "ms\n";
  std::cout << "Average runin time: " << cumulative_times.second / runs << "ms\n";
}

void CheckExploitabilityTwo() {
  int iterations = 1;

  std::string name = "leduc_poker";

  std::shared_ptr<const Game> game = LoadGame(name);

  algorithms::CFRSolverBase solver_efg(*game, false, false, true, true);

  for (int i = 0; i < iterations; i++) {
    solver_efg.EvaluateAndUpdatePolicy();
  }

  auto policy_efg = solver_efg.AveragePolicy();
  std::cout << "Normal efg cfr exploitability: " << algorithms::Exploitability(*game, *policy_efg) << "\n";

  algorithms::InfostateCFR solver_is(*game);

  solver_is.RunSimultaneousIterations(iterations);
  auto policy_is = solver_is.AveragePolicy();
  algorithms::TabularBestResponse best_response_pone(*game, 0, &*policy_is);
  std::cout << best_response_pone.Value(*game->NewInitialState()) << "\n";
  algorithms::TabularBestResponse best_response_ptwo(*game, 1, &*policy_is);
  std::cout << best_response_ptwo.Value(*game->NewInitialState()) << "\n";
  std::cout << BestResponse(solver_is.trees(), *policy_is) << "\n";

  for (int player = 0; player < 2; player++) {
    algorithms::InfostateCFR best_response_is = algorithms::InfostateCFR(*game);
    int opponent = 1 - player;
    algorithms::BanditVector &opponent_bandits = best_response_is.bandits_modifiable()[opponent];
    for (algorithms::DecisionId id : opponent_bandits.range()) {
      algorithms::InfostateNode *node = best_response_is.trees()[opponent]->decision_infostate(id);
      std::string infostate = node->infostate_string();
      ActionsAndProbs infostate_policy = policy_is->GetStatePolicy(infostate);
      std::vector<double> probs = GetProbs(infostate_policy);
      auto fixable_bandit = std::make_unique<algorithms::bandits::FixableStrategy>(probs);
      opponent_bandits[id] = std::move(fixable_bandit);
    }
    best_response_is.RunSimultaneousIterations(100);
    auto cf_values = best_response_is.RootValues();
    std::cout << cf_values << "\n";
  }

  std::cout << "Infostate cfr exploitability: " << algorithms::Exploitability(*game, *policy_is) << "\n";
}

void CheckExploitability(int iterations) {
  std::string name = "leduc_poker";

  std::shared_ptr<const Game> game = LoadGame(name);

  algorithms::CFRSolverBase solver_efgsave(*game, false, true, true, true);

  for (int i = 0; i < iterations; i++) {
    solver_efgsave.EvaluateAndUpdatePolicy();
  }

  auto policy_efgsave = solver_efgsave.AveragePolicy();
  std::cout << "Save states efg exploitability: " << algorithms::Exploitability(*game, *policy_efgsave) << "\n";
//
//  algorithms::CFRSolverBase solver_efg(*game, false, false, false, false);
//
//  for (int i = 0; i < iterations; i++) {
//    solver_efg.EvaluateAndUpdatePolicy();
//  }

//  auto policy_efg = solver_efg.AveragePolicy();
//  std::cout << "Normal efg exploitability: " << algorithms::Exploitability(*game, *policy_efg) << "\n";

  algorithms::InfostateCFR solver_is(*game);

  solver_is.RunSimultaneousIterations(iterations);
  auto policy_is = solver_is.AveragePolicy();
  std::cout << "Infostate exploitability: " << algorithms::Exploitability(*game, *policy_is) << "\n";
}

std::vector<int> GenerateNewBoardCard(const std::vector<int> &board_cards, std::mt19937 &mt) {
  std::vector<int> new_board_cards = board_cards;
  std::uniform_int_distribution<int> dist(0, 51);
  new_board_cards.push_back(dist(mt));
  while (std::find(board_cards.begin(), board_cards.end(), new_board_cards.back()) != board_cards.end()) {
    new_board_cards[4] = dist(mt);
  }
  return new_board_cards;
}

std::array<std::vector<double>, 2> GenerateRanges(
    const std::vector<int> &board_cards, int new_board_card, std::mt19937 &mt,
    const std::array<std::vector<double>, 2> &ranges) {
  int range_index = 0;
  std::array<std::vector<double>, 2> new_ranges = ranges;
  for (int card_one = 0; card_one < 51; card_one++) {
    for (int card_two = card_one + 1; card_two < 52; card_two++) {
      open_spiel::universal_poker::logic::CardSet hand(std::vector<int>({card_one, card_two}));
      if (std::find(board_cards.begin(), board_cards.end(), card_two) != board_cards.end() or
          std::find(board_cards.begin(), board_cards.end(), card_one) != board_cards.end()) {
        SPIEL_CHECK_EQ(ranges[0][range_index], 0);
        SPIEL_CHECK_EQ(ranges[1][range_index], 0);
      } else if (card_one == new_board_card or card_two == new_board_card) {
        new_ranges[0][range_index] = 0;
        new_ranges[1][range_index] = 0;
      } else {
        for (int player = 0; player < 2; player++) {
          std::uniform_real_distribution<double> distribution(0., new_ranges[player][range_index]);
          new_ranges[player][range_index] = distribution(mt);
        }
      }
      range_index++;
    }
  }
  return new_ranges;
}

std::vector<int> GenerateActions(
    const std::shared_ptr<const Game> &game, std::mt19937 &mt, const std::vector<int> &actions_so_far) {
  std::vector<int> action_sequence = actions_so_far;
  std::unique_ptr<State> state = game->NewInitialState();
  int chance_visited = 0;
  int action_index = 0;
  while (chance_visited < 9) {
    if (state->IsChanceNode()) {
      chance_visited++;
      state->ApplyAction(state->LegalActions()[0]);
    } else {
      if (action_index < actions_so_far.size()) {
        state->ApplyAction(actions_so_far[action_index]);
        action_index++;
      } else {
        std::vector<Action> actions = state->LegalActions();
        SPIEL_CHECK_GT(actions.size(), 0);
        absl::uniform_int_distribution<> dis(0, actions.size() - 1);
        Action action = universal_poker::kFold;
        while (action == universal_poker::kFold) {
          action = actions[dis(mt)];
        }
        state->ApplyAction(action);
        action_sequence.push_back(action);
      }
    }
  }
  return action_sequence;
}

void GenerateAndSaveRiverSubgamesFromTurnSubgame(
    int n_situations, const std::string &file_name, std::mt19937 &mt, const std::vector<int> &prev_board_cards,
    const std::vector<int> &prev_action_sequence, const std::array<std::vector<double>, 2> &prev_ranges) {
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
  std::shared_ptr<const Game> game = LoadGame(name);
  std::ofstream oss;
  oss.open(file_name);
  oss.flags(std::ios::scientific);
  oss.precision(std::numeric_limits<double>::digits10 + 1);
  for (int situation = 0; situation < n_situations; situation++) {
    std::vector<int> board_cards = GenerateNewBoardCard(prev_board_cards, mt);
    std::array<std::vector<double>, 2> ranges = GenerateRanges(prev_board_cards, board_cards.back(), mt, prev_ranges);
    std::vector<int> action_sequence = GenerateActions(game, mt, prev_action_sequence);

    // Writing to file
    oss << "b ";
    for (int card : board_cards) {
      oss << card << " ";
    }
    oss << "r ";
    for (double probability : ranges[0]) {
      oss << probability << " ";
    }
    for (double probability : ranges[1]) {
      oss << probability << " ";
    }
    oss << "a ";
    oss << action_sequence.size() << " ";
    for (int action : action_sequence) {
      oss << action << " ";
    }
    oss << "\n";
  }
}

template<class bidiiter>
bidiiter RandomUnique(bidiiter begin, bidiiter end, size_t num_random, std::mt19937 &mt) {
  size_t left = std::distance(begin, end);
  while (num_random--) {
    bidiiter r = begin;
    std::uniform_int_distribution<int> distribution(0, left - 1);
    std::advance(r, distribution(mt));
    std::swap(*begin, *r);
    ++begin;
    --left;
  }
  return begin;
}

std::vector<int> GenerateBoardCards(std::mt19937 &mt) {
  std::vector<int> cards(52);
  for (int i = 0; i < 52; ++i)
    cards[i] = i;
  RandomUnique(cards.begin(), cards.end(), 5, mt);
  return std::vector<int>(cards.begin(), cards.begin() + 5);
}

void AssignProbabilities(std::vector<double> &player_range, std::vector<size_t> &indexes, int start,
                         int end, double probability, std::mt19937 &mt) {
  if (end == start) {
    player_range[indexes[start]] = probability;
  } else {
    std::uniform_real_distribution<double> distribution(0., probability);
    double p = distribution(mt);
    int length = end - start + 1;
    int part_one = length / 2;
    AssignProbabilities(player_range, indexes, start, start + part_one - 1, p, mt);
    AssignProbabilities(player_range, indexes, start + part_one, end, probability - p, mt);
  }
}

std::array<std::vector<double>, 2> GenerateRanges(const std::vector<int> &board_cards, std::mt19937 &mt) {
  std::vector<int> full_cards = board_cards;
  full_cards.insert(full_cards.begin(), 1);
  full_cards.insert(full_cards.begin(), 0);
  std::vector<int> hand_strength;
  hand_strength.reserve(1326);
  // Compute strengths and mapping from cards to possible hands
  int impossible_hands = 0;
  for (int card_one = 0; card_one < 52 - 1; card_one++) {
    full_cards[0] = card_one;
    for (int card_two = card_one + 1; card_two < 52; card_two++) {
      if (std::find(board_cards.begin(), board_cards.end(), card_two) != board_cards.end() or
          std::find(board_cards.begin(), board_cards.end(), card_one) != board_cards.end()) {
        hand_strength.push_back(-1);
        impossible_hands++;
      } else {
        full_cards[1] = card_two;
        universal_poker::logic::CardSet cards = universal_poker::logic::CardSet(full_cards);
        hand_strength.push_back(cards.RankCards());
      }
    }
  }
  // Sort cards by strength
  std::vector<size_t> idx(hand_strength.size());
  iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&hand_strength](size_t i1, size_t i2) { return hand_strength[i1] < hand_strength[i2]; });
  std::array<std::vector<double>, 2> ranges;
  ranges = {std::vector<double>(1326, 0.), std::vector<double>(1326, 0.)};
  AssignProbabilities(ranges[0], idx, impossible_hands, 1325, 1, mt);
  AssignProbabilities(ranges[1], idx, impossible_hands, 1325, 1, mt);
  std::array<double, 2> probabilities_back{0, 0};
  for (int i : idx) {
    if (hand_strength[i] == -1) {
      SPIEL_CHECK_EQ(ranges[0][i], 0);
      SPIEL_CHECK_EQ(ranges[1][i], 0);
    } else {
      for (int player = 0; player < 2; player++) {
        probabilities_back[player] += ranges[player][i];
      }
    }
  }
  SPIEL_CHECK_FLOAT_NEAR(probabilities_back[0], 1, 1e-6);
  SPIEL_CHECK_FLOAT_NEAR(probabilities_back[1], 1, 1e-6);
  return ranges;
}

std::vector<int> GenerateActions(const std::shared_ptr<const Game> &game, std::mt19937 &mt) {
  std::vector<int> action_sequence;
  std::unique_ptr<State> state = game->NewInitialState();
  int chance_visited = 0;
  while (chance_visited < 9) {
    if (state->IsChanceNode()) {
      chance_visited++;
      state->ApplyAction(state->LegalActions()[0]);
    } else {
      std::vector<Action> actions = state->LegalActions();
      SPIEL_CHECK_GT(actions.size(), 0);
      absl::uniform_int_distribution<> dis(0, actions.size() - 1);
      Action action = universal_poker::kFold;
      while (action == universal_poker::kFold) {
        action = actions[dis(mt)];
      }
      state->ApplyAction(action);
      action_sequence.push_back(action);
    }
  }
  return action_sequence;
}

void GenerateAndSavePokerSituations(int n_situations, const std::string &file_name, std::mt19937 &mt) {
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
  std::shared_ptr<const Game> game = LoadGame(name);
  std::ofstream oss;
  oss.open(file_name);
  oss.flags(std::ios::scientific);
  oss.precision(std::numeric_limits<double>::digits10 + 1);
  for (int situation = 0; situation < n_situations; situation++) {
    // Generation
    std::vector<int> board_cards = GenerateBoardCards(mt);
    std::array<std::vector<double>, 2> ranges = GenerateRanges(board_cards, mt);
    std::vector<int> action_sequence = GenerateActions(game, mt);

    // Writing to file
    oss << "b ";
    for (int card : board_cards) {
      oss << card << " ";
    }
    oss << "r ";
    for (double probability : ranges[0]) {
      oss << probability << " ";
    }
    for (double probability : ranges[1]) {
      oss << probability << " ";
    }
    oss << "a ";
    oss << action_sequence.size() << " ";
    for (int action : action_sequence) {
      oss << action << " ";
    }
    oss << "\n";
  }
}

std::vector<int> ReadBoardCards(std::ifstream &file) {
  std::string read_label;
  std::vector<int> board_cards(5, 0);
  file >> read_label;
  SPIEL_CHECK_EQ(read_label, "b");
  for (int i = 0; i < 5; i++) {
    file >> board_cards[i];
  }
  return board_cards;
}

std::array<std::vector<double>, 2> ReadCFVs(std::ifstream &file) {
  std::array<std::vector<double>, 2> cfvs = {std::vector<double>(1326, 0.), std::vector<double>(1326, 0.)};
  std::string read_label;
  file >> read_label;
  SPIEL_CHECK_EQ(read_label, "v");
  for (int player = 0; player < 2; player++) {
    for (int i = 0; i < 1326; i++) {
      file >> cfvs[player][i];
    }
  }
  return cfvs;
}

std::array<std::vector<double>, 2> ReadRanges(std::ifstream &file) {
  std::array<std::vector<double>, 2> ranges = {std::vector<double>(1326, 0.), std::vector<double>(1326, 0.)};
  std::string read_label;
  file >> read_label;
  SPIEL_CHECK_EQ(read_label, "r");
  for (int player = 0; player < 2; player++) {
    for (int i = 0; i < 1326; i++) {
      file >> ranges[player][i];
    }
  }
  return ranges;
}

std::vector<int> ReadActions(std::ifstream &file) {
  std::string read_label;
  file >> read_label;
  SPIEL_CHECK_EQ(read_label, "a");
  int num_actions;
  file >> num_actions;
  std::vector<int> action_sequence(num_actions, 0);
  for (int action = 0; action < num_actions; action++) {
    file >> action_sequence[action];
  }
  return action_sequence;
}

std::array<std::vector<double>, 2> SolveLimitPokerSituationFromInputs(const std::vector<int> &board_cards,
                                                                      const std::array<std::vector<double>, 2> &ranges,
                                                                      const std::vector<int> &action_sequence,
                                                                      int iterations) {
  std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                     "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
  std::shared_ptr<const Game> game = LoadGame(name);

  std::unique_ptr<State> state = game->NewInitialState();

  std::vector<int> initial_cards;
  int card = 0;
  while (initial_cards.size() < 4) {
    if (std::find(board_cards.begin(), board_cards.end(), card) == board_cards.end()) {
      initial_cards.push_back(card);
    }
    card++;
  }

  // Deal 4 cards
  state->ApplyAction(initial_cards[0]);
  state->ApplyAction(initial_cards[1]);
  state->ApplyAction(initial_cards[2]);
  state->ApplyAction(initial_cards[3]);

  // Pre-flop betting
  int action_index = 0;
  while (state->IsPlayerNode()) {
    state->ApplyAction(action_sequence[action_index]);
    action_index++;
  }

  // Deal 3 board cards (Flop)
  state->ApplyAction(board_cards[0]);
  state->ApplyAction(board_cards[1]);
  state->ApplyAction(board_cards[2]);

  // Flop betting
  while (state->IsPlayerNode()) {
    state->ApplyAction(action_sequence[action_index]);
    action_index++;
  }

  // Deal board card (Turn)
  state->ApplyAction(board_cards[3]);

  // Turn betting
  while (state->IsPlayerNode()) {
    state->ApplyAction(action_sequence[action_index]);
    action_index++;
  }

  // Deal board card (River)
  state->ApplyAction(board_cards[4]);

  std::shared_ptr<Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

  std::vector<double> chance_reaches(1326, 1. / 1326);

  algorithms::PokerData poker_data = algorithms::PokerData(*state);

  UpdateChanceReaches(chance_reaches, poker_data, board_cards);

  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakePokerInfostateTrees(state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage);

  auto out = std::make_shared<Subgame>(game, public_observer, trees);

  std::shared_ptr<const PublicStateEvaluator>
      terminal_evaluator = std::make_shared<const PokerTerminalEvaluatorLinear>(poker_data, board_cards);

  SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                       std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

  solver.RunSimultaneousIterations(iterations);

  PublicState root_public_state = out->public_states[0];
  SPIEL_CHECK_TRUE(root_public_state.IsInitial());

  return root_public_state.values;
}

void SolvePokerSubgames(const std::string &file_in, const std::string &file_out, int num_games, int iterations) {
  // Create in and out filestreams
  std::ifstream read_file(file_in);
  std::ofstream oss;
  oss.open(file_out);
  oss.flags(std::ios::scientific);
  oss.precision(std::numeric_limits<double>::digits10 + 1);
  // Iterate over different situations
  for (int game_index = 0; game_index < num_games; game_index++) {
    // Load the situation from the input file
    std::vector<int> board_cards = ReadBoardCards(read_file);
    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file);
    std::vector<int> action_sequence = ReadActions(read_file);

    // Solve the situation
    std::array<std::vector<double>, 2>
        cf_values = SolveLimitPokerSituationFromInputs(board_cards, ranges, action_sequence, iterations);

    // Write the solution to the output file     // Writing to file
    oss << "b ";
    for (int card : board_cards) {
      oss << card << " ";
    }
    oss << "r ";
    for (double probability : ranges[0]) {
      oss << probability << " ";
    }
    for (double probability : ranges[1]) {
      oss << probability << " ";
    }
    oss << "a ";
    oss << action_sequence.size() << " ";
    for (int action : action_sequence) {
      oss << action << " ";
    }

    oss << "v ";
    for (double probability : cf_values[0]) {
      oss << probability << " ";
    }
    for (double probability : cf_values[1]) {
      oss << probability << " ";
    }
    oss << "\n";
  }
}

void NetworkTraining(const std::string &file_name,
                     int training_samples,
                     int validation_samples,
                     int epochs,
                     int batch_size) {

  std::cout << training_samples << " " << validation_samples << " " << epochs << " " << batch_size << "\n";

  auto net = std::make_shared<Net>();

  torch::Device device("cpu");

  net->to(device);

  std::ifstream read_file(file_name);

  torch::Tensor training_data_tensor = torch::zeros({training_samples, 2704});
  torch::Tensor training_target_tensor = torch::zeros({training_samples, 2652});
  for (int i = 0; i < training_samples; i++) {
    std::vector<int> board_cards = ReadBoardCards(read_file);
    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file);
    std::vector<int> action_sequence = ReadActions(read_file);
    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file);

    training_data_tensor[i][board_cards[4]] = 1;
    for (int card_index = 0; card_index < 1326; card_index++) {
      training_data_tensor[i][card_index + 52] = ranges[0][card_index];
      training_data_tensor[i][card_index + 1326 + 52] = ranges[1][card_index];

      training_target_tensor[i][card_index] = cfvs[0][card_index];
      training_target_tensor[i][card_index + 1326] = cfvs[1][card_index];
    }
  }

  torch::Tensor validation_data_tensor = torch::zeros({validation_samples, 2704});
  torch::Tensor validation_target_tensor = torch::zeros({validation_samples, 2652});
  for (int i = 0; i < validation_samples; i++) {
    std::vector<int> board_cards = ReadBoardCards(read_file);
    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file);
    std::vector<int> action_sequence = ReadActions(read_file);
    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file);

    training_data_tensor[i][board_cards[4]] = 1;
    for (int card_index = 0; card_index < 1326; card_index++) {
      validation_data_tensor[i][card_index + 52] = ranges[0][card_index];
      validation_data_tensor[i][card_index + 1326 + 52] = ranges[1][card_index];

      validation_target_tensor[i][card_index] = cfvs[0][card_index];
      validation_target_tensor[i][card_index + 1326] = cfvs[1][card_index];
    }
  }

  auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(), torch::optim::AdamOptions(1e-3));
  int batches = training_samples / batch_size;

  std::cout << "Initial:   ";

  torch::Tensor init_train_output = net->forward(training_data_tensor);
  torch::Tensor init_train_loss = torch::mse_loss(init_train_output, training_target_tensor);
  std::cout << "Training loss: " << init_train_loss.item().to<double>() << "   ";

  torch::Tensor init_output = net->forward(validation_data_tensor);
  torch::Tensor init_loss = torch::mse_loss(init_output, validation_target_tensor);
  std::cout << "Validation loss: " << init_loss.item().to<double>() << "\n";

  for (int epoch = 0; epoch < epochs; epoch++) {
    std::cout << "Epoch " << epoch << ":   ";
    double cumulative_loss = 0;
    for (int batch = 0; batch < batches; batch++) {
      int i_s = batch * batch_size;
      int i_e = (batch + 1) * batch_size - 1;
      optimizer->zero_grad();
      torch::Tensor output = net->forward(training_data_tensor[i_s, i_e]);
      torch::Tensor loss = torch::mse_loss(output, training_target_tensor[i_s, i_e]);
      loss.backward();
      optimizer->step();
      cumulative_loss += loss.item().to<double>();
      std::cout << "." << std::flush;
    }
    std::cout << "\nTraining loss: " << cumulative_loss / batches << "   ";
    torch::Tensor output = net->forward(validation_data_tensor);
    torch::Tensor loss = torch::mse_loss(output, validation_target_tensor);
    std::cout << "Validation loss: " << loss.item().to<double>() << "\n";
  }
}

}
}
}

std::array<std::vector<double>, 2> GetReachesFromVector(const std::vector<double> &range_vector) {
  std::array<std::vector<double>, 2> ranges = {std::vector<double>(1326, 0.), std::vector<double>(1326, 0.)};

  for (int player = 0; player < 2; player++) {
    for (int i = 0; i < 1326; i++) {
      ranges[player][i] = range_vector[i + player * 1326];
    }
  }
  return ranges;
}

void ConvertRangesFromDescendingSuitToAscendingSuit() {
  std::string ranks = "23456789TJQKA";
  std::string suits_des = "shdc";
  std::string suits_asc = "cdhs";
  std::vector<std::string> cards_asc(52);
  std::vector<std::string> cards_des(52);
  int card_index = 0;
  for (int card_rank = 0; card_rank < 13; card_rank++) {
    for (int card_suit = 0; card_suit < 4; card_suit++) {
      cards_asc[card_index] = ranks.substr(card_rank, 1) + suits_asc.substr(card_suit, 1);
      cards_des[card_index] = ranks.substr(card_rank, 1) + suits_des.substr(card_suit, 1);
      card_index++;
    }
  }
  std::unordered_map<std::string, int> hands_to_index_asc;
  std::vector<std::string> index_to_hand_des(1326);
  int hand_index = 0;
  for (int card_one = 0; card_one < 51; card_one++) {
    for (int card_two = card_one + 1; card_two < 52; card_two++) {
      if (card_one / 4 == card_two / 4) {
        index_to_hand_des[hand_index] = cards_des[card_two] + cards_des[card_one];
      } else {
        index_to_hand_des[hand_index] = cards_des[card_one] + cards_des[card_two];
      }
      hands_to_index_asc[cards_asc[card_one] + cards_asc[card_two]] = hand_index;
      hand_index++;
    }
  }
  std::cout << index_to_hand_des << "\n";
  std::vector<double> transformed_ranges(2652, -1);
  std::stringstream range_stream(SUBGAME_TWO_STRING);
  for (int i = 0; i < 1326; i++) {
    range_stream >> transformed_ranges[hands_to_index_asc[index_to_hand_des[i]]];
  }
  for (int i = 0; i < 1326; i++) {
    range_stream >> transformed_ranges[hands_to_index_asc[index_to_hand_des[i]] + 1326];
  }
  std::cout << std::setprecision(16);
  for (float f : transformed_ranges) {
    std::cout << f << ", ";
  }
}

int main(int argc, char **argv) {
  std::string file_template = argv[1];
  int training_samples = std::atoi(argv[2]);
  int validation_samples = std::atoi(argv[3]);
  int epochs = std::atoi(argv[4]);
  int batch_size = std::atoi(argv[5]);
  open_spiel::papers_with_code::NetworkTraining(
      file_template, training_samples, validation_samples, epochs, batch_size);
//  if (argc > 2) {
//    int iterations = 1000;
//    int situations = 1;
//    int machines = 1;
//    // Linear evaluator infostate CFR
//    if (std::strcmp(argv[1], "-gen") == 0) {
//      std::string file_in = argv[2];
//      std::vector<int> board_cards = {23, 28, 30, 32};
//      std::vector<int> action_sequence = {1, 1, 2, 2, 1};
//      std::array<std::vector<double>, 2> ranges = GetReachesFromVector(SUBGAME_ONE_RANGES);
//      std::random_device rd;
//      std::mt19937 mt(rd());
//      for (int i = 0; i < machines; i++) {
//        open_spiel::papers_with_code::GenerateAndSaveRiverSubgamesFromTurnSubgame(
//            situations / machines, file_in + std::to_string(i), mt, board_cards, action_sequence, ranges);
//      }
//    }
//
//    if (std::strcmp(argv[1], "-sol") == 0) {
//      std::string file_in = argv[2];
//      std::string file_out = argv[3];
//      open_spiel::papers_with_code::SolvePokerSubgames(file_in, file_out, situations / machines, iterations);
//    }
//  } else {
//    std::cout
//        << "Please specify the experiment to run. -gen + filename to generate data and -sol + file_in + file_out to solve the situations";
//  }
//  if (argc > 2) {
//    int iterations = 1000;
//    int situations = 100000;
//    int machines = 50;
//    // Linear evaluator infostate CFR
//    if (std::strcmp(argv[1], "-gen") == 0) {
//      std::string file_in = argv[2];
//      std::random_device rd;
//      std::mt19937 mt(rd());
//      for (int i = 0; i < machines; i++) {
//        open_spiel::papers_with_code::GenerateAndSavePokerSituations(
//            situations / machines, file_in + std::to_string(i), mt);
//      }
//    }
//
//    if (std::strcmp(argv[1], "-sol") == 0) {
//      std::string file_in = argv[2];
//      std::string file_out = argv[3];
//      open_spiel::papers_with_code::SolvePokerSubgames(file_in, file_out, situations / machines, iterations);
//    }
//  } else {
//    std::cout
//        << "Please specify the experiment to run. -gen + filename to generate data and -sol + file_in + file_out to solve the situations";
//  }
//  open_spiel::papers_with_code::GenerateAndSavePokerSituations(poker_situations, file_name);

//  open_spiel::papers_with_code::CheckExploitability();
//  open_spiel::papers_with_code::CheckExploitability(iterations);
  // Leduc tests
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFREfgHashed);
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFREfg);
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFRInfostate);

//   Limit tests
//  if (argc > 1) {
//    int iterations = 1000;
//    int runs = 1;
//    int n_arguments = argc;
//    // Linear evaluator infostate CFR
//    bool do_next_part = false;
//    for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
//      if (std::strcmp(argv[argument_index], "-isl") == 0 or strcmp(argv[argument_index], "-all") == 0) {
//        do_next_part = true;
//        break;
//      }
//    }
//    if (do_next_part) {
//      std::cout << "Infostate CFR experiment with Linear evaluator:\n";
//      open_spiel::papers_with_code::MeasureTime(
//          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFRPokerSpecificLinear);
//    }
//    // Quadratic evaluator infostate CFR
//    do_next_part = false;
//    for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
//      if (std::strcmp(argv[argument_index], "-isq") == 0 or strcmp(argv[argument_index], "-all") == 0) {
//        do_next_part = true;
//        break;
//      }
//    }
//    if (do_next_part) {
//      std::cout << "Infostate CFR experiment with Quadratic evaluator:\n";
//      open_spiel::papers_with_code::MeasureTime(
//          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFRPokerSpecificQuadratic);
//    }
//    // Efg CFR without saving structure
//    do_next_part = false;
//    for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
//      if (std::strcmp(argv[argument_index], "-efg") == 0 or strcmp(argv[argument_index], "-all") == 0) {
//        do_next_part = true;
//        break;
//      }
//    }
//    if (do_next_part) {
//      std::cout << "EFG CFR experiment:\n";
//      open_spiel::papers_with_code::MeasureTime(
//          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfg);
//    }
//    // Efg CFR with saving the structure
//    do_next_part = false;
//    for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
//      if (std::strcmp(argv[argument_index], "-efgh") == 0 or strcmp(argv[argument_index], "-all") == 0) {
//        do_next_part = true;
//        break;
//      }
//    }
//    if (do_next_part) {
//      std::cout << "Hashed EFG CFR experiment:\n";
//      open_spiel::papers_with_code::MeasureTime(
//          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfgHashed);
//    }
//  } else {
//    std::cout
//        << "Please specify the experiment to run. -isl for infostate CFR with linear evaluator, -efg for EFG CFR, "
//           "-efgh for EFG CFR where the tree is build and saved, -isq for infostate CFR with quadratic evaluator and "
//           "-all for all experiments";
//  }

//  open_spiel::papers_with_code::SmallUniversalPokerTrunkTest();
//  open_spiel::papers_with_code::TestSameInfostates();
}