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
#include "open_spiel/algorithms/infostate_tree.h"
#include "subgame.h"
#include "algorithms/cfr.h"

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

std::pair<int, int> UniversalPokerRiverCFRPokerSpecific(int iterations) {
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

}
}
}

int main(int argc, char **argv) {
  int iterations = 1;
  int runs = 1;

  // Leduc tests
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFREfgHashed);
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFREfg);
//  open_spiel::papers_with_code::MeasureTime(runs, iterations, open_spiel::papers_with_code::SolverLeducCFRInfostate);

  // Limit tests
  if (argc > 1) {
    if (std::strcmp(argv[1], "-is") == 0 or strcmp(argv[1], "-all") == 0) {
      std::cout << "Infostate CFR experiment:\n";
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFRPokerSpecific);
    }
    if (std::strcmp(argv[1], "-efg") == 0 or strcmp(argv[1], "-all") == 0) {
      std::cout << "EFG CFR experiment:\n";
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfg);
    }

    if (std::strcmp(argv[1], "-efgh") == 0 or strcmp(argv[1], "-all") == 0) {
      std::cout << "Hashed EFG CFR experiment:\n";
      open_spiel::papers_with_code::MeasureTime(
          runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfgHashed);
    }
  } else {
    std::cout
        << "Please specify the experiment to run. -is for infostate CFR, -efg for EFG CFR, -efgh for hashed EFG CFR and -all for all experiments";
  }

//  open_spiel::papers_with_code::SmallUniversalPokerTrunkTest();
//  open_spiel::papers_with_code::TestSameInfostates();
}