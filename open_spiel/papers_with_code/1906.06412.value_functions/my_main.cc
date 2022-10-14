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
#include "algorithms/ortools/sequence_form_lp.h"
#include "open_spiel/algorithms/poker_data.h"
#include "algorithms/expected_returns.h"

#include <iostream>

std::chrono::high_resolution_clock::time_point GetTime() {
    return std::chrono::high_resolution_clock::now();
}

int ElapsedTime(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    return setup_duration.count();
}

void LogLine(const std::string &text, const std::string &suffix) {
    std::ofstream logfile;
    logfile.open("log_file_" + suffix, std::ios_base::app);
    logfile << text << "\n";
    logfile.close();
}

void Log(const std::string &text, const std::string &suffix) {
    std::ofstream logfile;
    logfile.open("log_file_" + suffix, std::ios_base::app);
    logfile << text;
    logfile.close();
}

void ClearLog(const std::string &suffix) {
    std::ofstream logfile;
    logfile.open("log_file_" + suffix);
    logfile.close();
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
    std::vector <std::string> cards_asc(52);
    std::vector <std::string> cards_des(52);
    int card_index = 0;
    for (int card_rank = 0; card_rank < 13; card_rank++) {
        for (int card_suit = 0; card_suit < 4; card_suit++) {
            cards_asc[card_index] = ranks.substr(card_rank, 1) + suits_asc.substr(card_suit, 1);
            cards_des[card_index] = ranks.substr(card_rank, 1) + suits_des.substr(card_suit, 1);
            card_index++;
        }
    }
    std::unordered_map<std::string, int> hands_to_index_asc;
    std::vector <std::string> index_to_hand_des(1326);
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

namespace open_spiel {
    namespace papers_with_code {
        namespace {

            std::unique_ptr <State> GetPokerStatesAfterMoves(const std::shared_ptr<const Game> &game,
                                                             std::vector<int> board_cards,
                                                             std::vector<int> action_sequence,
                                                             int cards_in_hand) {
                std::unique_ptr <State> state = game->NewInitialState();

                std::vector<int> initial_cards;
                int card = 0;
                while (initial_cards.size() < 2 * cards_in_hand) {
                    if (std::find(board_cards.begin(), board_cards.end(), card) == board_cards.end()) {
                        initial_cards.push_back(card);
                    }
                    card++;
                }

                // Deal Initial cards
                for (int initial_card : initial_cards) {
                    state->ApplyAction(initial_card);
                }

                // First round
                int action_index = 0;
                while (state->IsPlayerNode()) {
                    if (action_sequence.size() == action_index) {
                        return state;
                    }
                    state->ApplyAction(action_sequence[action_index]);
                    action_index++;
                }

                // Deal 3 board cards (Flop)
                if (board_cards.empty()) {
                    return state;
                }
                state->ApplyAction(board_cards[0]);
                if (board_cards.size() == 1) {
                    return state;
                }
                state->ApplyAction(board_cards[1]);
                if (board_cards.size() == 2) {
                    return state;
                }
                state->ApplyAction(board_cards[2]);

                // Other rounds
                int board_card_index = 3;
                while (true) {
                    while (state->IsPlayerNode()) {
                        if (action_sequence.size() == action_index) {
                            return state;
                        }
                        state->ApplyAction(action_sequence[action_index]);
                        action_index++;
                    }
                    if (board_cards.size() == board_card_index) {
                        return state;
                    }
                    state->ApplyAction(board_cards[board_card_index]);
                    board_card_index++;
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

                std::unique_ptr <State> state = game->NewInitialState();

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
                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                std::vector <std::shared_ptr<algorithms::InfostateTree>>
                        trees = algorithms::MakePokerInfostateTrees(state, std::vector<double>(1326, 1. / 1326),
                                                                    infostate_observer, 1000,
                                                                    kDlCfrInfostateTreeStorage, cards);

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

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

                std::unique_ptr <State> state = game->NewInitialState();

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
                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                std::vector<double> chance_reaches(1326, 1. / 1326);

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                UpdateChanceReaches(chance_reaches, poker_data, cards);

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees =
                        algorithms::MakePokerInfostateTrees(state,
                                                            chance_reaches,
                                                            infostate_observer,
                                                            1000,
                                                            kDlCfrInfostateTreeStorage,
                                                            cards);

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

                std::unique_ptr <State> state = game->NewInitialState();

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
                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                std::vector<double> chance_reaches(1326, 1. / 1326);

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                UpdateChanceReaches(chance_reaches, poker_data, cards);

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees =
                        algorithms::MakePokerInfostateTrees(state,
                                                            chance_reaches,
                                                            infostate_observer,
                                                            1000,
                                                            kDlCfrInfostateTreeStorage,
                                                            cards);

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

                auto fixed_policy = solver.AveragePolicy();

                std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(), TabularPolicy()};

                for (int player = 0; player < 2; player++) {
                    algorithms::BanditVector &bandits = solver.bandits()[player];
                    for (algorithms::DecisionId id : bandits.range()) {
                        algorithms::InfostateNode *node = solver.subgame()->trees[player]->decision_infostate(id);
                        const std::string &infostate = node->infostate_string();
                        ActionsAndProbs infostate_policy = fixed_policy->GetStatePolicy(infostate);
                        separated_policies[player].SetStatePolicy(infostate, infostate_policy);
                    }
                }

                double nash_conv = 0;

                for (int player = 0; player < 2; player++) {
                    SubgameSolver best_response = SubgameSolver(out, nullptr, terminal_evaluator,
                                                                std::make_shared<std::mt19937>(0),
                                                                "RegretMatchingPlus");
                    best_response.bandits() = MakeResponseBandits(trees, separated_policies[1 - player]);
                    best_response.RunSimultaneousIterations(1);
                    std::cout << best_response.RootValues() << "\n";
                    nash_conv += best_response.RootValues()[player];
                }
                std::cout << "Exploitability: " << nash_conv / 2 << "\n";
                return std::pair<int, int>(setup_duration.count(), run_duration.count());
            }

            std::pair<int, int> UniversalPokerTurnCFRPokerSpecificLinear(int iterations) {
                std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                   "firstPlayer=2 1,numSuits=2,numRanks=5,numHoleCards=2,numBoardCards=0 3 "
                                   "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
//  std::string name = "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 50,"
//                     "firstPlayer=2 1 1 "
//                     "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
//                     "1 1,stack=20000 20000,bettingAbstraction=fcpa)";
                std::shared_ptr<const Game> game = LoadGame(name);

                std::vector<int> board_cards = {0, 2, 7, 9};
                std::vector<int> action_sequence = {1, 1, 1, 1};

                std::unique_ptr <State> state = GetPokerStatesAfterMoves(game, board_cards, action_sequence, 2);

                auto start = std::chrono::high_resolution_clock::now();
                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                std::vector<double> chance_reaches(45, 1. / 45);

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                UpdateChanceReaches(chance_reaches, poker_data, board_cards);

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
                        state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage, board_cards);

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

                SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                                     std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
                auto end = std::chrono::high_resolution_clock::now();
                auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                start = std::chrono::high_resolution_clock::now();
                solver.RunSimultaneousIterations(iterations);
                end = std::chrono::high_resolution_clock::now();
                auto run_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                auto fixed_policy = solver.AveragePolicy();

                std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(), TabularPolicy()};

                for (int player = 0; player < 2; player++) {
                    algorithms::BanditVector &bandits = solver.bandits()[player];
                    for (algorithms::DecisionId id : bandits.range()) {
                        algorithms::InfostateNode *node = solver.subgame()->trees[player]->decision_infostate(id);
                        const std::string &infostate = node->infostate_string();
                        ActionsAndProbs infostate_policy = fixed_policy->GetStatePolicy(infostate);
                        separated_policies[player].SetStatePolicy(infostate, infostate_policy);
                    }
                }

                double nash_conv = 0;

                for (int player = 0; player < 2; player++) {
                    SubgameSolver best_response = SubgameSolver(out, nullptr, terminal_evaluator,
                                                                std::make_shared<std::mt19937>(0),
                                                                "RegretMatchingPlus");
                    best_response.bandits() = MakeResponseBandits(trees, separated_policies[1 - player]);
                    best_response.RunSimultaneousIterations(2);
                    std::cout << best_response.RootValues() << "\n";
                    nash_conv += best_response.RootValues()[player];
                }
                std::cout << "Exploitability: " << nash_conv / 2 << "\n";
                return std::pair<int, int>(setup_duration.count(), run_duration.count());
            }

            void SmallUniversalPokerTrunkTest() {
                int num_suits = 2;
                int num_ranks = 6;
                std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                   "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks=" +
                                   std::to_string(num_ranks)
                                   + ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                std::shared_ptr<const Game> game = LoadGame(name);

                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});

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
                std::unique_ptr <State> state = game->NewInitialState();
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

                std::vector <std::shared_ptr<algorithms::InfostateTree>>
                        tree_poker = algorithms::MakePokerInfostateTrees(
                        state, std::vector<double>(66, 1. / 66),
                        infostate_observer, 1000, kDlCfrInfostateTreeStorage, {4, 5, 6, 7, 8});

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
                std::vector <std::pair<int, int>> test_inputs = {{2, 6}};
                for (std::pair<int, int> game_spec : test_inputs) {
                    int num_suits = game_spec.first;
                    int num_ranks = game_spec.second;
                    SPIEL_CHECK_LE(num_suits, 4);
                    SPIEL_CHECK_LE(num_ranks, 13);
                    std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                       "firstPlayer=2 1,numSuits=" + std::to_string(num_suits) + ",numRanks="
                                       + std::to_string(num_ranks)
                                       +
                                       ",numHoleCards=2,numBoardCards=0 3 1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";

                    std::shared_ptr<const Game> game = LoadGame(name);

                    std::unique_ptr <State> state = game->NewInitialState();

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

                    std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                    std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                    std::vector <std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
                            state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage, board_cards);

                    auto poker_specific_subgame = std::make_shared<Subgame>(game, public_observer, trees);

                    // General subgame construction
                    std::vector <std::unique_ptr<State>> starting_states;
                    std::unique_ptr <State> initial_state = game->NewInitialState();

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
                                if (std::find(board_cards.begin(), board_cards.end(), action_two) !=
                                    board_cards.end()) {
                                    continue;
                                }
                                auto child_three = child_two->Child(action_two);
                                for (Action action_three : child_three->LegalActions()) {
                                    if (std::find(board_cards.begin(), board_cards.end(), action_three) !=
                                        board_cards.end()) {
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

                    std::vector <std::shared_ptr<algorithms::InfostateTree>>
                            trees_general = algorithms::MakeInfostateTrees(
                            starting_states, general_chance_reaches, infostate_observer, 1000,
                            kDlCfrInfostateTreeStorage);

                    auto general_subgame = std::make_shared<Subgame>(game, public_observer, trees_general);

                    SPIEL_CHECK_EQ(general_subgame->public_states.size(), poker_specific_subgame->public_states.size());
                    for (int i = 0; i < general_subgame->public_states.size(); i++) {
                        auto general_public_state = &general_subgame->public_states[i];
                        if (!general_public_state->IsTerminal()) {
                            continue;
                        }
                        auto poker_public_state = &poker_specific_subgame->public_states[i];
                        std::vector <std::set<std::string>> general_infostates(2, std::set<std::string>());
                        std::vector <std::set<std::string>> poker_infostates(2, std::set<std::string>());
                        SPIEL_CHECK_EQ(general_public_state->nodes[0].size(), general_public_state->nodes[1].size());
                        for (int node_index = 0; node_index < general_public_state->nodes[0].size(); node_index++) {
                            for (Player player = 0; player < 2; player++) {
                                general_infostates[player].insert(
                                        general_public_state->nodes[player][node_index]->infostate_string());
                            }
                        }
                        for (int node_index = 0; node_index < poker_public_state->nodes[0].size(); node_index++) {
                            for (Player player = 0; player < 2; player++) {
                                if (poker_public_state->nodes[player][node_index]->terminal_chance_reach_prob() > 0) {
                                    poker_infostates[player].insert(
                                            poker_public_state->nodes[player][node_index]->infostate_string());
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
                std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(rng, pot_size, board_cards,
                                                                                      uniform_reaches);

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
                std::shared_ptr<const Game> game = universal_poker::MakeRandomSubgame(rng, pot_size, board_cards,
                                                                                      uniform_reaches);

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
                std::vector <std::pair<int, int>> collected_times;
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
                std::cout << "Normal efg cfr exploitability: " << algorithms::Exploitability(*game, *policy_efg)
                          << "\n";

                algorithms::InfostateCFR solver_is(*game);

                solver_is.RunSimultaneousIterations(iterations);
                auto policy_is = solver_is.AveragePolicy();
                algorithms::TabularBestResponse best_response_pone(*game, 0, &*policy_is);
                std::cout << best_response_pone.Value(*game->NewInitialState()) << "\n";
                algorithms::TabularBestResponse best_response_ptwo(*game, 1, &*policy_is);
                std::cout << best_response_ptwo.Value(*game->NewInitialState()) << "\n";
                std::cout << BestResponse(solver_is.trees(), *policy_is) << "\n";

                double nash_conv = 0.;

                std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(), TabularPolicy()};

                for (int player = 0; player < 2; player++) {
                    algorithms::InfostateCFR best_response_is = algorithms::InfostateCFR(*game);
                    int opponent = 1 - player;
                    algorithms::BanditVector &opponent_bandits = best_response_is.bandits_modifiable()[opponent];
                    for (algorithms::DecisionId id : opponent_bandits.range()) {
                        algorithms::InfostateNode *node = best_response_is.trees()[opponent]->decision_infostate(id);
                        const std::string &infostate = node->infostate_string();
                        ActionsAndProbs infostate_policy = policy_is->GetStatePolicy(infostate);
                        separated_policies[player].SetStatePolicy(infostate, infostate_policy);
                    }
                }

                for (int player = 0; player < 2; player++) {
                    algorithms::InfostateCFR best_response_is = algorithms::InfostateCFR(*game);
                    best_response_is.bandits_modifiable() =
                            MakeResponseBandits(algorithms::MakeInfostateTrees(*game), separated_policies[player]);
                    best_response_is.RunSimultaneousIterations(1);
                    std::cout << best_response_is.RootValues() << "\n";
                }
            }

            void CheckExploitability(int iterations) {
                std::string name = "leduc_poker";

                std::shared_ptr<const Game> game = LoadGame(name);

                algorithms::CFRSolverBase solver_efgsave(*game, false, true, true, true);

                for (int i = 0; i < iterations; i++) {
                    solver_efgsave.EvaluateAndUpdatePolicy();
                }

                auto policy_efgsave = solver_efgsave.AveragePolicy();
                std::cout << "Save states efg exploitability: " << algorithms::Exploitability(*game, *policy_efgsave)
                          << "\n";
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

            std::array<std::vector<double>, 2> GenerateRangesContinuationFullPoker(
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
                                std::uniform_real_distribution<double> distribution(0.,
                                                                                    new_ranges[player][range_index]);
                                new_ranges[player][range_index] = distribution(mt);
                            }
                        }
                        range_index++;
                    }
                }
                return new_ranges;
            }

            std::vector<int> GenerateActionsContinuationFullPoker(
                    const std::shared_ptr<const Game> &game, std::mt19937 &mt, const std::vector<int> &actions_so_far) {
                std::vector<int> action_sequence = actions_so_far;
                std::unique_ptr <State> state = game->NewInitialState();
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
                            std::vector <Action> actions = state->LegalActions();
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
                    int n_situations, const std::string &file_name, std::mt19937 &mt,
                    const std::vector<int> &prev_board_cards,
                    const std::vector<int> &prev_action_sequence,
                    const std::array<std::vector<double>, 2> &prev_ranges) {
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
                    std::array<std::vector<double>, 2> ranges =
                            GenerateRangesContinuationFullPoker(prev_board_cards, board_cards.back(), mt, prev_ranges);
                    std::vector<int> action_sequence = GenerateActionsContinuationFullPoker(game, mt,
                                                                                            prev_action_sequence);

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

            std::vector<int> GenerateBoardCardsGeneral(std::mt19937 &mt, int cards_in_deck, int cards_to_generate) {
                std::vector<int> cards(cards_in_deck);
                for (int i = 0; i < cards_in_deck; ++i)
                    cards[i] = i;
                RandomUnique(cards.begin(), cards.end(), cards_to_generate, mt);
                return std::vector<int>(cards.begin(), cards.begin() + cards_to_generate);
            }

            std::vector<int> GenerateBoardCardsFullRiver(std::mt19937 &mt) {
                return GenerateBoardCardsGeneral(mt, 52, 5);
            }

            void AssignProbabilities(std::vector<double> &player_range, std::vector <size_t> &indexes, int start,
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

            std::array<std::vector<double>, 2> GenerateRangesGeneral(
                    const std::vector<int> &board_cards, std::mt19937 &mt, algorithms::PokerData poker_data) {
                std::vector<int> full_cards = board_cards;
                for (int i = 0; i < poker_data.cards_in_hand_; i++) {
                    full_cards.insert(full_cards.begin(), 0);
                }
                std::vector<int> hand_strength;
                hand_strength.reserve(poker_data.num_hands_);
                // Compute strengths and mapping from cards to possible hands
                int impossible_hands = 0;
                for (int hand_index = 0; hand_index < poker_data.num_hands_; hand_index++) {
                    bool card_intersecting = false;
                    for (int card : poker_data.hand_to_cards_[hand_index]) {
                        if (std::find(board_cards.begin(), board_cards.end(), card) != board_cards.end()) {
                            card_intersecting = true;
                        }
                    }
                    if (card_intersecting) {
                        hand_strength.push_back(-1);
                        impossible_hands++;
                    } else {
                        for (int i = 0; i < poker_data.cards_in_hand_; i++) {
                            full_cards[i] = ConvertToFullPokerCard(poker_data.hand_to_cards_[hand_index][i],
                                                                   poker_data);
                        }
                        universal_poker::logic::CardSet cards = universal_poker::logic::CardSet(full_cards);
                        hand_strength.push_back(cards.RankCards());
                    }
                }

                // Sort cards by strength
                std::vector <size_t> idx(hand_strength.size());
                iota(idx.begin(), idx.end(), 0);
                std::stable_sort(idx.begin(), idx.end(),
                                 [&hand_strength](size_t i1, size_t i2) {
                                     return hand_strength[i1] < hand_strength[i2];
                                 });
                std::array<std::vector<double>, 2> ranges;
                ranges = {std::vector<double>(poker_data.num_hands_, 0.),
                          std::vector<double>(poker_data.num_hands_, 0.)};
                std::uniform_real_distribution<double> distribution(0., 1);
                double p = distribution(mt);
                AssignProbabilities(ranges[0], idx, impossible_hands, poker_data.num_hands_ - 1, (p < 0.1 ? 0 : 1), mt);
                p = distribution(mt);
                AssignProbabilities(ranges[1], idx, impossible_hands, poker_data.num_hands_ - 1, (p < 0.1 ? 0 : 1), mt);
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
                return ranges;
            }

            std::vector<int>
            GenerateActionsGeneral(const std::shared_ptr<const Game> &game, std::mt19937 &mt, int chance_nodes) {
                std::vector<int> action_sequence;
                std::unique_ptr <State> state = game->NewInitialState();
                int chance_visited = 0;
                while (chance_visited < chance_nodes) {
                    if (state->IsChanceNode()) {
                        chance_visited++;
                        state->ApplyAction(state->LegalActions()[0]);
                    } else {
                        std::vector <Action> actions = state->LegalActions();
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

            std::vector<int> GenerateActionsFullRiver(const std::shared_ptr<const Game> &game, std::mt19937 &mt) {
                return GenerateActionsGeneral(game, mt, 9);
            }

            void GenerateAndSavePokerSituations(
                    int n_situations, const std::string &file_name, std::mt19937 &mt, const std::string &poker_type) {
                std::string name;
                int cards_in_hand;
                int n_board_cards;
                if (poker_type == "full") {
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker_type == "small") {
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=2,numRanks=6,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker_type == "leduc") {
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=2,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else if (poker_type == "three_card") {
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=1,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }
                std::shared_ptr<const Game> game = LoadGame(name);
                std::ofstream oss;
                oss.open(file_name);
                oss.flags(std::ios::scientific);
                auto state = GetPokerStatesAfterMoves(game, {}, {}, cards_in_hand);
                algorithms::PokerData poker_data(*state);
                oss.precision(std::numeric_limits<double>::digits10 + 1);
                for (int situation = 0; situation < n_situations; situation++) {
                    // Generation
                    std::vector<int> board_cards = GenerateBoardCardsGeneral(mt, poker_data.num_cards_, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = GenerateRangesGeneral(board_cards, mt, poker_data);
                    std::vector<int> action_sequence = GenerateActionsGeneral(game, mt,
                                                                              board_cards.size() + 2 * cards_in_hand);

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

            std::vector<int> ReadBoardCards(std::ifstream &file, int num_board_cards) {
                std::string read_label;
                std::vector<int> board_cards(num_board_cards, 0);
                file >> read_label;
                SPIEL_CHECK_EQ(read_label, "b");
                for (int i = 0; i < num_board_cards; i++) {
                    file >> board_cards[i];
                }
                return board_cards;
            }

            std::array<std::vector<double>, 2> ReadCFVs(std::ifstream &file, int num_hands) {
                std::array<std::vector<double>, 2> cfvs = {std::vector<double>(num_hands, 0.),
                                                           std::vector<double>(num_hands, 0.)};
                std::string read_label;
                file >> read_label;
                SPIEL_CHECK_EQ(read_label, "v");
                for (int player = 0; player < 2; player++) {
                    for (int i = 0; i < num_hands; i++) {
                        file >> cfvs[player][i];
                    }
                }
                return cfvs;
            }

            std::array<std::vector<double>, 2> ReadRanges(std::ifstream &file, int num_hands) {
                std::array<std::vector<double>, 2> ranges = {std::vector<double>(num_hands, 0.),
                                                             std::vector<double>(num_hands, 0.)};
                std::string read_label;
                file >> read_label;
                SPIEL_CHECK_EQ(read_label, "r");
                for (int player = 0; player < 2; player++) {
                    for (int i = 0; i < num_hands; i++) {
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

            std::array<std::vector<double>, 2> SolveLimitPokerSituationFromInputs(
                    const std::vector<int> &board_cards, const std::array<std::vector<double>, 2> &ranges,
                    const std::vector<int> &action_sequence, int iterations, std::string poker_type,
                    algorithms::PokerData poker_data,
                    const std::shared_ptr<const Game> &game) {
                auto state = GetPokerStatesAfterMoves(game, board_cards, action_sequence, poker_data.cards_in_hand_);
                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);

                UpdateChanceReaches(chance_reaches, poker_data, board_cards);
                for (auto &chance_reach : chance_reaches) {
                    if (chance_reach > 0) {
                        chance_reach = 1;
                    }
                }
                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees =
                        algorithms::MakePokerInfostateTrees(
                                state, chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage,
                                board_cards);

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                out->initial_state().beliefs = ranges;

                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();


                SubgameSolver solver = SubgameSolver(out, nullptr, terminal_evaluator,
                                                     std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                solver.RunSimultaneousIterations(iterations);

                PublicState root_public_state = out->public_states[0];
                SPIEL_CHECK_TRUE(root_public_state.IsInitial());

                return root_public_state.values;
            }

            void SolvePokerSubgames(const std::string &file_in, const std::string &file_out,
                                    int num_games, int iterations, const std::string &poker_type) {
                // Create in and out filestreams
                std::ifstream read_file(file_in);
                std::ofstream oss;
                oss.open(file_out);
                oss.flags(std::ios::scientific);
                oss.precision(std::numeric_limits<double>::digits10 + 1);

                // Create the game
                std::string name;
                int cards_in_hand;
                int n_board_cards;
                if (poker_type == "full") {
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker_type == "small") {
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=2,numRanks=6,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker_type == "leduc") {
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=2,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else if (poker_type == "three_card") {
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=1,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc, three_card");
                }
                std::shared_ptr<const Game> game = LoadGame(name);
                auto state = GetPokerStatesAfterMoves(game, {}, {}, cards_in_hand);
                algorithms::PokerData poker_data(*state);

                // Iterate over different situations
                for (int game_index = 0; game_index < num_games; game_index++) {
                    // Load the situation from the input file
                    std::vector<int> board_cards = ReadBoardCards(read_file, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file, poker_data.num_hands_);
                    std::vector<int> action_sequence = ReadActions(read_file);

                    // Solve the situation
                    std::array<std::vector<double>, 2> cf_values = SolveLimitPokerSituationFromInputs(
                            board_cards, ranges, action_sequence, iterations, poker_type, poker_data, game);

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

            int GetPotFromActions(const std::vector<int> &action_sequence, const std::string &poker_type,
                                  const std::vector<int> &round_bets, int initial_pot) {
                int round = 0;
                int pot = initial_pot;
                bool first_round_action = true;
                for (int action : action_sequence) {
                    if (action == 2) {
                        pot += 2 * round_bets[round];
                        first_round_action = false;
                    } else {
                        if (first_round_action) {
                            first_round_action = false;
                        } else {
                            round++;
                            first_round_action = true;
                        }
                    }
                }

                return pot;
            }

            int GetGamePotFromActions(std::vector<int> action_sequence) {
                std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                   "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                                   "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                std::shared_ptr<const Game> game = LoadGame(name);

                std::unique_ptr <State> state = game->NewInitialState();

                // Deal 4 cards
                state->ApplyAction(0);
                state->ApplyAction(1);
                state->ApplyAction(2);
                state->ApplyAction(3);

                // Pre-flop betting
                int action_index = 0;
                while (state->IsPlayerNode()) {
                    state->ApplyAction(action_sequence[action_index]);
                    action_index++;
                }

                // Deal 3 board cards (Flop)
                state->ApplyAction(4);
                state->ApplyAction(5);
                state->ApplyAction(6);

                // Flop betting
                while (state->IsPlayerNode()) {
                    state->ApplyAction(action_sequence[action_index]);
                    action_index++;
                }

                // Deal board card (Turn)
                state->ApplyAction(7);

                // Turn betting
                while (state->IsPlayerNode()) {
                    state->ApplyAction(action_sequence[action_index]);
                    action_index++;
                }

                const auto &poker_state = open_spiel::down_cast<const universal_poker::UniversalPokerState &>(*state);
                int pot = 0;
                for (int p = 0; p < 2; p++) {
                    pot += poker_state.acpc_state().Ante(p);
                }
                return pot;
            }

            void NetworkTraining(const std::string &file_name,
                                 int training_samples,
                                 int validation_samples,
                                 int epochs,
                                 int batch_size,
                                 bool normalize,
                                 const std::string &log_file,
                                 const std::string &poker_type,
                                 int layer_size,
                                 int layer_number) {
                int cards_in_hand;
                int n_board_cards;
                int deck_cards;
                int num_hands;
                int initial_pot;
                std::vector<int> round_bets;
                if (poker_type == "full") {
                    deck_cards = 52;
                    num_hands = 1326;
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    initial_pot = 20;
                    round_bets = {10, 10, 20, 20};
                } else if (poker_type == "small") {
                    num_hands = 66;
                    deck_cards = 12;
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    initial_pot = 20;
                    round_bets = {10, 10, 20, 20};
                } else if (poker_type == "leduc") {
                    num_hands = 6;
                    deck_cards = 6;
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    initial_pot = 2;
                    round_bets = {2, 4};
                } else if (poker_type == "three_card") {
                    deck_cards = 3;
                    num_hands = 3;
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    initial_pot = 2;
                    round_bets = {2, 4};
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }
                ClearLog(log_file);
                LogLine("Started training", log_file);

                auto net = std::make_shared<Net>(deck_cards, num_hands, layer_size, layer_number);

                torch::Device device("cpu");

                net->to(device);

                std::ifstream read_file(file_name);

                std::cout << training_samples << ", " << net->input_size_ << "\n";

                int batches = training_samples / batch_size;

                std::vector <torch::Tensor> training_data;
                std::vector <torch::Tensor> training_targets;
                training_data.reserve(batches);
                training_targets.reserve(batches);
                int batch_index = 0;
                int point_index = 0;
                for (int i = 0; i < training_samples; i++) {
                    if (point_index == 0) {
                        training_data.push_back(torch::zeros({batch_size, net->input_size_}));
                        training_targets.push_back(torch::zeros({batch_size, net->output_size_}));
                    }
                    std::vector<int> board_cards = ReadBoardCards(read_file, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file, num_hands);
                    std::vector<int> action_sequence = ReadActions(read_file);
                    int pot = GetPotFromActions(action_sequence, poker_type, round_bets, initial_pot);
                    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file, num_hands);

                    training_data[batch_index][point_index][board_cards.back()] = 1;
                    std::vector<double> range_magnitudes(2);
                    if (normalize) {
                        for (int card_index = 0; card_index < num_hands; card_index++) {
                            range_magnitudes[0] += ranges[0][card_index];
                            range_magnitudes[1] += ranges[1][card_index];
                        }
                    }
                    for (double &range_magnitude : range_magnitudes) {
                        if (range_magnitude < 1e-3) {
                            range_magnitude = 1;
                        }
                    }
                    for (int card_index = 0; card_index < num_hands; card_index++) {
                        training_data[batch_index][point_index][card_index + deck_cards] =
                                ranges[0][card_index] / (normalize ? range_magnitudes[0] : 1);
                        training_data[batch_index][point_index][card_index + num_hands + deck_cards] =
                                ranges[1][card_index] / (normalize ? range_magnitudes[1] : 1);

                        training_targets[batch_index][point_index][card_index] =
                                cfvs[0][card_index] / ((normalize ? range_magnitudes[1] : 1) * pot);
                        training_targets[batch_index][point_index][card_index + num_hands] =
                                cfvs[1][card_index] / ((normalize ? range_magnitudes[0] : 1) * pot);
                    }
                    training_data[batch_index][point_index][net->input_size_ - 1] = pot;
                    point_index++;
                    if (point_index == batch_size) {
                        point_index = 0;
                        batch_index++;
                    }
                }

                torch::Tensor validation_data_tensor = torch::zeros({validation_samples, net->input_size_});
                torch::Tensor validation_target_tensor = torch::zeros({validation_samples, net->output_size_});
                for (int i = 0; i < validation_samples; i++) {
                    std::vector<int> board_cards = ReadBoardCards(read_file, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file, num_hands);
                    std::vector<int> action_sequence = ReadActions(read_file);
                    int pot = GetPotFromActions(action_sequence, poker_type, round_bets, initial_pot);
                    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file, num_hands);

                    validation_data_tensor[i][board_cards.back()] = 1;

                    std::vector<double> range_magnitudes(2);
                    if (normalize) {
                        for (int card_index = 0; card_index < num_hands; card_index++) {
                            range_magnitudes[0] += ranges[0][card_index];
                            range_magnitudes[1] += ranges[1][card_index];
                        }
                    }
                    for (double &range_magnitude : range_magnitudes) {
                        if (range_magnitude < 1e-3) {
                            range_magnitude = 1;
                        }
                    }
                    for (int card_index = 0; card_index < num_hands; card_index++) {
                        validation_data_tensor[i][card_index + deck_cards] =
                                ranges[0][card_index] / (normalize ? range_magnitudes[0] : 1);
                        validation_data_tensor[i][card_index + num_hands + deck_cards] =
                                ranges[1][card_index] / (normalize ? range_magnitudes[1] : 1);

                        validation_target_tensor[i][card_index] =
                                cfvs[0][card_index] / ((normalize ? range_magnitudes[1] : 1) * pot);
                        validation_target_tensor[i][card_index + num_hands] =
                                cfvs[1][card_index] / ((normalize ? range_magnitudes[0] : 1) * pot);
                    }
                    validation_data_tensor[i][net->input_size_ - 1] = pot;
                }

                std::cout << training_data[0] << "\n";
                std::cout << training_targets[0] << "\n";

                std::ofstream oss;
                oss.open(
                        poker_type + "_" + std::to_string(layer_size) + "_" + std::to_string(layer_number) + "_losses");

                std::cout << "Batches: " << batches << "\n";

                double lower_lr_bound = 1e-6;
                auto optimizer = std::make_shared<torch::optim::Adam>(net->parameters(),
                                                                      torch::optim::AdamOptions(1e-4));

                LogLine("Initial:   ", log_file);
                double init_cumulative_loss = 0;
                for (int batch = 0; batch < batches; batch++) {
                    torch::Tensor output = net->forward(training_data[batch]);
                    torch::Tensor loss = torch::nn::functional::smooth_l1_loss(output, training_targets[batch]);
                    init_cumulative_loss += loss.item().to<double>();
                }
                std::cout << "Training loss: " << init_cumulative_loss / batches << "\n";
                Log("Training loss: ", log_file);
                Log(std::to_string(init_cumulative_loss / batches), log_file);
                Log("   ", log_file);

                torch::Tensor init_output = net->forward(validation_data_tensor);
                torch::Tensor init_loss = torch::nn::functional::smooth_l1_loss(init_output, validation_target_tensor);
                std::cout << "Validation loss: " << init_loss.item().to<double>() << "\n";
                Log("Validation loss: ", log_file);
                Log(std::to_string(init_loss.item().to<double>()), log_file);
                LogLine("   ", log_file);

                oss << init_cumulative_loss / batches << " " << init_loss.item().to<double>() << "\n";

                for (int epoch = 0; epoch < epochs; epoch++) {
                    std::cout << "Epoch " << epoch << ":   ";
                    Log("Epoch ", log_file);
                    Log(std::to_string(epoch), log_file);
                    LogLine(":   ", log_file);
                    double cumulative_loss = 0;
                    for (int batch = 0; batch < batches; batch++) {
                        torch::Tensor output = net->forward(training_data[batch]);
                        torch::Tensor loss = torch::nn::functional::smooth_l1_loss(output, training_targets[batch]);
                        loss.backward();
                        optimizer->step();
                        cumulative_loss += loss.item().to<double>();
                        std::cout << "." << std::flush;
                        if (epoch % 10 == 0) {
                            torch::save(net,
                                        "models/" + poker_type + "_epoch_" + std::to_string(epoch) + "_layers_"
                                        + std::to_string(layer_number) + "x" + std::to_string(layer_size));
                        }
                    }
                    Log("Training loss: ", log_file);
                    Log(std::to_string(cumulative_loss / batches), log_file);
                    Log("   ", log_file);
                    std::cout << "\nTraining loss: " << cumulative_loss / batches << "   ";
                    torch::Tensor output = net->forward(validation_data_tensor);
                    torch::Tensor loss = torch::nn::functional::smooth_l1_loss(output, validation_target_tensor);
                    std::cout << "Validation loss: " << loss.item().to<double>() << "\n";
                    Log("Validation loss: ", log_file);
                    Log(std::to_string(loss.item().to<double>()), log_file);
                    LogLine("   ", log_file);
                    oss << cumulative_loss / batches << " " << loss.item().to<double>() << "\n";
                    if (epoch > 200) {
                        for (auto &group : optimizer->param_groups()) {
                            if (group.has_options()) {
                                auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
                                options.lr(std::max(options.lr() * 0.999, lower_lr_bound));
                                std::cout << options.lr() << "\n";
                            }
                        }
                    }
                }
            }

            void NetExploitabilityTrunkStrategy(
                    int iterations, int full_iterations, const std::string &net_file,
                    const std::string &log_file, const std::string &poker, int layer_size, int layer_number) {
                ClearLog(log_file);
                LogLine("Exploitability experiment", log_file);
                auto start_time = GetTime();
                std::vector<int> board_cards;
                std::string name;
                std::vector<int> action_sequence;
                int cards_in_hand;
                std::array<std::vector<double>, 2> ranges;
                if (poker == "full") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {23, 28, 30, 32};
                    cards_in_hand = 2;
                    ranges = GetReachesFromVector(SUBGAME_ONE_RANGES);
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "small") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {1, 3, 5, 9};
                    cards_in_hand = 2;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=2,numRanks=6,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "leduc") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=2,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else if (poker == "three_card") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=1,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }
                std::shared_ptr<const Game> game = LoadGame(name);

                std::unique_ptr <State> state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                         cards_in_hand);
                std::unique_ptr <State> full_state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                              cards_in_hand);

                LogLine("States created: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);

                UpdateChanceReaches(chance_reaches, poker_data, board_cards);

                std::vector<double> full_chance_reaches = chance_reaches;

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
                        state, chance_reaches, infostate_observer, 1, kDlCfrInfostateTreeStorage, board_cards);

                LogLine("Made DL trees: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                if (!ranges[0].empty()) {
                    out->initial_state().beliefs = ranges;
                }

                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

                std::shared_ptr<const PublicStateEvaluator> leaf_evaluator =
                        std::make_shared<const RiverNetworkLeafEvaluator>(
                                net_file, poker_data.num_cards_, poker_data.num_hands_, layer_size, layer_number);

                SubgameSolver solver = SubgameSolver(out, leaf_evaluator, terminal_evaluator,
                                                     std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Made DL solver: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                // Create solver for full TURN
                std::vector <std::shared_ptr<algorithms::InfostateTree>> full_trees = algorithms::MakePokerInfostateTrees(
                        full_state, full_chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage,
                        board_cards);

                LogLine("Made full trees: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto full_out = std::make_shared<Subgame>(game, public_observer, full_trees);

                if (!ranges[0].empty()) {
                    full_out->initial_state().beliefs = ranges;
                }

                SubgameSolver full_solver = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                          std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Made full solver: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                for (int iteration = 0; iteration < iterations; iteration++) {
                    solver.RunSimultaneousIterations(1, true);
                    LogLine("Iteration: " + std::to_string(iteration), log_file);
                }

                LogLine("Computed trunk strategy: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto strategy = down_cast<algorithms::BanditsAveragePolicy>(*solver.AveragePolicy());

                std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(), TabularPolicy()};

                separated_policies[0] = strategy.TabularizeAveragePlayer(0);
                separated_policies[1] = strategy.TabularizeAveragePlayer(1);

//  for (int player = 0; player < 2; player++) {
//    algorithms::BanditVector &bandits = solver.bandits()[player];
//    for (algorithms::DecisionId id : bandits.range()) {
//      algorithms::InfostateNode *node = solver.subgame()->trees[player]->decision_infostate(id);
//      const std::string &infostate = node->infostate_string();
//      ActionsAndProbs infostate_policy = strategy->GetStatePolicy(infostate);
//      separated_policies[player].SetStatePolicy(infostate, infostate_policy);
//    }
//  }

                LogLine("Created separated policies from trunk strategy: " + std::to_string(ElapsedTime(start_time)),
                        log_file);
                start_time = GetTime();

                double nash_conv = 0;

                for (int player = 0; player < 2; player++) {
                    SubgameSolver best_response = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                                std::make_shared<std::mt19937>(0),
                                                                "RegretMatchingPlus");
                    algorithms::BanditVector &bandits = best_response.bandits()[player];
                    for (algorithms::DecisionId id : bandits.range()) {
                        algorithms::InfostateNode *node = best_response.subgame()->trees[player]->decision_infostate(
                                id);
                        const std::string &infostate = node->infostate_string();
                        ActionsAndProbs infostate_policy = strategy.GetStatePolicy(infostate);
                        if (!infostate_policy.empty()) {
                            bandits[id] = std::make_unique<algorithms::bandits::FixedStrategy>(
                                    GetProbs(infostate_policy));
                        }
                    }

                    LogLine("Prepared solver for DL BR: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    for (int full_iteration = 0; full_iteration < full_iterations; full_iteration++) {
                        best_response.RunSimultaneousIterations(1);
                        LogLine("Iteration " + std::to_string(full_iteration), log_file);
                    }

                    LogLine("Solved the exploitability: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    auto response_strategy = down_cast<algorithms::BanditsAveragePolicy>(
                            *best_response.AveragePolicy());
                    auto separated_response_strategy = response_strategy.TabularizeAverage();

                    for (Player bandit_player = 0; bandit_player < 2; bandit_player++) {
                        auto policy = best_response.AveragePolicy();
                        algorithms::BanditVector &local_bandits = best_response.bandits()[bandit_player];
                        for (algorithms::DecisionId id : local_bandits.range()) {
                            algorithms::InfostateNode *node = best_response.subgame()->trees[bandit_player]->decision_infostate(
                                    id);
                            const std::string &infostate = node->infostate_string();
                            ActionsAndProbs infostate_policy = separated_response_strategy[bandit_player].GetStatePolicy(
                                    infostate);
                            if (!infostate_policy.empty()) {
                                local_bandits[id] = std::make_unique<algorithms::bandits::FixedStrategy>(
                                        GetProbs(infostate_policy));
                            }
                        }
                    }

                    LogLine("Copied strategies: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    best_response.Reset();
                    best_response.RunSimultaneousIterations(2);
//    std::cout << "Best response: " << best_response.RootValues() << "\n";
                    Log("Best response ", log_file);
                    Log(std::to_string(best_response.RootValues()[0]) + ", ", log_file);
                    LogLine(std::to_string(best_response.RootValues()[1]), log_file);
                    nash_conv += best_response.RootValues()[1 - player];

                    LogLine("Best response evaluated: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();
                }
                std::cout << nash_conv / 2 << "\n";
                LogLine("Exploitability: " + std::to_string(nash_conv / 2), log_file);
            }

            void ComputeNetLosses(const std::string &data_file, int samples_from, int samples,
                                  const std::string &net_directory,
                                  int models, bool normalize, const std::string &log_file,
                                  const std::string &poker_type,
                                  int layer_size, int layer_number) {
                int cards_in_hand;
                int n_board_cards;
                int deck_cards;
                int num_hands;
                int initial_pot;
                std::vector<int> round_bets;
                if (poker_type == "full") {
                    deck_cards = 52;
                    num_hands = 1326;
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    initial_pot = 20;
                    round_bets = {10, 10, 20, 20};
                } else if (poker_type == "small") {
                    num_hands = 66;
                    deck_cards = 12;
                    cards_in_hand = 2;
                    n_board_cards = 5;
                    initial_pot = 20;
                    round_bets = {10, 10, 20, 20};
                } else if (poker_type == "leduc") {
                    num_hands = 6;
                    deck_cards = 6;
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    initial_pot = 2;
                    round_bets = {2, 4};
                } else if (poker_type == "three_card") {
                    deck_cards = 3;
                    num_hands = 3;
                    cards_in_hand = 1;
                    n_board_cards = 1;
                    initial_pot = 2;
                    round_bets = {2, 4};
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }

                ClearLog(log_file);

                auto net = std::make_shared<Net>(deck_cards, num_hands, layer_size, layer_number);

                torch::Device device("cpu");

                net->to(device);

                std::ifstream read_file(data_file);

                torch::Tensor data_tensor = torch::zeros({samples, net->input_size_});
                torch::Tensor target_tensor = torch::zeros({samples, net->output_size_});
                for (int i = 0; i < samples_from; i++) {
                    std::vector<int> board_cards = ReadBoardCards(read_file, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file, num_hands);
                    std::vector<int> action_sequence = ReadActions(read_file);
                    int pot = GetPotFromActions(action_sequence, poker_type, round_bets, initial_pot);
                    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file, num_hands);
                }

                for (int i = 0; i < samples; i++) {
                    std::vector<int> board_cards = ReadBoardCards(read_file, n_board_cards);
                    std::array<std::vector<double>, 2> ranges = ReadRanges(read_file, num_hands);
                    std::vector<int> action_sequence = ReadActions(read_file);
                    int pot = GetPotFromActions(action_sequence, poker_type, round_bets, initial_pot);
                    std::array<std::vector<double>, 2> cfvs = ReadCFVs(read_file, num_hands);

                    data_tensor[i][board_cards.back()] = 1;

                    std::vector<double> range_magnitudes(2);
                    if (normalize) {
                        for (int card_index = 0; card_index < num_hands; card_index++) {
                            range_magnitudes[0] += ranges[0][card_index];
                            range_magnitudes[1] += ranges[1][card_index];
                        }
                    }
                    for (double &range_magnitude : range_magnitudes) {
                        if (range_magnitude < 1e-3) {
                            range_magnitude = 1;
                        }
                    }
                    for (int card_index = 0; card_index < num_hands; card_index++) {
                        data_tensor[i][card_index + deck_cards] =
                                ranges[0][card_index] / (normalize ? range_magnitudes[0] : 1);
                        data_tensor[i][card_index + num_hands + deck_cards] =
                                ranges[1][card_index] / (normalize ? range_magnitudes[1] : 1);

                        target_tensor[i][card_index] =
                                cfvs[0][card_index] / ((normalize ? range_magnitudes[1] : 1) * pot);
                        target_tensor[i][card_index + num_hands] =
                                cfvs[1][card_index] / ((normalize ? range_magnitudes[0] : 1) * pot);
                    }
                }

                for (int i = 0; i < models; i++) {
                    std::string net_file = net_directory + "_epoch_" + std::to_string(i * 10) + "_layers_" +
                                           std::to_string(layer_size);
                    torch::load(net, net_file);

                    torch::Tensor output = net->forward(data_tensor);
                    torch::Tensor loss = torch::nn::functional::smooth_l1_loss(output, target_tensor);
//    torch::Tensor loss_inf = torch::subtract(output, target_tensor).max();
                    std::cout << net_file << "Loss: " << loss.item().to<double>() << "\n";
//    std::cout << net_file  << "Loss inf: " << loss_inf.item().to<double>() << "\n";
                }
            }

            void CDBR(int iterations, int bad_iterations, const std::string &net_file, const std::string &log_file,
                      const std::string &poker, int layer_size, int layer_number) {
                ClearLog(log_file);
                LogLine("Started CDBR", log_file);

                auto complete_beginning = GetTime();
                auto start = GetTime();
                std::vector<int> board_cards;
                std::string name;
                std::vector<int> action_sequence;
                int cards_in_hand;
                std::array<std::vector<double>, 2> ranges;
                if (poker == "full") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {23, 28, 30, 32};
                    cards_in_hand = 2;
                    ranges = GetReachesFromVector(SUBGAME_ONE_RANGES);
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "small") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {1, 3, 5, 9};
                    cards_in_hand = 2;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=2,numRanks=6,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "leduc") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=2,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else if (poker == "three_card") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=1,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }
                std::shared_ptr<const Game> game = LoadGame(name);

                std::unique_ptr <State> state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                         cards_in_hand);
                std::unique_ptr <State> full_state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                              cards_in_hand);

                LogLine("States created: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);

                UpdateChanceReaches(chance_reaches, poker_data, board_cards);

                std::vector<double> full_chance_reaches = chance_reaches;

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
                        state, chance_reaches, infostate_observer, 1, kDlCfrInfostateTreeStorage, board_cards);

                LogLine("Trees done: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                if (!ranges[0].empty()) {
                    out->initial_state().beliefs = ranges;
                }
                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

                std::shared_ptr<const PublicStateEvaluator> leaf_evaluator =
                        std::make_shared<const RiverNetworkLeafEvaluator>(net_file, poker_data.num_cards_,
                                                                          poker_data.num_hands_, layer_size,
                                                                          layer_number);

                SubgameSolver solver = SubgameSolver(out, leaf_evaluator, terminal_evaluator,
                                                     std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Solver done: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                // Create solver for full TURN
                std::vector <std::shared_ptr<algorithms::InfostateTree>> full_trees = algorithms::MakePokerInfostateTrees(
                        full_state, full_chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage,
                        board_cards);

                LogLine("Full trees done: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                auto full_out = std::make_shared<Subgame>(game, public_observer, full_trees);

                if (!ranges[0].empty()) {
                    full_out->initial_state().beliefs = ranges;
                }

                SubgameSolver full_solver = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                          std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Full solver done: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                full_solver.RunSimultaneousIterations(bad_iterations);

                auto bad_policy = down_cast<algorithms::BanditsAveragePolicy>(*full_solver.AveragePolicy());

                LogLine("Bad policy creation: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                std::array<TabularPolicy, 2> bad_separated_policies = {TabularPolicy(), TabularPolicy()};

                bad_separated_policies[0] = bad_policy.TabularizeAveragePlayer(0);
                bad_separated_policies[1] = bad_policy.TabularizeAveragePlayer(1);

                LogLine("Bad policy copying: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                std::vector <algorithms::BanditVector> dl_bandits = MakeResponseBandits(trees,
                                                                                        bad_separated_policies[1]);

                for (algorithms::DecisionId id : dl_bandits[1].range()) {
                    solver.bandits()[1][id] = std::move(dl_bandits[1][id]);
                }

                LogLine("Response saving: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                for (int iteration = 0; iteration < iterations; iteration++) {
                    solver.RunSimultaneousIterations(1, true);
                    LogLine("Iteration: " + std::to_string(iteration), log_file);
                }

                LogLine("Rest of the response computation: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                auto cdbr_trunk = down_cast<algorithms::BanditsAveragePolicy>(*solver.AveragePolicy());

                auto cdbr_policy_with_opponent = bad_separated_policies[1];
                cdbr_policy_with_opponent.ImportPolicy(cdbr_trunk.TabularizeAveragePlayer(0));

                SubgameSolver cdbr_solver = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                          std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
                cdbr_solver.bandits() = MakeResponseBandits(full_trees, cdbr_policy_with_opponent);

                LogLine("Response saving for BR: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                cdbr_solver.RunSimultaneousIterations(2);
                std::cout << "CD best response: " << cdbr_solver.RootValues() << "\n";
                LogLine("CD best response: " + std::to_string(cdbr_solver.RootValues()[0]) + ", "
                        + std::to_string(cdbr_solver.RootValues()[1]) + "\n", log_file);

                LogLine("BR: " + std::to_string(ElapsedTime(start)), log_file);
                start = GetTime();

                SubgameSolver best_response = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                            std::make_shared<std::mt19937>(0), "RegretMatchingPlus");
                best_response.bandits() = MakeResponseBandits(full_trees, bad_separated_policies[1]);
                best_response.RunSimultaneousIterations(2);
                std::cout << "Best response: " << best_response.RootValues() << "\n";
                LogLine("Best response: " + std::to_string(best_response.RootValues()[0]) + ", "
                        + std::to_string(best_response.RootValues()[1]) + "\n", log_file);

                LogLine("Full BR: " + std::to_string(ElapsedTime(start)), log_file);

                LogLine("Full time: " + std::to_string(ElapsedTime(complete_beginning)), log_file);
            }

            void OracleTest(int iterations,
                            int full_iterations,
                            int subgame_iterations,
                            const std::string &log_file,
                            const std::string &poker) {
                ClearLog(log_file);
                LogLine("Oracle experiment", log_file);
                std::vector<int> board_cards;
                std::string name;
                std::vector<int> action_sequence;
                int cards_in_hand;
                std::array<std::vector<double>, 2> ranges;
                if (poker == "full") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {23, 28, 30, 32};
                    cards_in_hand = 2;
                    ranges = GetReachesFromVector(SUBGAME_ONE_RANGES);
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "small") {
                    action_sequence = {1, 1, 1, 2, 1};
                    board_cards = {1, 3, 5, 9};
                    cards_in_hand = 2;
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                           "firstPlayer=2 1,numSuits=2,numRanks=6,numHoleCards=2,numBoardCards=0 3 "
                           "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
                } else if (poker == "leduc") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=2,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else if (poker == "three_card") {
                    cards_in_hand = 1;
                    action_sequence = {};
                    board_cards = {};
                    name = "universal_poker(betting=limit,numPlayers=2,numRounds=2,blind=1 1,"
                           "firstPlayer=1 1,numSuits=1,numRanks=3,numHoleCards=1,numBoardCards=0 1"
                           ",raiseSize=2 4,maxRaises=2 2)";
                } else {
                    SpielFatalError("Incorrect poker type. Possible types: full, small, leduc");
                }

                auto start_time = GetTime();
                std::shared_ptr<const Game> game = LoadGame(name);

                std::unique_ptr <State> state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                         cards_in_hand);
                std::unique_ptr <State> full_state = GetPokerStatesAfterMoves(game, board_cards, action_sequence,
                                                                              cards_in_hand);

                LogLine("States created: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                std::shared_ptr <Observer> infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                std::shared_ptr <Observer> public_observer = game->MakeObserver(kPublicStateObsType, {});

                algorithms::PokerData poker_data = algorithms::PokerData(*state);

                std::vector<double> chance_reaches(poker_data.num_hands_, 1. / poker_data.num_hands_);

                UpdateChanceReaches(chance_reaches, poker_data, board_cards);

                std::vector<double> full_chance_reaches = chance_reaches;

                std::vector <std::shared_ptr<algorithms::InfostateTree>> trees = algorithms::MakePokerInfostateTrees(
                        state, chance_reaches, infostate_observer, 1, kDlCfrInfostateTreeStorage, board_cards);

                LogLine("Made DL trees: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto out = std::make_shared<Subgame>(game, public_observer, trees);

                if (!ranges[0].empty()) {
                    out->initial_state().beliefs = ranges;
                }
                std::shared_ptr<const PublicStateEvaluator>
                        terminal_evaluator = std::make_shared<const GeneralPokerTerminalEvaluatorLinear>();

                std::shared_ptr<const PublicStateEvaluator> leaf_evaluator =
                        std::make_shared<const PokerCFREvaluator>(game, terminal_evaluator, public_observer,
                                                                  infostate_observer,
                                                                  subgame_iterations);

                SubgameSolver solver = SubgameSolver(out, leaf_evaluator, terminal_evaluator,
                                                     std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Made DL solver: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                // Create solver for full TURN
                std::vector <std::shared_ptr<algorithms::InfostateTree>> full_trees = algorithms::MakePokerInfostateTrees(
                        full_state, full_chance_reaches, infostate_observer, 1000, kDlCfrInfostateTreeStorage,
                        board_cards);

                LogLine("Made full trees: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto full_out = std::make_shared<Subgame>(game, public_observer, full_trees);

                if (!ranges[0].empty()) {
                    full_out->initial_state().beliefs = ranges;
                }

                SubgameSolver full_solver = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                          std::make_shared<std::mt19937>(0), "RegretMatchingPlus");

                LogLine("Made full solver: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                for (int iteration = 0; iteration < iterations; iteration++) {
                    solver.RunSimultaneousIterations(1);
                    LogLine("Iteration: " + std::to_string(iteration), log_file);
                }

                return;

                LogLine("Computed trunk strategy: " + std::to_string(ElapsedTime(start_time)), log_file);
                start_time = GetTime();

                auto strategy = down_cast<algorithms::BanditsAveragePolicy>(*solver.AveragePolicy());

                std::array<TabularPolicy, 2> separated_policies = {TabularPolicy(), TabularPolicy()};

                separated_policies[0] = strategy.TabularizeAveragePlayer(0);
                separated_policies[1] = strategy.TabularizeAveragePlayer(1);

//  for (int player = 0; player < 2; player++) {
//    algorithms::BanditVector &bandits = solver.bandits()[player];
//    for (algorithms::DecisionId id : bandits.range()) {
//      algorithms::InfostateNode *node = solver.subgame()->trees[player]->decision_infostate(id);
//      const std::string &infostate = node->infostate_string();
//      ActionsAndProbs infostate_policy = strategy->GetStatePolicy(infostate);
//      separated_policies[player].SetStatePolicy(infostate, infostate_policy);
//    }
//  }

                LogLine("Created separated policies from trunk strategy: " + std::to_string(ElapsedTime(start_time)),
                        log_file);
                start_time = GetTime();

                double nash_conv = 0;

                for (int player = 0; player < 2; player++) {
                    SubgameSolver best_response = SubgameSolver(full_out, nullptr, terminal_evaluator,
                                                                std::make_shared<std::mt19937>(0),
                                                                "RegretMatchingPlus");
                    algorithms::BanditVector &bandits = best_response.bandits()[player];
                    for (algorithms::DecisionId id : bandits.range()) {
                        algorithms::InfostateNode *node = best_response.subgame()->trees[player]->decision_infostate(
                                id);
                        const std::string &infostate = node->infostate_string();
                        ActionsAndProbs infostate_policy = strategy.GetStatePolicy(infostate);
                        if (!infostate_policy.empty()) {
                            bandits[id] = std::make_unique<algorithms::bandits::FixedStrategy>(
                                    GetProbs(infostate_policy));
                        }
                    }

                    LogLine("Prepared solver for DL BR: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    for (int full_iteration = 0; full_iteration < full_iterations; full_iteration++) {
                        best_response.RunSimultaneousIterations(1);
                        LogLine("Iteration " + std::to_string(full_iteration), log_file);
                    }

                    LogLine("Solved the exploitability: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    auto response_strategy = down_cast<algorithms::BanditsAveragePolicy>(
                            *best_response.AveragePolicy());
                    auto separated_response_strategy = response_strategy.TabularizeAverage();

                    for (Player bandit_player = 0; bandit_player < 2; bandit_player++) {
                        auto policy = best_response.AveragePolicy();
                        algorithms::BanditVector &local_bandits = best_response.bandits()[bandit_player];
                        for (algorithms::DecisionId id : local_bandits.range()) {
                            algorithms::InfostateNode *node = best_response.subgame()->trees[bandit_player]->decision_infostate(
                                    id);
                            const std::string &infostate = node->infostate_string();
                            ActionsAndProbs infostate_policy = separated_response_strategy[bandit_player].GetStatePolicy(
                                    infostate);
                            if (!infostate_policy.empty()) {
                                local_bandits[id] = std::make_unique<algorithms::bandits::FixedStrategy>(
                                        GetProbs(infostate_policy));
                            }
                        }
                    }

                    LogLine("Copied strategies: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();

                    best_response.Reset();
                    best_response.RunSimultaneousIterations(2);
                    std::cout << "Best response: " << best_response.RootValues() << "\n";
                    Log("Best response ", log_file);
                    Log(std::to_string(best_response.RootValues()[0]) + ", ", log_file);
                    LogLine(std::to_string(best_response.RootValues()[1]), log_file);
                    nash_conv += best_response.RootValues()[1 - player];

                    LogLine("Best response evaluated: " + std::to_string(ElapsedTime(start_time)), log_file);
                    start_time = GetTime();
                }
                std::cout << "Exploitability: " << nash_conv / 2 << "\n";
                LogLine("Exploitability: " + std::to_string(nash_conv / 2), log_file);
            }

        }
    }
}

int main(int argc, char **argv) {
    //   Limit tests
    if (argc > 1) {
        int iterations = 1000;
        int runs = 1;
        int n_arguments = argc;
        // Linear evaluator infostate CFR
        bool do_next_part = false;
        for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
            if (std::strcmp(argv[argument_index], "-isl") == 0 or strcmp(argv[argument_index], "-all") == 0) {
                do_next_part = true;
                break;
            }
        }
        if (do_next_part) {
            std::cout << "Infostate CFR experiment with Linear evaluator:\n";
            open_spiel::papers_with_code::MeasureTime(
                    runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFRPokerSpecificLinear);
        }
        // Quadratic evaluator infostate CFR
        do_next_part = false;
        for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
            if (std::strcmp(argv[argument_index], "-isq") == 0 or strcmp(argv[argument_index], "-all") == 0) {
                do_next_part = true;
                break;
            }
        }
        if (do_next_part) {
            std::cout << "Infostate CFR experiment with Quadratic evaluator:\n";
            open_spiel::papers_with_code::MeasureTime(
                    runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFRPokerSpecificQuadratic);
        }
        // Efg CFR without saving structure
        do_next_part = false;
        for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
            if (std::strcmp(argv[argument_index], "-efg") == 0 or strcmp(argv[argument_index], "-all") == 0) {
                do_next_part = true;
                break;
            }
        }
        if (do_next_part) {
            std::cout << "EFG CFR experiment:\n";
            open_spiel::papers_with_code::MeasureTime(
                    runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfg);
        }
        // Efg CFR with saving the structure
        do_next_part = false;
        for (int argument_index = 1; argument_index < n_arguments; argument_index++) {
            if (std::strcmp(argv[argument_index], "-efgh") == 0 or strcmp(argv[argument_index], "-all") == 0) {
                do_next_part = true;
                break;
            }
        }
        if (do_next_part) {
            std::cout << "Hashed EFG CFR experiment:\n";
            open_spiel::papers_with_code::MeasureTime(
                    runs, iterations, open_spiel::papers_with_code::UniversalPokerRiverCFREfgHashed);
        }
    } else {
        std::cout
                << "Please specify the experiment to run. -isl for infostate CFR with linear evaluator, -efg for EFG CFR, "
                   "-efgh for EFG CFR where the tree is build and saved, -isq for infostate CFR with quadratic evaluator and "
                   "-all for all experiments";
    }
}