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
#include "open_spiel/algorithms/poker_data.h"
#include "algorithms/expected_returns.h"

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

            std::pair<int, int> UniversalPokerRiverCFRPokerSpecificLinear(int iterations) {
                std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                   "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                                   "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
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

            std::pair<int, int> UniversalPokerRiverCFRPokerSpecificQuadratic(int iterations) {
                std::string name = "universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=10 5,"
                                   "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
                                   "1 1,raiseSize=10 10 20 20,maxRaises=3 4 4 4)";
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
        } // unnamed namespace
    }     // papers_with_code
}         // open_spiel

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