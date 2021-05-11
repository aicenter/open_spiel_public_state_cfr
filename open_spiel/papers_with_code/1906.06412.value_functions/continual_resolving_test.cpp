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

#include "open_spiel/spiel_bots.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/bot.h"
#include "open_spiel//games/nfg_game.h"

#include <cmath>
#include <iostream>


namespace open_spiel {
    namespace papers_with_code {
        namespace {

            // Tests if CFVs saved in the public states for the continual resolving are the correct ones
            void TestBasicCFVs() {
                // constructs biased matching pennies
                const char *kSampleNFGString = R"###(
                        NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
                        { "Player 1" "Player 2" } { 2 2 }

                        1 -1 0 0 0 0 2 -2
                    )###";

                std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(kSampleNFGString);
                const int trunk_iterations = 5;


                // prepared values for the test
                std::string infoset_strings[2][4] = {{
                                                             "Observing player: 0. Terminal. History string: 1, 1",
                                                             "Observing player: 0. Terminal. History string: 1, 0",
                                                             "Observing player: 0. Terminal. History string: 0, 1",
                                                             "Observing player: 0. Terminal. History string: 0, 0"
                                                     },
                                                     {
                                                             "Observing player: 1. Terminal. History string: 1, 1",
                                                             "Observing player: 1. Terminal. History string: 0, 1",
                                                             "Observing player: 1. Terminal. History string: 1, 0",
                                                             "Observing player: 1. Terminal. History string: 0, 0"
                                                     }};

                double reference_values[2][4][5] = {{
                                                            {1,  0.5,  1. / 3,  0.25,   0.2},
                                                            {0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0},
                                                            {0.5,  0.75,  5. / 6,   0.875,   0.9}},
                                                    {       {-1, -1.5, -7. / 6, -0.875, -0.7},
                                                            {0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0},
                                                            {-0.5, -0.25, -5. / 12, -0.5625, -0.65}}};

                // creating a CFR subgame solver
                std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
                        MakeTerminalEvaluator();
                std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
                        MakeDummyEvaluator();
                std::shared_ptr<Observer> public_observer =
                        game->MakeObserver(kPublicStateObsType, {});
                std::shared_ptr<Observer> infostate_observer =
                        game->MakeObserver(kInfoStateObsType, {});

                auto leaf_evaluator = std::make_shared<CFREvaluator>(
                        game, 1, nonterminal_evaluator,
                        terminal_evaluator, public_observer, infostate_observer);
                leaf_evaluator->reset_subgames_on_evaluation = false;  // Needed for test !
                leaf_evaluator->bandit_name = "RegretMatching";
                leaf_evaluator->nonterminal_evaluator = leaf_evaluator;
                leaf_evaluator->num_cfr_iterations = 1;
                leaf_evaluator->save_values_policy = SaveValuesPolicy::kCurrentCfValues;


                auto subgame = std::make_shared<Subgame>(game, 1);
                auto subgame_solver = std::make_unique<SubgameSolver>(subgame, leaf_evaluator,
                                                                      terminal_evaluator, "RegretMatching",
                                                                      SaveValuesPolicy::kCurrentCfValues,
                                                                      true);

                // we do 5 iterations and check the CFVs after each iteration
                for (int i; i < trunk_iterations; i++) {
                    subgame_solver->RunSimultaneousIterations(1);

                    for (auto &public_state : subgame->public_states) {
                        if (public_state.IsTerminal()) {
                            for (int player = 0; player < 2; player++) {
                                auto CFVs = public_state.GetCFVs(player);
                                for (int infoset_index = 0; infoset_index < 4; infoset_index++) {
                                    SPIEL_CHECK_FLOAT_EQ(reference_values[player][infoset_index][i],
                                                         CFVs.at(infoset_strings[player][infoset_index]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    open_spiel::papers_with_code::TestBasicCFVs();
}