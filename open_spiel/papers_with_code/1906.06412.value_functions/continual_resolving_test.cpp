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
#include "open_spiel/games/nfg_game.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#include "tabularize_bot.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/algorithms/best_response.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

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
                leaf_evaluator->save_values_policy = PolicySelection::kCurrentPolicy;


                auto subgame = std::make_shared<Subgame>(game, 1);
                auto subgame_solver = std::make_unique<SubgameSolver>(subgame, leaf_evaluator,
                                                                      terminal_evaluator, "RegretMatching",
                                                                      PolicySelection::kCurrentPolicy,
                                                                      true);

                // we do 5 iterations and check the CFVs after each iteration
                for (int i = 0; i < trunk_iterations; i++) {
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

            void TestKuhnGadget() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                auto subgame_factory = std::make_unique<SubgameFactory>();
                subgame_factory->game = game;
                subgame_factory->infostate_observer = game->MakeObserver(kInfoStateObsType, {});
                subgame_factory->public_observer = game->MakeObserver(kPublicStateObsType, {});
                subgame_factory->hand_observer = game->MakeObserver(kHandHistoryObsType, {});
                subgame_factory->max_move_ahead_limit = 1;
                subgame_factory->max_particles = 100;

                TabularPolicy optimal_policy = kuhn_poker::GetOptimalPolicy(0);

                auto subgame = subgame_factory->MakeTrunk(3);

                std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
                        MakeTerminalEvaluator();
                std::shared_ptr<const PublicStateEvaluator> dummy_evaluator =
                        MakeDummyEvaluator();
                std::shared_ptr<Observer> public_observer =
                        game->MakeObserver(kPublicStateObsType, {});
                std::shared_ptr<Observer> infostate_observer =
                        game->MakeObserver(kInfoStateObsType, {});

                std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
                        std::make_shared<CFREvaluator>(game, 20, dummy_evaluator, terminal_evaluator,
                                                       public_observer, infostate_observer, 100000);

                auto solver = std::make_unique<SubgameSolver>(subgame, nonterminal_evaluator,
                                                              terminal_evaluator, "FixableStrategy",
                                                              PolicySelection::kAveragePolicy, true);


                for (int player = 0; player < 2; player++) {
                    algorithms::BanditVector &bandits = solver->bandits()[player];
                    for (algorithms::DecisionId id : bandits.range()) {
                        algorithms::InfostateNode *node = subgame->trees[player]->decision_infostate(id);
                        ActionsAndProbs infostate_policy = optimal_policy.GetStatePolicy(node->infostate_string());
                        std::vector<double> probs = GetProbs(infostate_policy);
                        auto fixable_bandit = std::make_unique<algorithms::bandits::FixableStrategy>(probs);
                        bandits[id] = std::move(fixable_bandit);
                    }
                }

                solver->RunSimultaneousIterations(1);

                for (auto &public_state : subgame->public_states) {
                    if (public_state.IsLeaf() && public_state.nodes[0][0]->infostate_string().substr(1) == "p") {

                        std::mt19937 rnd_gen(0);
                        std::unique_ptr<ParticleSetPartition> partition = MakeParticleSetPartition(public_state,
                                                                                                   pow(10, 7),
                                                                                                   pow(10, -9), false,
                                                                                                   rnd_gen);
                        std::unique_ptr<ParticleSet> set = std::make_unique<ParticleSet>(partition->primary);
                        for (int player = 0; player < 2; player++) {
                            auto local_subgame = subgame_factory->MakeSubgameSafeResolving(*set, player,
                                                                                           public_state.GetCFVs(
                                                                                                   1 - player),
                                                                                           20);

                            SequenceFormLpSpecification specification(local_subgame->trees);
                            specification.SpecifyLinearProgram(player);
                            double game_value = specification.Solve();
                            TabularPolicy policy = specification.OptimalPolicy(player);
                            SPIEL_CHECK_FLOAT_NEAR(game_value, player == 0 ? -1. / 18 : 1. / 18, 0.001);

                            SPIEL_CHECK_EQ(policy.PolicyTable().size(), 3);

                            for (const auto &entry : policy.PolicyTable()) {
                                std::vector<double> slp_state_policy = GetProbs(entry.second);
                                std::vector<double> opt_state_policy = GetProbs(
                                        optimal_policy.GetStatePolicy(entry.first));
                                SPIEL_CHECK_EQ(slp_state_policy.size(), 2);
                                for (int action_index = 0; action_index < slp_state_policy.size(); action_index++) {
                                    SPIEL_CHECK_FLOAT_NEAR(slp_state_policy[action_index],
                                                           opt_state_policy[action_index], 0.0015);
                                }
                            }
                        }
                    }
                }
            }

            void TestKuhnExploitability() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                BotParameters params{
                        {"seed",                   BotParameter(0)},
                        {"cfr_iterations",         BotParameter(1000)},
                        {"max_move_ahead_limit",   BotParameter(1)},
                        {"max_particles",          BotParameter(1000)},
                        {"use_bandits_for_cfr",    BotParameter("RegretMatchingPlus")},
                        {"save_values_policy",     BotParameter("average")},
                        {"non_terminal_evaluator", BotParameter("cfr")},
                        {"subgame_cfr_iterations", BotParameter(10)},
                };

                SherlockBotFactory bot_factory = SherlockBotFactory();

                std::unique_ptr<SherlockBot> bot_player_one = bot_factory.Create(game, Player(0),
                                                                                 params, true);
                std::shared_ptr<TabularPolicy> bot_policy_player_one = tabularize_bot::FullBotPolicy(
                        std::move(bot_player_one), Player(0), game);

                std::unique_ptr<SherlockBot> bot_player_two = bot_factory.Create(game, Player(1),
                                                                                 params, true);
                std::shared_ptr<TabularPolicy> bot_policy_player_two = tabularize_bot::FullBotPolicy(
                        std::move(bot_player_two), Player(1), game);

                std::unique_ptr<State> root = game->NewInitialState();
                algorithms::TabularBestResponse best_response_one(*game, 1, bot_policy_player_one->PolicyTable());
                std::cout << best_response_one.Value(*root) << "\n";

                algorithms::TabularBestResponse best_response_two(*game, 0, bot_policy_player_two->PolicyTable());
                std::cout << best_response_two.Value(*root) << "\n";

                bot_policy_player_two->ImportPolicy(*bot_policy_player_one);

                std::cout << algorithms::Exploitability(*game, *bot_policy_player_two) << "\n";
            }

            void RPSCRTest() {
                const char *kSampleNFGString = R"###(
                        NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
                        { "Player 1" "Player 2" } { 3 3 }

                        0 0   1 -1   -1 1   -1 1   0 0   1 -1   1 -1   -1 1   0 0
                )###";

                std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(kSampleNFGString);
                game = ConvertToTurnBased(*game);

                std::unique_ptr<State> state = game->NewInitialState();

                std::unordered_map<std::string, double> CFVs;
                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                for (Action action : state->LegalActions()) {
                    std::unique_ptr<State> child = state->Child(action);
                    CFVs.emplace(child->InformationStateString(0), 0);
                    infoset_one = child->InformationStateString(1);
                    start_states.push_back(std::move(child));
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                   {1. / 3, 1. / 3, 1. / 3},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 0,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {1. / 3, 1. / 3, 1. / 3},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 0, 10);

                std::string tree_safe_player_reference = "(((({-0.00})({-0.00})({-0.00}))[({-1.00}{0.00}{1.00})({-1.00}{0.00}{1.00})({-1.00}{0.00}{1.00})]))";
                std::string tree_safe_opponent_reference = "([(({-1.00})({0.00})({1.00}))(({0.00}))][(({-1.00})({0.00})({1.00}))(({0.00}))][(({-1.00})({0.00})({1.00}))(({0.00}))])";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});
                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(0., specification.Solve());

                for (float prob : GetProbs(specification.OptimalPolicy(1).GetStatePolicy(infoset_one))) {
                    SPIEL_CHECK_FLOAT_EQ(prob, 1. / 3);
                }
            }

            void BRPSCRTest() {
                const char *kSampleNFGString = R"###(
                        NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
                        { "Player 1" "Player 2" } { 3 3 }

                        0 0   1 -1   -2 2   -1 1   0 0   3 -3   2 -2   -3 3   0 0
                )###";

                std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(kSampleNFGString);
                game = ConvertToTurnBased(*game);

                std::unique_ptr<State> state = game->NewInitialState();

                std::unordered_map<std::string, double> CFVs;
                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                for (Action action : state->LegalActions()) {
                    std::unique_ptr<State> child = state->Child(action);
                    CFVs.emplace(child->InformationStateString(0), 0);
                    infoset_one = child->InformationStateString(1);
                    start_states.push_back(std::move(child));
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states, {0.5, 1. / 3., 1. / 6},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 0,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {0.5, 1. / 3., 1. / 6},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 0, 10);

                std::string tree_safe_player_reference = "(((({-0.00})({-0.00})({-0.00}))[({-1.00}{0.00}{2.00})({-2.00}{0.00}{3.00})({-3.00}{0.00}{1.00})]))";
                std::string tree_safe_opponent_reference = "([(({-1.00})({0.00})({2.00}))(({0.00}))][(({-2.00})({0.00})({3.00}))(({0.00}))][(({-3.00})({0.00})({1.00}))(({0.00}))])";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});
                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(0., specification.Solve());

                std::vector<float> expected_policy = {0.5, 1. / 3, 1. / 6};
                auto solved_policy = GetProbs(specification.OptimalPolicy(1).GetStatePolicy(infoset_one));
                for (int i = 0; i < expected_policy.size(); i++) {
                    SPIEL_CHECK_FLOAT_EQ(solved_policy[i], expected_policy[i]);
                }
            }

            void SmallerEqTest() {
                const char *kSampleNFGString = R"###(
                        NFG 1 R "Selten (IJGT, 75), Figure 2, normal form"
                        { "Player 1" "Player 2" } { 3 3 }

                        1 -1   0 0   0 0   0 0   1 -1   0 0   1 -1   1 -1   -5 5
                )###";

                std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(kSampleNFGString);
                game = ConvertToTurnBased(*game);

                std::unique_ptr<State> state = game->NewInitialState();

                std::unordered_map<std::string, double> CFVs;
                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                std::vector<float> prepared_CFVs = {0.5, 0.5, 0};
                auto legal_actions = state->LegalActions();
                for (int i = 0; i < prepared_CFVs.size(); i++) {
                    std::unique_ptr<State> child = state->Child(legal_actions[i]);
                    CFVs.emplace(child->InformationStateString(0), prepared_CFVs[i]);
                    infoset_one = child->InformationStateString(1);
                    start_states.push_back(std::move(child));
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states, {0.5, 0.5, 0},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 0,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states, {0.5, 0.5, 0},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 0, 10);

                std::string tree_safe_player_reference = "(((({-0.00})({-0.50})({-0.50}))[({-1.00}{-1.00}{5.00})({-1.00}{0.00}{0.00})({-1.00}{0.00}{0.00})]))";
                std::string tree_safe_opponent_reference = "([(({-5.00})({0.00})({0.00}))(({0.00}))][(({0.00})({1.00})({1.00}))(({0.50}))][(({0.00})({1.00})({1.00}))(({0.50}))])";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});
                specification.SpecifyLinearProgram(1);

                SPIEL_CHECK_FLOAT_EQ(-0.5, specification.Solve());

                std::vector<float> expected_policy = {0.5, 0.5, 0};
                auto solved_policy = GetProbs(specification.OptimalPolicy(1).GetStatePolicy(infoset_one));
                for (int i = 0; i < expected_policy.size(); i++) {
                    SPIEL_CHECK_FLOAT_EQ(solved_policy[i], expected_policy[i]);
                }
            }

            void KuhnCheckSituation() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                std::unique_ptr<State> state = game->NewInitialState();

                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                std::unordered_map<std::string, double> CFVs = {{"0p", -1.},
                                                                {"1p", -1. / 3},
                                                                {"2p", 7. / 6}};
                for (Action action_first_chance_node : state->LegalActions()) {
                    std::unique_ptr<State> child_first_chance_node = state->Child(action_first_chance_node);
                    for (Action action_second_chance_node : child_first_chance_node->LegalActions()) {
                        std::unique_ptr<State> child_player_one_node = child_first_chance_node->Child(
                                action_second_chance_node);
                        std::unique_ptr<State> check_state_player_two = child_player_one_node->Child(Action(0));
                        start_states.push_back(std::move(check_state_player_two));
                    }
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                   {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                    1. / 6, 1. / 6},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 0,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                      1. / 6, 1. / 6},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 0, 10);

                std::string tree_safe_player_reference = "((((({-1.17}))(({0.33})))[(({-1.00})({-1.00}))(({-2.00}{-2.00}{1.00}{1.00}))])(((({-1.17}))(({1.00})))[(({-1.00})({1.00}))(({-2.00}{1.00}{1.00}{2.00}))])(((({0.33}))(({1.00})))[(({1.00})({1.00}))(({1.00}{1.00}{2.00}{2.00}))]))";
                std::string tree_safe_opponent_reference = "([((({-0.33}))(({-0.33})))((({-1.00}))(({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])][((({-1.00}))(({-1.00})))((({-1.00}))(({-1.00}))[({-1.00}{-1.00})({-2.00}{-2.00})])][((({1.00}))(({1.00}))[({-1.00}{-1.00})({2.00}{2.00})])((({1.17}))(({1.17})))])";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});

                specification.SpecifyLinearProgram(0);
                SPIEL_CHECK_FLOAT_EQ(-1. / 18, specification.Solve());

                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(1. / 18, specification.Solve());

                TabularPolicy solved_slp_policy = specification.OptimalPolicy(1);

                TabularPolicy kuhn_optimal_policy = kuhn_poker::GetOptimalPolicy(0);

                SPIEL_CHECK_EQ(solved_slp_policy.PolicyTable().size(), 3);

                for (const auto &info_state_strategy : solved_slp_policy.PolicyTable()) {
                    SPIEL_CHECK_EQ(GetProbs(info_state_strategy.second).size(), 2);
                    ActionsAndProbs state_optimal_policy = kuhn_optimal_policy.GetStatePolicy(
                            info_state_strategy.first);
                    for (int i = 0; i < 2; i++) {
                        SPIEL_CHECK_FLOAT_EQ(GetProbs(state_optimal_policy)[i],
                                             GetProbs(info_state_strategy.second)[i]);
                    }
                }
            }

            void KuhnBetSituation() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                std::unique_ptr<State> state = game->NewInitialState();

                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                std::unordered_map<std::string, double> CFVs = {{"0b", -1.},
                                                                {"1b", -1. / 2},
                                                                {"2b", 7. / 6}};
                for (Action action_first_chance_node : state->LegalActions()) {
                    std::unique_ptr<State> child_first_chance_node = state->Child(action_first_chance_node);
                    for (Action action_second_chance_node : child_first_chance_node->LegalActions()) {
                        std::unique_ptr<State> child_player_one_node = child_first_chance_node->Child(
                                action_second_chance_node);
                        std::unique_ptr<State> check_state_player_two = child_player_one_node->Child(Action(1));
                        start_states.push_back(std::move(check_state_player_two));
                    }
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                   {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                    1. / 6, 1. / 6},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 0,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                      1. / 6, 1. / 6},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 0, 10);

                std::string tree_safe_player_reference = "(((({-1.17})({0.50}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.17})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])((({0.50})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})]))";
                std::string tree_safe_opponent_reference = "([(({-0.50})({-0.50}))(({-2.00})({1.00})({1.00})({2.00}))][(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.17})({1.17}))])";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});

                specification.SpecifyLinearProgram(0);
                SPIEL_CHECK_FLOAT_EQ(-1. / 9, specification.Solve());

                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(1. / 9, specification.Solve());

                TabularPolicy solved_slp_policy = specification.OptimalPolicy(1);

                TabularPolicy kuhn_optimal_policy = kuhn_poker::GetOptimalPolicy(0);

                SPIEL_CHECK_EQ(solved_slp_policy.PolicyTable().size(), 3);

                for (const auto &info_state_strategy : solved_slp_policy.PolicyTable()) {
                    SPIEL_CHECK_EQ(GetProbs(info_state_strategy.second).size(), 2);
                    ActionsAndProbs state_optimal_policy = kuhn_optimal_policy.GetStatePolicy(
                            info_state_strategy.first);
                    for (int i = 0; i < 2; i++) {
                        SPIEL_CHECK_FLOAT_EQ(GetProbs(state_optimal_policy)[i],
                                             GetProbs(info_state_strategy.second)[i]);
                    }
                }
            }

            void KuhnLastPublicSituationAlphaMin() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                std::unique_ptr<State> state = game->NewInitialState();

                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                std::unordered_map<std::string, double> CFVs = {{"0pb", -1.},
                                                                {"1pb", -1. / 2},
                                                                {"2pb", 7. / 6}};
                for (Action action_first_chance_node : state->LegalActions()) {
                    std::unique_ptr<State> child_first_chance_node = state->Child(action_first_chance_node);
                    for (Action action_second_chance_node : child_first_chance_node->LegalActions()) {
                        std::unique_ptr<State> child_player_one_node = child_first_chance_node->Child(
                                action_second_chance_node);
                        std::unique_ptr<State> check_state_player_two = child_player_one_node->Child(Action(0))->Child(
                                Action(1));
                        start_states.push_back(std::move(check_state_player_two));
                    }
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                   {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                    1. / 6, 1. / 6},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 1,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {1. / 6, 1. / 6, 1. / 6, 1. / 6,
                                                                                      1. / 6, 1. / 6},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 1, 10);

                std::string tree_safe_player_reference = "([(({-0.50})({-0.50}))(({-2.00})({1.00})({1.00})({2.00}))][(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.17})({1.17}))])";
                std::string tree_safe_opponent_reference = "(((({-1.17})({0.50}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.17})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])((({0.50})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})]))";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});

                specification.SpecifyLinearProgram(0);
                SPIEL_CHECK_FLOAT_EQ(1. / 9, specification.Solve());

                TabularPolicy solved_slp_policy = specification.OptimalPolicy(0);

                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(-1. / 9, specification.Solve());

                TabularPolicy kuhn_optimal_policy = kuhn_poker::GetOptimalPolicy(0);

                SPIEL_CHECK_EQ(solved_slp_policy.PolicyTable().size(), 3);

                for (const auto &info_state_strategy : solved_slp_policy.PolicyTable()) {
                    SPIEL_CHECK_EQ(GetProbs(info_state_strategy.second).size(), 2);
                    ActionsAndProbs state_optimal_policy = kuhn_optimal_policy.GetStatePolicy(
                            info_state_strategy.first);
                    for (int i = 0; i < 2; i++) {
                        SPIEL_CHECK_FLOAT_EQ(GetProbs(state_optimal_policy)[i],
                                             GetProbs(info_state_strategy.second)[i]);
                    }
                }
            }

            void KuhnLastPublicSituationAlphaMax() {
                std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

                std::unique_ptr<State> state = game->NewInitialState();

                std::vector<std::unique_ptr<State>> start_states;

                std::string infoset_one;
                std::unordered_map<std::string, double> CFVs = {{"0pb", -1.},
                                                                {"1pb", 1},
                                                                {"2pb", 7. / 5}};
                for (Action action_first_chance_node : state->LegalActions()) {
                    std::unique_ptr<State> child_first_chance_node = state->Child(action_first_chance_node);
                    for (Action action_second_chance_node : child_first_chance_node->LegalActions()) {
                        std::unique_ptr<State> child_player_one_node = child_first_chance_node->Child(
                                action_second_chance_node);
                        std::unique_ptr<State> check_state_player_two = child_player_one_node->Child(Action(0))->Child(
                                Action(1));
                        start_states.push_back(std::move(check_state_player_two));
                    }
                }

                auto tree_safe_player = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                   {1. / 9, 1. / 9, 1. / 6, 1. / 6, 0,
                                                                                    0},
                                                                                   game->MakeObserver(kInfoStateObsType,
                                                                                                      {}), 1, CFVs, 1,
                                                                                   10);

                auto tree_safe_opponent = algorithms::MakeInfostateTreeSafeResolving(start_states,
                                                                                     {1. / 9, 1. / 9, 1. / 6, 1. / 6, 0,
                                                                                      0},
                                                                                     game->MakeObserver(
                                                                                             kInfoStateObsType, {}), 0,
                                                                                     CFVs, 1, 10);

                std::string tree_safe_player_reference = "([(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({-2.00})({1.00})({1.00})({2.00}))(({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.40})({1.40}))])";
                std::string tree_safe_opponent_reference = "(((({-1.00})({-1.40}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.00})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})])((({-1.40})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})]))";
                SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2), tree_safe_player_reference);
                SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2), tree_safe_opponent_reference);

                SequenceFormLpSpecification specification({tree_safe_opponent, tree_safe_player});

                specification.SpecifyLinearProgram(0);
                SPIEL_CHECK_FLOAT_EQ(-1. / 3, specification.Solve());

                TabularPolicy solved_slp_policy = specification.OptimalPolicy(0);

                specification.SpecifyLinearProgram(1);
                SPIEL_CHECK_FLOAT_EQ(1. / 3, specification.Solve());

                TabularPolicy kuhn_optimal_policy = kuhn_poker::GetOptimalPolicy(1. / 3);

                SPIEL_CHECK_EQ(solved_slp_policy.PolicyTable().size(), 3);

                for (const auto &info_state_strategy : solved_slp_policy.PolicyTable()) {
                    SPIEL_CHECK_EQ(GetProbs(info_state_strategy.second).size(), 2);
                    ActionsAndProbs state_optimal_policy = kuhn_optimal_policy.GetStatePolicy(
                            info_state_strategy.first);
                    for (int i = 0; i < 2; i++) {
                        SPIEL_CHECK_FLOAT_EQ(GetProbs(state_optimal_policy)[i],
                                             GetProbs(info_state_strategy.second)[i]);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    // Test automatic CFV extraction on simple matrix game
    open_spiel::papers_with_code::TestBasicCFVs();
    // Creates fixed trunk of Kuhn automatically retrieves CFVs, constructs
    // a sub-game, solves it and checks if the results are close to optimal
    open_spiel::papers_with_code::TestKuhnGadget();
//    open_spiel::papers_with_code::TestKuhnExploitability();
    // Tests on matrix game for correct gadget game generations and resolving
    open_spiel::papers_with_code::RPSCRTest();
    open_spiel::papers_with_code::BRPSCRTest();
    open_spiel::papers_with_code::SmallerEqTest();
    // Tests for correctly resolved Gadget game on kuhn with all the values handcrafted beforehand
    open_spiel::papers_with_code::KuhnCheckSituation();
    open_spiel::papers_with_code::KuhnBetSituation();
    open_spiel::papers_with_code::KuhnLastPublicSituationAlphaMin();
    open_spiel::papers_with_code::KuhnLastPublicSituationAlphaMax();
}