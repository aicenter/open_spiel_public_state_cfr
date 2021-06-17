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

// Tests if CFVs saved in the leaf public states are correct (used in the
// continual resolving), by emulating CFR iterations and comparing with
// reference average values.
void TestBasicCFVs() {
  // Constructs biased matching pennies.
  const char* kSampleNFGString = R"###(
      NFG 1 R ""
      { "Player 1" "Player 2" } { 2 2 }

      1 -1 0 0 0 0 2 -2
  )###";

  std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(kSampleNFGString);
  const int trunk_iterations = 5;

  // Prepared infostate values for the test.
  std::string infostate_strings[2][4] = {
      {
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
      }
  };

  // Player - infostate index - trunk iteration number.
  double reference_values[2][4][5] = {
      {
          // Payoff 2
          {1, 0.5, 1. / 3, 0.25, 0.2},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          // Payoff 1
          {0.5, 0.75, 5. / 6, 0.875, 0.9}
      },
      {
          // Payoff -2
          {-1, -1.5, -7. / 6, -0.875, -0.7},
          {0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0},
          // Payoff -1
          {-0.5, -0.25, -5. / 12, -0.5625, -0.65}
      }
  };

  // Create a subgame solver that is essentially just CFR.
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

  auto subgame = std::make_shared<Subgame>(game, /*max_moves=*/1);
  auto subgame_solver = std::make_unique<SubgameSolver>(
      subgame, leaf_evaluator, terminal_evaluator, "RegretMatching",
      PolicySelection::kCurrentPolicy, /*safe_resolving=*/true);

  // We do 5 iterations and check the CFVs after each iteration.
  for (int i = 0; i < trunk_iterations; i++) {
    subgame_solver->RunSimultaneousIterations(1);
    for (const auto& public_state : subgame->public_states) {
      if (!public_state.IsTerminal()) continue;

      for (int player = 0; player < 2; player++) {
        auto CFVs = public_state.InfostateAvgValues(player);
        for (int is_index = 0; is_index < 4; is_index++) {
          SPIEL_CHECK_FLOAT_EQ(reference_values[player][is_index][i],
                               CFVs.at(infostate_strings[player][is_index]));
        }
      }
    }
  }
}

// Creates a fixed trunk for Kuhn with optimal policy, automatically retrieves
// CFVs for such trunk, constructs a sub-game, solves it and checks if the
// results are close to optimal.
void TestKuhnGadget() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  auto subgame_factory = std::make_unique<SubgameFactory>();
  subgame_factory->game = game;
  subgame_factory->infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  subgame_factory->public_observer = game->MakeObserver(kPublicStateObsType, {});
  subgame_factory->hand_observer = game->MakeObserver(kHandHistoryObsType, {});
  subgame_factory->max_move_ahead_limit = 1;
  subgame_factory->max_particles = 100;

  auto subgame = subgame_factory->MakeTrunk(3);
  auto solver = std::make_unique<SubgameSolver>(
      subgame, MakeApproxOracleEvaluator(game), MakeTerminalEvaluator(),
      "FixableStrategy", PolicySelection::kAveragePolicy, true);

  TabularPolicy optimal_policy = kuhn_poker::GetOptimalPolicy(/*alpha=*/0);

  // Fix trunk strategies with optimal policy.
  for (int player = 0; player < 2; player++) {
    algorithms::BanditVector& bandits = solver->bandits()[player];
    for (algorithms::DecisionId id : bandits.range()) {
      algorithms::InfostateNode* node =
          subgame->trees[player]->decision_infostate(id);
      ActionsAndProbs infostate_policy =
          optimal_policy.GetStatePolicy(node->infostate_string());
      std::vector<double> probs = GetProbs(infostate_policy);
      auto fixable_bandit =
          std::make_unique<algorithms::bandits::FixableStrategy>(probs);
      bandits[id] = std::move(fixable_bandit);
    }
  }

  // Compute cf. values in each public state.
  solver->RunSimultaneousIterations(1);

  // Compare resulting policies and game values.
  // FIXME(David): why only pass?
  for (const PublicState& public_state : subgame->public_states) {
    if (public_state.IsLeaf()
        && public_state.nodes[0][0]->infostate_string().substr(1) == "p") {
      std::unique_ptr<ParticleSet> set = ParticlesFromState(public_state);

      for (int player = 0; player < 2; player++) {
        auto local_subgame = subgame_factory->MakeSubgameSafeResolving(
            *set, player, public_state.InfostateAvgValues(1 - player),
            algorithms::kNoMoveAheadLimit);

        SequenceFormLpSpecification specification(local_subgame->trees);
        specification.SpecifyLinearProgram(player);

        double game_value = specification.Solve();
        SPIEL_CHECK_FLOAT_NEAR(game_value,
                               player == 0 ? -1. / 18 : 1. / 18,
                               0.001);

        TabularPolicy policy = specification.OptimalPolicy(player);
        SPIEL_CHECK_EQ(policy.PolicyTable().size(), 3);

        for (const auto&[infostate, actions_and_probs] : policy.PolicyTable()) {
          std::vector<double> slp_state_policy = GetProbs(actions_and_probs);
          std::vector<double> opt_state_policy = GetProbs(
              optimal_policy.GetStatePolicy(infostate));
          SPIEL_CHECK_EQ(slp_state_policy.size(), 2);

          for (int action_index = 0; action_index < slp_state_policy.size();
               action_index++) {
            SPIEL_CHECK_FLOAT_NEAR(slp_state_policy[action_index],
                                   opt_state_policy[action_index], 0.0015);
          }
        }
      }
    }
  }
}

// Compute exploitability of the bot.
void KuhnExploitabilityBenchmark() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  BotParameters params{
      {"seed", BotParameter(0)},
      {"cfr_iterations", BotParameter(1000)},
      {"max_move_ahead_limit", BotParameter(1)},
      {"max_particles", BotParameter(1000)},
      {"use_bandits_for_cfr", BotParameter("RegretMatchingPlus")},
      {"save_values_policy", BotParameter("average")},
      {"non_terminal_evaluator", BotParameter("cfr")},
      {"subgame_cfr_iterations", BotParameter(10)},
  };

  SherlockBotFactory bot_factory = SherlockBotFactory();
  std::unique_ptr<State> root = game->NewInitialState();

  TabularPolicy full_policy;
  for (int pl = 0; pl < 2; ++pl) {
    std::unique_ptr<Bot> bot = bot_factory.Create(game, pl, params);
    std::shared_ptr<TabularPolicy> player_policy =
        tabularize_bot::FullBotPolicy(std::move(bot), pl, *game);
    algorithms::TabularBestResponse best_response(
        *game, 1-pl, player_policy->PolicyTable());
    std::cout << "BR against PL" << pl << ": "
              << best_response.Value(*root) << "\n";
    full_policy.ImportPolicy(*player_policy);
  }
  std::cout << "Expl: "
            << algorithms::Exploitability(*game, full_policy) << "\n";
}

void CheckNfgGame(const std::string& nfg_string,
                  const std::vector<double>& constraints_values,
                  const std::vector<double>& chance_reach_probs,
                  const std::vector<double>& expected_policy,
                  double expected_game_value,
                  std::string expected_player_certificate,
                  std::string expected_opponent_certificate) {
  std::shared_ptr<const Game> game = nfg_game::LoadNFGGame(nfg_string);
  game = ConvertToTurnBased(*game);
  std::unique_ptr<State> root_state = game->NewInitialState();
  auto infostate_observer = game->MakeObserver(kInfoStateObsType, {});

  std::unordered_map<std::string, double> CFVs;
  std::vector<std::unique_ptr<State>> start_states;

  std::string infoset_one;
  auto legal_actions = root_state->LegalActions();
  SPIEL_CHECK_EQ(constraints_values.size(), legal_actions.size());
  for (int i = 0; i < constraints_values.size(); i++) {
    std::unique_ptr<State> child = root_state->Child(legal_actions[i]);
    CFVs.emplace(child->InformationStateString(0), constraints_values[i]);
    infoset_one = child->InformationStateString(1);
    start_states.push_back(std::move(child));
  }

  auto trees = algorithms::MakeResolvingInfostateTrees(
      start_states, chance_reach_probs, infostate_observer,
      /*ft_player=*/0, CFVs);
  auto tree_safe_opponent = trees[0];
  auto tree_safe_player = trees[1];

  SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2),
                 expected_player_certificate);
  SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2),
                 expected_opponent_certificate);

  SequenceFormLpSpecification lp_spec(trees);
  lp_spec.SpecifyLinearProgram(Player{1});
  SPIEL_CHECK_FLOAT_EQ(expected_game_value, lp_spec.Solve());

  auto solved_policy =
      GetProbs(lp_spec.OptimalPolicy(1).GetStatePolicy(infoset_one));
  for (int i = 0; i < expected_policy.size(); i++) {
    SPIEL_CHECK_FLOAT_EQ(solved_policy[i], expected_policy[i]);
  }
}

void TestResolvingOnRPS() {
  CheckNfgGame(
      R"###(
      NFG 1 R ""
      { "Player 1" "Player 2" } { 3 3 }

      0 0   1 -1   -1 1   -1 1   0 0   1 -1   1 -1   -1 1   0 0
    )###",
      /*constraints_values=*/ {0., 0., 0.},
      /*chance_reach_probs=*/ {1. / 3, 1. / 3, 1. / 3},
      /*expected_policy=*/    {1. / 3, 1. / 3, 1. / 3},
      /*expected_game_value=*/ 0.,
      /*expected_player_certificate=*/"(("
                                      // Observation of constaints, without the
                                      // ability to change the outcome.
                                      "(({-0.00})({-0.00})({-0.00}))"
                                      // Standard RPS outcomes.
                                      "[({-1.00}{0.00}{1.00})"
                                      "({-1.00}{0.00}{1.00})"
                                      "({-1.00}{0.00}{1.00})]"
                                      "))",
      /*expected_opponent_certificat=*/"("
                                       // Follow / terminate for each action,
                                       // with standard RPS outcomes.
                                       "[(({-1.00})({0.00})({1.00}))(({0.00}))]"
                                       "[(({-1.00})({0.00})({1.00}))(({0.00}))]"
                                       "[(({-1.00})({0.00})({1.00}))(({0.00}))]"
                                       ")");

}

void TestResolvingOnBiasedRPS() {
  CheckNfgGame(
      R"###(
      NFG 1 R ""
      { "Player 1" "Player 2" } { 3 3 }

      0 0   1 -1   -2 2   -1 1   0 0   3 -3   2 -2   -3 3   0 0
    )###",
      /*constraints_values=*/ {0., 0., 0.},
      /*chance_reach_probs=*/ {0.5, 1. / 3., 1. / 6},
      /*expected_policy=*/    {0.5, 1. / 3, 1. / 6},
      /*expected_game_value=*/ 0.,
      /*expected_player_certificate=*/"(("
                                      "(({-0.00})({-0.00})({-0.00}))"
                                      "[({-1.00}{0.00}{2.00})({-2.00}{0.00}{3.00})({-3.00}{0.00}{1.00})]"
                                      "))",
      /*expected_opponent_certificat=*/"("
                                       "[(({-1.00})({0.00})({2.00}))(({0.00}))]"
                                       "[(({-2.00})({0.00})({3.00}))(({0.00}))]"
                                       "[(({-3.00})({0.00})({1.00}))(({0.00}))]"
                                       ")");
}

void TestResolvingNfgWithSmallerEqSupport() {
  CheckNfgGame(
      R"###(
      NFG 1 R ""
      { "Player 1" "Player 2" } { 3 3 }

      1 -1   0 0   0 0   0 0   1 -1   0 0   1 -1   1 -1   -5 5
    )###",
      /*constraints_values=*/ {0.5, 0.5, 0},
      /*chance_reach_probs=*/ {0.5, 0.5, 0},
      /*expected_policy=*/    {0.5, 0.5, 0},
      /*expected_game_value=*/-0.5,
      /*expected_player_certificate=*/"(("
                                      "(({-0.00})({-0.50})({-0.50}))"
                                      "[({-1.00}{-1.00}{5.00})({-1.00}{0.00}{0.00})({-1.00}{0.00}{0.00})]"
                                      "))",
      /*expected_opponent_certificat=*/"("
                                       "[(({-5.00})({0.00})({0.00}))(({0.00}))]"
                                       "[(({0.00})({1.00})({1.00}))(({0.50}))]"
                                       "[(({0.00})({1.00})({1.00}))(({0.50}))]"
                                       ")");
}

void CheckKuhnGame(std::vector<std::vector<Action>> start_histories,
                   const std::unordered_map<std::string, double>& constraints_values,
                   const std::vector<double>& chance_reach_probs,
                   double kuhn_alpha_param,
                   Player ft_player,
                   const std::vector<double>& expected_game_values,
                   std::string expected_player_certificate,
                   std::string expected_opponent_certificate) {

  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  auto infostate_observer = game->MakeObserver(kInfoStateObsType, {});

  std::vector<std::unique_ptr<State>> start_states;
  for (auto& history : start_histories) {
    std::unique_ptr<State> s = game->NewInitialState();
    for (Action a: history) s->ApplyAction(a);
    start_states.push_back(std::move(s));
  }

  auto trees = algorithms::MakeResolvingInfostateTrees(
      start_states, chance_reach_probs, infostate_observer,
      ft_player, constraints_values);
  auto tree_safe_opponent = trees[ft_player];
  auto tree_safe_player = trees[1 - ft_player];

  SPIEL_CHECK_EQ(tree_safe_player->root().MakeCertificate(2),
                 expected_player_certificate);
  SPIEL_CHECK_EQ(tree_safe_opponent->root().MakeCertificate(2),
                 expected_opponent_certificate);

  SequenceFormLpSpecification specification(trees);
  TabularPolicy solved_slp_policy;
  for (int pl = 0; pl < 2; ++pl) {
    specification.SpecifyLinearProgram(pl);
    SPIEL_CHECK_FLOAT_EQ(expected_game_values[pl], specification.Solve());
    if (pl == 1 - ft_player) {
      solved_slp_policy = specification.OptimalPolicy(1 - ft_player);
    }
  }

  TabularPolicy kuhn_optimal_policy =
      kuhn_poker::GetOptimalPolicy(kuhn_alpha_param);
  SPIEL_CHECK_EQ(solved_slp_policy.PolicyTable().size(), 3);

  for (const auto&[infostate, strategy] : solved_slp_policy.PolicyTable()) {
    SPIEL_CHECK_EQ(strategy.size(), 2);
    ActionsAndProbs state_optimal_policy =
        kuhn_optimal_policy.GetStatePolicy(infostate);
    for (int i = 0; i < 2; i++) {
      SPIEL_CHECK_FLOAT_EQ(GetProbs(state_optimal_policy)[i],
                           GetProbs(strategy)[i]);
    }
  }
}

void TestResolvingKuhnPass() {
  CheckKuhnGame(
    /*start_histories=*/{{0,1,0}, {0,2,0}, {1,0,0}, {1,2,0}, {2,0,0}, {2,1,0}},
    /*constraints_values=*/{{"0p", -1.}, {"1p", -1. / 3}, {"2p", 7. / 6}},
    /*chance_reach_probs=*/{1./6, 1./6, 1./6, 1./6, 1./6, 1./6},
    /*kuhn_alpha_param=*/0,
    /*ft_player=*/0,
    /*expected_game_values=*/{-1./18, 1./18},
    /*expected_player_certificate=*/"((((({-1.17}))(({0.33})))[(({-1.00})({-1.00}))(({-2.00}{-2.00}{1.00}{1.00}))])(((({-1.17}))(({1.00})))[(({-1.00})({1.00}))(({-2.00}{1.00}{1.00}{2.00}))])(((({0.33}))(({1.00})))[(({1.00})({1.00}))(({1.00}{1.00}{2.00}{2.00}))]))",
    /*expected_opponent_certificate=*/"([((({-0.33}))(({-0.33})))((({-1.00}))(({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])][((({-1.00}))(({-1.00})))((({-1.00}))(({-1.00}))[({-1.00}{-1.00})({-2.00}{-2.00})])][((({1.00}))(({1.00}))[({-1.00}{-1.00})({2.00}{2.00})])((({1.17}))(({1.17})))])");
}

void TestResolvingKuhnBet() {
  CheckKuhnGame(
    /*start_histories=*/{{0,1,1}, {0,2,1}, {1,0,1}, {1,2,1}, {2,0,1}, {2,1,1}},
    /*constraints_values=*/{{"0b", -1.}, {"1b", -1./2}, {"2b", 7./6}},
    /*chance_reach_probs=*/{1./6, 1./6, 1./6, 1./6, 1./6, 1./6},
    /*kuhn_alpha_param=*/0,
    /*ft_player=*/0,
    /*expected_game_values=*/{-1./9, 1./9},
    /*expected_player_certificate=*/"(((({-1.17})({0.50}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.17})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])((({0.50})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})]))",
    /*expected_opponent_certificate=*/"([(({-0.50})({-0.50}))(({-2.00})({1.00})({1.00})({2.00}))][(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.17})({1.17}))])");
}

void TestResolvingKuhnPassBetAlphaMin() {
  CheckKuhnGame(
      /*start_histories=*/{{0,1,0,1}, {0,2,0,1}, {1,0,0,1}, {1,2,0,1}, {2,0,0,1}, {2,1,0,1}},
      /*constraints_values=*/{{"0pb", -1.}, {"1pb", -1./2}, {"2pb", 7./6}},
      /*chance_reach_probs=*/{1./6, 1./6, 1./6, 1./6, 1./6, 1./6},
      /*kuhn_alpha_param=*/0,  // Min alpha
      /*ft_player=*/1,
      /*expected_game_values=*/{1./9, -1./9},
      /*expected_player_certificate=*/"(((({-1.17})({0.50}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.17})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})])((({0.50})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})]))",
      /*expected_opponent_certificate=*/"([(({-0.50})({-0.50}))(({-2.00})({1.00})({1.00})({2.00}))][(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.17})({1.17}))])");
}

void TestResolvingKuhnPassBetAlphaMax() {
  CheckKuhnGame(
      /*start_histories=*/{{0,1,0,1}, {0,2,0,1}, {1,0,0,1}, {1,2,0,1}, {2,0,0,1}, {2,1,0,1}},
      /*constraints_values=*/{{"0pb", -1.}, {"1pb", 1}, {"2pb", 7./5}},
      /*chance_reach_probs=*/{1./9, 1./9, 1./6, 1./6, 0., 0.},
      /*kuhn_alpha_param=*/1/3.,  // Max alpha
      /*ft_player=*/1,
      /*expected_game_values=*/{-1./3, 1./3},
      /*expected_player_certificate=*/"(((({-1.00})({-1.40}))[({-1.00}{-1.00})({-2.00}{-2.00})])((({-1.00})({1.00}))[({-1.00}{-1.00})({2.00}{2.00})])((({-1.40})({1.00}))[({-1.00}{-1.00})({-2.00}{2.00})]))",
      /*expected_opponent_certificate=*/"([(({-1.00})({-1.00}))(({-2.00})({-2.00})({1.00})({1.00}))][(({-2.00})({1.00})({1.00})({2.00}))(({1.00})({1.00}))][(({1.00})({1.00})({2.00})({2.00}))(({1.40})({1.40}))])");
}

}  // namespace
}  // papers_with_code
}  // open_spiel

int main(int argc, char** argv) {
  // Test automatic CFV extraction on a simple matrix game.
  open_spiel::papers_with_code::TestBasicCFVs();

  // Creates fixed trunk of Kuhn automatically retrieves CFVs, constructs
  // a sub-game, solves it and checks if the results are close to optimal.
  open_spiel::papers_with_code::TestKuhnGadget();
  // The evaluation takes a while to run.
  // open_spiel::papers_with_code::KuhnExploitabilityBenchmark();

  // Tests on matrix game for correct gadget game generations and resolving.
  open_spiel::papers_with_code::TestResolvingOnRPS();
  open_spiel::papers_with_code::TestResolvingOnBiasedRPS();
  open_spiel::papers_with_code::TestResolvingNfgWithSmallerEqSupport();

  // Tests for correctly resolved Gadget game on Kuhn with all the values
  // handcrafted beforehand.
  open_spiel::papers_with_code::TestResolvingKuhnPass();
  open_spiel::papers_with_code::TestResolvingKuhnBet();
  open_spiel::papers_with_code::TestResolvingKuhnPassBetAlphaMin();
  open_spiel::papers_with_code::TestResolvingKuhnPassBetAlphaMax();
}