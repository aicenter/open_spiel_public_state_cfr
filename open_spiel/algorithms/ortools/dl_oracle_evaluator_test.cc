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

#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"

#include <string>
#include <iostream>

#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {
namespace {

void MakeFixedBandit(BanditVector& vec, std::string infostate_string,
                     const std::vector<double>& policy) {
  DecisionId id = vec.tree()->DecisionIdFromInfostateString(infostate_string);
  if (!id.is_undefined()) {
    vec[id] = std::make_unique<bandits::FixedStrategy>(policy);
  }
}

std::vector<BanditVector> MakeKuhnParametricPolicy(
    dlcfr::DepthLimitedCFR* dl_solver, double a) {
  // Set strategy as described in https://en.wikipedia.org/wiki/Kuhn_poker

  // Player 0
  const std::shared_ptr<InfostateTree> tree0 = dl_solver->trees()[0];
  BanditVector vec0(tree0.get());
  MakeFixedBandit(vec0, "0"  ,  { 1. - a      , a          });
  MakeFixedBandit(vec0, "0pb",  { 1           , 0.         });
  MakeFixedBandit(vec0, "1"  ,  { 1.          , 0.         });
  MakeFixedBandit(vec0, "1pb",  { 2 / 3. - a  , a + 1 / 3. });
  MakeFixedBandit(vec0, "2"  ,  { 1. - 3. * a , 3. * a     });
  MakeFixedBandit(vec0, "2pb",  { 0.          , 1.         });

  // Player 1
  const std::shared_ptr<InfostateTree> tree1 = dl_solver->trees()[1];
  BanditVector vec1(tree1.get());
  MakeFixedBandit(vec1, "0p",  { 2 / 3. , 1 / 3. });
  MakeFixedBandit(vec1, "0b",  { 1      , 0.     });
  MakeFixedBandit(vec1, "1p",  { 1.     , 0.     });
  MakeFixedBandit(vec1, "1b",  { 2 / 3. , 1 / 3. });
  MakeFixedBandit(vec1, "2p",  { 0.     , 1.     });
  MakeFixedBandit(vec1, "2b",  { 0      , 1.     });

  std::vector<BanditVector> out;
  out.push_back(std::move(vec0));
  out.push_back(std::move(vec1));
  return out;
}

void TestOptimalValuesKuhnBettingPublicState() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  auto leaf_evaluator = std::make_shared<OracleEvaluator>(
      game, infostate_observer);
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  dlcfr::DepthLimitedCFR dl_solver(game, /*trunk_depth_limit=*/3,
                                   leaf_evaluator, terminal_evaluator);
  dlcfr::LeafPublicState& bet_state = dl_solver.public_leaves()[1];

  // Make sure there is no regression and infostates are properly arranged
  // as when writing this test.
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[0][0]->infostate_string(), "0b");
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[0][1]->infostate_string(), "1b");
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[0][2]->infostate_string(), "2b");
  const int PL0_J = 0;
  const int PL0_Q = 1;
  const int PL0_K = 2;
  // This does not follow an intuitive order, because of the way how the tree
  // is constructed: we recurse through dealing card 0 to player 0, and player 1
  // receives cards 1 or 2, so we also build those infostates first.
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[1][0]->infostate_string(), "1b");
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[1][1]->infostate_string(), "2b");
  SPIEL_CHECK_EQ(bet_state.leaf_nodes[1][2]->infostate_string(), "0b");
  const int PL1_J = 2;
  const int PL1_Q = 0;
  const int PL1_K = 1;

  // If the cf. values are computed with BR instead of CBR, the value function
  // does not return correct values for \alpha \in [0; 0.25)
  // Let's test various parametrizations to make sure that the values are indeed
  // correctly computed.
  std::vector<float> test_parametrizations = { 0., 0.1, 0.25, 0.5, 1. };

  for (float alpha : test_parametrizations) {
    bet_state.ranges = {
        // Player 0 bets with these probabilities when it has cards J, Q or K
        std::vector<double>({alpha, alpha, 1 - alpha}),
        // Player 1 did not act before this public state, so ranges are fixed.
        std::vector<double>({1., 1., 1.})
    };
    leaf_evaluator->EvaluatePublicState(&bet_state, /*context=*/nullptr);
    if (alpha < 0.25) {
      // PL1 passes
      SPIEL_CHECK_FLOAT_NEAR(bet_state.values[0][PL0_J], -1 / 6., 1e-10);
      SPIEL_CHECK_FLOAT_NEAR(bet_state.values[0][PL0_K], 1 / 3., 1e-10);
    } else {
      // PL1 bets
      SPIEL_CHECK_FLOAT_NEAR(bet_state.values[0][PL0_J], -2 / 3., 1e-10);
      SPIEL_CHECK_FLOAT_NEAR(bet_state.values[0][PL0_K], 1 / 2., 1e-10);
    }
    SPIEL_CHECK_FLOAT_NEAR(bet_state.values[0][PL0_Q], -1 / 6., 1e-10);

    SPIEL_CHECK_FLOAT_NEAR(bet_state.values[1][PL1_J], -1 / 6., 1e-6);
    SPIEL_CHECK_FLOAT_NEAR(bet_state.values[1][PL1_Q],
        std::fmax((2 * alpha - 1) / 3., -1/6.), 1e-6);
    SPIEL_CHECK_FLOAT_NEAR(bet_state.values[1][PL1_K], 2 * alpha / 3, 1e-6);
  }
}

void TestTrunkExploitabilityOptimalValuesKuhn() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  auto oracle_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);
  dlcfr::DepthLimitedCFR dl_solver(game, /*trunk_depth_limit=*/3,
                                   oracle_evaluator, terminal_evaluator);

  // Range of values that produce Nash (optimal).
  for (double a : std::vector<double>{0., 1 / 6., 1 / 3.,}) {
    std::vector<BanditVector> policy = MakeKuhnParametricPolicy(&dl_solver, a);
    double actual_exploitability = TrunkExploitability(policy, &dl_solver);
    SPIEL_CHECK_FLOAT_NEAR(actual_exploitability, 0., 1e-10);
  }

  std::vector<BanditVector> policy =
      MakeKuhnParametricPolicy(&dl_solver, 1 / 2.);
  double actual_exploitability = TrunkExploitability(policy, &dl_solver);
  SPIEL_CHECK_GT(actual_exploitability, 0.);
}

void TestOptimalValuesKuhn() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  auto oracle_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);
  dlcfr::DepthLimitedCFR dl_solver(game, /*trunk_depth_limit=*/3,
                                   oracle_evaluator, terminal_evaluator);

  for (double a : std::vector<double>{
    0.,
    1/6.,
    1/3.,  // Range of values that produce Nash (optimal).
  }) {
    std::cout << "a: " << a << '\n';
    // Set player strategies.
    // Nomenclature of variables: "action"_"infostate" where action=p/b (pass/bet)
    const double
      // Player 0
      p_0   = 1. - a      , b_0   = a         ,
      p_1   = 1.          , b_1   = 0.        ,
      p_2   = 1. - 3. * a , b_2   = 3. * a    ,
      p_0pb = 1.          , b_0pb = 0.        ,
      p_1pb = 2 / 3. - a  , b_1pb = a + 1 / 3.,
      p_2pb = 0.          , b_2pb = 1.        ,
      // Player 1
      p_0p = 2 / 3.       , b_0p = 1 / 3.     ,
      p_0b = 1.           , b_0b = 0.         ,
      p_1p = 1.           , b_1p = 0.         ,
      p_1b = 2 / 3.       , b_1b = 1 / 3.     ,
      p_2p = 0.           , b_2p = 1.         ,
      p_2b = 0.           , b_2b = 1.;

    // Counterfactual values of infostates under the specified profile.
    // Nomenclature of variables: v"player"_"infostate"
    const double
     // Player 0, pass
      v0_0p = p_1p * -1/6. + b_1p * (p_0pb * -1/6. + b_0pb * -2/6.)
            + p_2p * -1/6. + b_2p * (p_0pb * -1/6. + b_0pb * -2/6.),
      v0_1p = p_0p *  1/6. + b_0p * (p_1pb * -1/6. + b_1pb *  2/6.)
            + p_2p * -1/6. + b_2p * (p_1pb * -1/6. + b_1pb * -2/6.),
      v0_2p = p_0p *  1/6. + b_0p * (p_2pb * -1/6. + b_2pb *  2/6.)
            + p_1p *  1/6. + b_1p * (p_2pb * -1/6. + b_2pb *  2/6.),
      // Player 0, bet
      v0_0b = p_1b *  1/6. + b_1b * -2/6.
            + p_2b *  1/6. + b_2b * -2/6.,
      v0_1b = p_0b *  1/6. + b_0b *  2/6.
            + p_2b *  1/6. + b_2b * -2/6.,
      v0_2b = p_0b *  1/6. + b_0b *  2/6.
            + p_1b *  1/6. + b_1b *  2/6.,
      // Player 1, pass:
      v1_0p = p_1 * (p_0p * -1/6. + b_0p * (p_1pb * 1/6. + b_1pb * -2/6.))
            + p_2 * (p_0p * -1/6. + b_0p * (p_2pb * 1/6. + b_2pb * -2/6.)),
      v1_1p = p_0 * (p_1p *  1/6. + b_1p * (p_0pb * 1/6. + b_0pb *  2/6.))
            + p_2 * (p_1p * -1/6. + b_1p * (p_2pb * 1/6. + b_2pb * -2/6.)),
      v1_2p = p_0 * (p_2p *  1/6. + b_2p * (p_0pb * 1/6. + b_0pb *  2/6.))
            + p_1 * (p_2p *  1/6. + b_2p * (p_1pb * 1/6. + b_1pb *  2/6.)),
      // Player 1, bet:
      v1_0b = b_1 * (p_0b * -1/6. + b_0b * -2/6.)
            + b_2 * (p_0b * -1/6. + b_0b * -2/6.),
      v1_1b = b_0 * (p_1b * -1/6. + b_1b *  2/6.)
            + b_2 * (p_1b * -1/6. + b_1b * -2/6.),
      v1_2b = b_0 * (p_2b * -1/6. + b_2b *  2/6.)
            + b_1 * (p_2b * -1/6. + b_2b *  2/6.);

    const double root_val0 = (p_0 * v0_0p + p_1 * v0_1p + p_2 * v0_2p) +
                             (b_0 * v0_0b + b_1 * v0_1b + b_2 * v0_2b);
    const double root_val1 = (v1_0p + v1_1p + v1_2p) + (v1_0b + v1_1b + v1_2b);
    SPIEL_CHECK_FLOAT_EQ(root_val0, -root_val1);

    std::vector<BanditVector> bandit_fixed_policy =
        MakeKuhnParametricPolicy(&dl_solver, a);

    dl_solver.PrepareRootReachProbs();
    for (int pl = 0; pl < 2; ++pl) {
      TopDownCurrentPolicy(
          *dl_solver.trees()[pl], bandit_fixed_policy[pl],
          absl::MakeSpan(dl_solver.reach_probs()[pl]));
    }
    dl_solver.EvaluateLeaves();

    auto& state_p = dl_solver.public_leaves()[0];
    auto& state_b = dl_solver.public_leaves()[1];
    auto& solvers_p = open_spiel::down_cast<OraclePublicStateContext*>(dl_solver.contexts()[0].get())->solvers;
    auto& solvers_b = open_spiel::down_cast<OraclePublicStateContext*>(dl_solver.contexts()[1].get())->solvers;

    std::cout << "# PASS" << std::endl;
    std::cout << "Policy pl0: " << solvers_p[0].OptimalPolicy(0).PolicyTable() << "\n";
    std::cout << "Policy pl1: " << solvers_p[1].OptimalPolicy(1).PolicyTable() << "\n";
    std::cout << "pl0" << std::endl;
    std::cout << state_p.leaf_nodes[0][0]->infostate_string() << ' ' << state_p.values[0][0] << ' ' << v0_0p << '\n';
    if(p_0) SPIEL_CHECK_FLOAT_EQ(state_p.values[0][0], v0_0p);
    std::cout << state_p.leaf_nodes[0][1]->infostate_string() << ' ' << state_p.values[0][1] << ' ' << v0_1p << '\n';
    if(p_1) SPIEL_CHECK_FLOAT_EQ(state_p.values[0][1], v0_1p);
    std::cout << state_p.leaf_nodes[0][2]->infostate_string() << ' ' << state_p.values[0][2] << ' ' << v0_2p << '\n';
    if(p_2) SPIEL_CHECK_FLOAT_EQ(state_p.values[0][2], v0_2p);
    std::cout << "pl1" << std::endl;
    std::cout << state_p.leaf_nodes[1][0]->infostate_string() << ' ' << state_p.values[1][0] << ' ' << v1_1p << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_p.values[1][0], v1_1p);
    std::cout << state_p.leaf_nodes[1][1]->infostate_string() << ' ' << state_p.values[1][1] << ' ' << v1_2p << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_p.values[1][1], v1_2p);
    std::cout << state_p.leaf_nodes[1][2]->infostate_string() << ' ' << state_p.values[1][2] << ' ' << v1_0p << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_p.values[1][2], v1_0p);
    std::cout << state_b.leaf_nodes[0][0]->infostate_string() << ' ' << state_b.values[0][0] << ' ' << v0_0b << '\n';

    std::cout << "# BET" << std::endl;
    std::cout << "Policy pl0: " << solvers_b[0].OptimalPolicy(0).PolicyTable() << "\n";
    std::cout << "Policy pl1: " << solvers_b[1].OptimalPolicy(1).PolicyTable() << "\n";
//    solvers_b[1].PrintProblemSpecification();
    std::cout << "pl0" << std::endl;
    std::cout << state_b.leaf_nodes[0][0]->infostate_string() << ' ' << state_b.values[0][0] << ' ' << v0_0b << '\n';
    if(b_0) SPIEL_CHECK_FLOAT_EQ(state_b.values[0][0], v0_0b);
    std::cout << state_b.leaf_nodes[0][1]->infostate_string() << ' ' << state_b.values[0][1] << ' ' << v0_1b << '\n';
    if(b_1) SPIEL_CHECK_FLOAT_EQ(state_b.values[0][1], v0_1b);
    std::cout << state_b.leaf_nodes[0][2]->infostate_string() << ' ' << state_b.values[0][2] << ' ' << v0_2b << '\n';
    if(b_2) SPIEL_CHECK_FLOAT_EQ(state_b.values[0][2], v0_2b);
    std::cout << "pl1" << std::endl;
    std::cout << state_b.leaf_nodes[1][0]->infostate_string() << ' ' << state_b.values[1][0] << ' ' << v1_1b << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_b.values[1][0], v1_1b);
    std::cout << state_b.leaf_nodes[1][1]->infostate_string() << ' ' << state_b.values[1][1] << ' ' << v1_0b << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_b.values[1][1], v1_0b);
    std::cout << state_b.leaf_nodes[1][2]->infostate_string() << ' ' << state_b.values[1][2] << ' ' << v1_2b << '\n';
    SPIEL_CHECK_FLOAT_EQ(state_b.values[1][2], v1_2b);
    std::cout << "----\n";
  }
}

void TestValueOracle(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  for (int trunk_depth_limit = 0; trunk_depth_limit < game->MaxMoveNumber();
       ++trunk_depth_limit) {

    std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
        dlcfr::MakeTerminalEvaluator();
    std::shared_ptr<Observer> public_observer =
        game->MakeObserver(kPublicStateObsType, {});
    std::shared_ptr<Observer> infostate_observer =
        game->MakeObserver(kInfoStateObsType, {});

    auto leaf_evaluator =
        std::make_shared<OracleEvaluator>(game, infostate_observer);

    dlcfr::DepthLimitedCFR dl_solver(game, trunk_depth_limit,
                                     leaf_evaluator, terminal_evaluator);
    dl_solver.RunSimultaneousIterations(3);
  }
}

void TestTrunkDeepAsWholeGameExploitability(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  dlcfr::DepthLimitedCFR dl_solver(game, 100, nullptr, terminal_evaluator);
  std::vector<BanditVector> uniform_bandits =
      MakeBanditVectors(dl_solver.trees(), "UniformStrategy");
  UniformPolicy uniform_policy;

  // Exploitability implementation does not support simultaneous move games.
  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    game = ConvertToTurnBased(*game);
  }
  const double expected_expl = Exploitability(*game, uniform_policy);
  const double actual_expl = TrunkExploitability(uniform_bandits, &dl_solver);
  SPIEL_CHECK_FLOAT_NEAR(expected_expl, actual_expl, 1e-10);
}

void TestOracleConvergence() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  auto oracle_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);

  dlcfr::DepthLimitedCFR dl_solver(
      game, 3, oracle_evaluator, terminal_evaluator);

  for (int i = 0; i < 500; ++i) {
    dl_solver.RunSimultaneousIterations(1);
    double current_expl = TrunkExploitability(dl_solver.bandits(), &dl_solver);
    double avg_expl = TrunkExploitability(dl_solver.bandits(), &dl_solver);
    std::cout << i << "," << current_expl << "," << avg_expl << "," << std::endl;
  }
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms::ortools;

int main(int argc, char** argv) {
  algorithms::TestTrunkExploitabilityOptimalValuesKuhn();
//  algorithms::TestOptimalValuesKuhn();
//  algorithms::TestTrunkExploitability();
//  algorithms::TestOracleConvergence();
//  algorithms::TestOptimalValuesKuhnBettingPublicState();
//
  std::vector<std::string> test_games = {
      "matrix_mp",
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=4,imp_info=True)",
  };
  for (const std::string& game_name : test_games) {
//    algorithms::TestTrunkDeepAsWholeGameExploitability(game_name);
//    algorithms::TestValueOracle(game_name);
  }
}
