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
    const std::vector<std::shared_ptr<InfostateTree>>& trees, double a) {
  // Set strategy as described in https://en.wikipedia.org/wiki/Kuhn_poker

  // Player 0
  BanditVector vec0(trees[0].get());
  MakeFixedBandit(vec0, "0"  ,  { 1. - a      , a          });
  MakeFixedBandit(vec0, "0pb",  { 1           , 0.         });
  MakeFixedBandit(vec0, "1"  ,  { 1.          , 0.         });
  MakeFixedBandit(vec0, "1pb",  { 2 / 3. - a  , a + 1 / 3. });
  MakeFixedBandit(vec0, "2"  ,  { 1. - 3. * a , 3. * a     });
  MakeFixedBandit(vec0, "2pb",  { 0.          , 1.         });

  // Player 1
  BanditVector vec1(trees[1].get());
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

void TestTrunkExploitabilityInKuhn() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  SequenceFormLpSpecification whole_game(*game);

  // Range of values that produce Nash in the trunk.
  for (double a : std::vector<double>{0., 1 / 6., 1 / 3.,}) {
    auto bandit_policy = MakeKuhnParametricPolicy(whole_game.trees(), a);
    BanditsCurrentPolicy policy(whole_game.trees(), bandit_policy);
    const double actual_expl = TrunkExploitability(&whole_game, policy,
                                                   /*strategy_epsilon=*/0.);
    SPIEL_CHECK_FLOAT_NEAR(actual_expl, 0., 1e-10);

    const double value0 = ComputeRootValueWhileFixingStrategy(
        &whole_game, policy, /*pl=*/0, /*strategy_epsilon=*/0.);
    SPIEL_CHECK_FLOAT_NEAR(value0, -1. / 18., 1e-10);

    const double value1 = ComputeRootValueWhileFixingStrategy(
        &whole_game, policy, /*pl=*/1, /*strategy_epsilon=*/0.);
    SPIEL_CHECK_FLOAT_NEAR(value1, 1. / 18., 1e-10);

    for (int pl = 0; pl < 2; ++pl) {
      const double actual_pl_expl =
          TrunkPlayerExploitability(&whole_game, policy, /*pl=*/pl,
                                    {}, /*strategy_epsilon=*/0.);
      SPIEL_CHECK_FLOAT_NEAR(actual_pl_expl, 0., 1e-10);
    }
  }

  {
    // Policy that is not a Nash.
    auto bandit_policy =
        MakeBanditVectors(whole_game.trees(), "UniformStrategy");
    BanditsCurrentPolicy policy(whole_game.trees(), bandit_policy);
    const double actual_expl = TrunkExploitability(&whole_game, policy,
                                                   /*strategy_epsilon=*/0.);
    SPIEL_CHECK_GT(actual_expl, 0.);
  }
}

void TestOptimalValuesKuhn() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<const dlcfr::PublicStateEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  auto oracle_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);
  dlcfr::DepthLimitedCFR dl_solver(game, /*trunk_depth_limit=*/3,
                                   oracle_evaluator, terminal_evaluator);

  // Range of values that produce Nash.
  for (double a : std::vector<double>{0., 1 / 6., 1 / 3.}) {
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
        MakeKuhnParametricPolicy(dl_solver.trees(), a);

    dl_solver.PrepareRootReachProbs();
    for (int pl = 0; pl < 2; ++pl) {
      TopDown(*dl_solver.trees()[pl], bandit_fixed_policy[pl],
              absl::MakeSpan(dl_solver.reach_probs()[pl]), 0);
    }
    dl_solver.EvaluateLeaves();

    auto& state_p = dl_solver.public_states()[0];
    auto& state_b = dl_solver.public_states()[1];

    SPIEL_CHECK_FLOAT_EQ(
        state_p.values[1][0] + state_p.values[1][1] + state_p.values[1][2],
        v1_1p + v1_2p + v1_0p);
    SPIEL_CHECK_FLOAT_EQ(
        state_b.values[1][0] + state_b.values[1][1] + state_b.values[1][2],
        v1_1b + v1_2b + v1_0b);
  }
}

void TestValueOracle(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);

  for (int trunk_depth_limit = 1; trunk_depth_limit <= game->MaxMoveNumber();
       ++trunk_depth_limit) {
    std::cout << "Value oracle for depth limit "
              << trunk_depth_limit << " " << std::flush;;
    std::shared_ptr<const dlcfr::PublicStateEvaluator> terminal_evaluator =
        dlcfr::MakeTerminalEvaluator();
    std::shared_ptr<Observer> public_observer =
        game->MakeObserver(kPublicStateObsType, {});
    std::shared_ptr<Observer> infostate_observer =
        game->MakeObserver(kInfoStateObsType, {});

    auto leaf_evaluator =
        std::make_shared<OracleEvaluator>(game, infostate_observer);

    dlcfr::DepthLimitedCFR dl_solver(game, trunk_depth_limit,
                                     leaf_evaluator, terminal_evaluator);
    for (int i = 0; i < 10; ++i) {
      dl_solver.RunSimultaneousIterations(100);
      std::cout << '.' << std::flush;
    }

    SequenceFormLpSpecification whole_game(*game);
    auto average_policy = dl_solver.AveragePolicy();
    const double expl = TrunkExploitability(&whole_game, *average_policy);
    std::cout << " expl=" << expl << "\n";
    SPIEL_CHECK_LT(expl, 3e-2);
  }
}

void TestOneSidedFixedStrategyExploitability(const std::string& game_name) {
  std::cout << "One sided exploitability ... " << std::flush;;
  std::shared_ptr<const Game> game = LoadGame(game_name);
  SequenceFormLpSpecification whole_game(*game);

  // Exploitability implementation does not support simultaneous move games.
  if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
    game = ConvertToTurnBased(*game);
  }
  UniformPolicy uniform_policy;
  const double expected_expl = Exploitability(*game, uniform_policy);

  auto bandit_policy = MakeBanditVectors(whole_game.trees(), "UniformStrategy");
  BanditsCurrentPolicy uniform_bandit_policy(whole_game.trees(), bandit_policy);
  const double actual_expl = TrunkExploitability(
      &whole_game, uniform_bandit_policy);
  SPIEL_CHECK_FLOAT_NEAR(expected_expl, actual_expl, 1e-10);
  std::cout << "ok.\n";
}


void TestConvergeWithCfrEvaluator(int trunk_depth) {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<const dlcfr::PublicStateEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  // CFR leaf evaluator.
  auto leaf_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      game, /*depth_limit=*/100, /*leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  leaf_evaluator->num_cfr_iterations = 2;

  SequenceFormLpSpecification whole_game(*game);
  dlcfr::DepthLimitedCFR dl_cfr(game, trunk_depth,
                                leaf_evaluator, terminal_evaluator);
  auto average_policy = dl_cfr.AveragePolicy();
  const double expl_before = TrunkExploitability(&whole_game, *average_policy);

  for (int i = 0; i < 100; ++i) {
    dl_cfr.RunSimultaneousIterations(1);
    std::cout << i << " " << TrunkExploitability(&whole_game, *average_policy) << "\n";
  }
  const double expl_after = TrunkExploitability(&whole_game, *average_policy);
  SPIEL_CHECK_NE(expl_before, expl_after);
  SPIEL_CHECK_LT(expl_after, 5e-3);
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms::ortools;

int main(int argc, char** argv) {
  algorithms::TestTrunkExploitabilityInKuhn();
  algorithms::TestOptimalValuesKuhn();
//  std::vector<std::string> test_games = {
//      "matrix_biased_mp",
//      "kuhn_poker",
////      "leduc_poker", // Fails!!
//      "goofspiel(players=2,num_cards=3,imp_info=True)",
//      "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)",
//  };
//  for (const std::string& game_name : test_games) {
//    std::cout << "\nTesting " << game_name << "\n";
//    algorithms::TestOneSidedFixedStrategyExploitability(game_name);
//    algorithms::TestValueOracle(game_name);
//  }
//  for (int trunk_depth = 3; trunk_depth <= 5; ++trunk_depth) {
    algorithms::TestConvergeWithCfrEvaluator(3);
//  }
}
