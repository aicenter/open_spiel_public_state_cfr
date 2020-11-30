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
        std::vector<float>({alpha, alpha, 1 - alpha}),
        // Player 1 did not act before this public state, so ranges are fixed.
        std::vector<float>({1., 1., 1.})
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

void TestOracleConvergence() {
  std::shared_ptr<const Game> game = LoadGame("leduc_poker");

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  auto leaf_evaluator =
      std::make_shared<OracleEvaluator>(game, infostate_observer);

  dlcfr::DepthLimitedCFR dl_solver(game, 4,
                                   leaf_evaluator, terminal_evaluator);
  for (int i = 0; i < 5000; ++i) {
    dl_solver.RunSimultaneousIterations(1);
    std::cout << i << "," << dl_solver.TrunkExploitability() << std::endl;
  }
}

}  // namespace
}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel

namespace algorithms = open_spiel::algorithms::ortools;

int main(int argc, char** argv) {
//  algorithms::TestOracleConvergence();
  algorithms::TestOptimalValuesKuhnBettingPublicState();

  std::vector<std::string> test_games = {
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=4,imp_info=True)",
  };
  for (const std::string& game_name : test_games) {
    algorithms::TestValueOracle(game_name);
  }
}
