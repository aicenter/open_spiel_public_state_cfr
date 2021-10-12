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

#include "open_spiel/papers_with_code/1906.06412.value_functions/evaluator.h"

#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/solver.h"


namespace open_spiel {
namespace papers_with_code {
namespace {

void CheckInfostatePolicy(
    const std::string& infostate, const Policy& a, const Policy& b) {
  ActionsAndProbs vec_policy = a.GetStatePolicy(infostate);
  ActionsAndProbs str_policy = b.GetStatePolicy(infostate);
  SPIEL_CHECK_EQ(vec_policy.size(), str_policy.size());
  for (int j = 0; j < vec_policy.size(); ++j) {
    SPIEL_CHECK_EQ(vec_policy[j].first, str_policy[j].first);
    SPIEL_CHECK_FLOAT_NEAR(vec_policy[j].second, str_policy[j].second, 1e-6);
  }
}

void CheckIterationConsistency(const Policy& a, const Policy& b,
                               const algorithms::InfostateTree& tree) {
  for (algorithms::DecisionId id : tree.AllDecisionIds()) {
    CheckInfostatePolicy(tree.decision_infostate(id)->infostate_string(), a, b);
  }
}

void TestTerminalEvaluatorHasSameIterations(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int cfr_iterations = 10;

  algorithms::InfostateCFR vec_solver(*game);

  // We use only the terminal evaluator.
  std::shared_ptr<PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
      MakeDummyEvaluator();
  auto subgame = std::make_shared<Subgame>(game, algorithms::kNoMoveAheadLimit);
  SubgameSolver dl_solver(subgame, nonterminal_evaluator, terminal_evaluator,
      /*rnd_gen=*/nullptr, "RegretMatching");

  std::shared_ptr<Policy> vec_avg = vec_solver.AveragePolicy();
  std::shared_ptr<Policy> dl_avg = dl_solver.AveragePolicy();
  std::shared_ptr<Policy> vec_cur = vec_solver.CurrentPolicy();
  std::shared_ptr<Policy> dl_cur = dl_solver.CurrentPolicy();

  for (int i = 0; i < cfr_iterations; ++i) {
    vec_solver.RunSimultaneousIterations(1);
    dl_solver.RunSimultaneousIterations(1);
    for (int pl = 0; pl < 2; ++pl) {
      CheckIterationConsistency(*vec_avg, *dl_avg, *vec_solver.trees()[pl]);
      CheckIterationConsistency(*vec_cur, *dl_cur, *vec_solver.trees()[pl]);
    }
  }
}

void TestOracleEvaluatorMP() {
  auto game = LoadGameAsTurnBased("matrix_mp");
  auto evaluator = MakeOracleEvaluator(game);
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);
  SPIEL_CHECK_EQ(all->public_states.size(), 3);
  SPIEL_CHECK_TRUE(all->public_states[0].IsInitial());
  SPIEL_CHECK_TRUE(all->public_states[2].IsTerminal());

  PublicState* s = &all->public_states[1];
  SPIEL_CHECK_EQ(s->beliefs[0].size(), 2);  // Player 0 played.
  SPIEL_CHECK_EQ(s->beliefs[1].size(), 1);  // But not player 1.
  std::unique_ptr<PublicStateContext> context = evaluator->CreateContext(*s);
  PublicStateContext* c = context.get();

  constexpr double eps = 1. / 1024;
  {
    s->beliefs[0] = {0., 1.};
    evaluator->EvaluatePublicState(s, c);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][0], 1.);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][1], -1.);
    SPIEL_CHECK_FLOAT_EQ(s->values[1][0], 1.);
  }
  {
    s->beliefs[0] = {0.5 - eps, 0.5 + eps};
    evaluator->EvaluatePublicState(s, c);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][0],  1);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][1], -1);
    SPIEL_CHECK_FLOAT_EQ(s->values[1][0], 2 * eps);
  }
  {
    s->beliefs[0] = {0.5 + eps, 0.5 - eps};
    evaluator->EvaluatePublicState(s, c);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][0], -1);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][1], 1);
    SPIEL_CHECK_FLOAT_EQ(s->values[1][0], 2 * eps);
  }
  {
    s->beliefs[0] = {1., 0.};
    evaluator->EvaluatePublicState(s, c);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][0], -1.);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][1], 1.);
    SPIEL_CHECK_FLOAT_EQ(s->values[1][0], 1.);
  }
  {
    s->beliefs[0] = {0.5, 0.5};
    evaluator->EvaluatePublicState(s, c);
    // The LP prefers to pick one action (Heads) rather than uniform.
    // The outcome of this test can depend on the choice of the LP solver.
    SPIEL_CHECK_FLOAT_EQ(s->values[0][0], 1);
    SPIEL_CHECK_FLOAT_EQ(s->values[0][1], -1);
    SPIEL_CHECK_FLOAT_EQ(s->values[1][0], 0.);
  }
}

void TestOracleEvaluator(const std::string& game_name) {
  std::cout << "\nEvaluating " << game_name << " with (approx) oracle\n";
  auto game = LoadGameAsTurnBased(game_name);
  auto oracle_evaluator = MakeOracleEvaluator(game);
  auto approx_evaluator = MakeApproxOracleEvaluator(game, 5000);
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);
  constexpr double kTol = 5e-3;

  // Compare the public state values between oracle
  // and approximative oracle value functions.
  auto TestCase = [&](PublicState* public_state,
                      PublicStateContext* oracle_context,
                      PublicStateContext* approx_context) {
    oracle_evaluator->EvaluatePublicState(public_state, oracle_context);
    std::array<std::vector<double>, 2> oracle_values = public_state->values;

    approx_evaluator->EvaluatePublicState(public_state, approx_context);
    std::array<std::vector<double>, 2> approx_values = public_state->values;

    for (int pl = 0; pl < 2; ++pl) {
      double oracle = 0;
      double approx = 0;
      for (int j = 0; j < public_state->nodes[pl].size(); ++j) {
        oracle += public_state->beliefs[pl][j] * oracle_values[pl][j];
        approx += public_state->beliefs[pl][j] * approx_values[pl][j];
      }
      SPIEL_CHECK_FLOAT_NEAR(oracle, approx, kTol);
    }
  };

  for (PublicState& public_state : all->public_states) {
    std::cout << "Public state: " << public_state.public_tensor.Tensor() << "\n";
    auto oracle_context = oracle_evaluator->CreateContext(public_state);
    auto approx_context = approx_evaluator->CreateContext(public_state);

    int num_nodes = public_state.nodes[0].size()
                  + public_state.nodes[1].size();
    bool random_combinations = num_nodes > 6;

    int num_combinations;
    if (random_combinations) {
      // Limiting test, too many nodes...
      num_combinations = 100;
    } else {
      num_combinations = 1 << num_nodes;
    }

    std::mt19937 rnd(0);
    for (int i = 0; i < num_combinations; ++i) {
      int mask;
      if (random_combinations) {
        mask = std::uniform_int_distribution<int>(0, 1 << num_nodes)(rnd);
      } else {
        mask = i;
      }

      // One-hot beliefs.
      int shift = 0;
      for (int j = 0; j < public_state.beliefs[0].size(); ++j) {
        public_state.beliefs[0][j] = (mask & (1 << shift)) > 0 ? 1. : 0.;
        shift++;
      }
      for (int j = 0; j < public_state.beliefs[1].size(); ++j) {
        public_state.beliefs[1][j] = (mask & (1 << shift)) > 0 ? 1. : 0.;
        shift++;
      }

      TestCase(&public_state, oracle_context.get(), approx_context.get());
    }

    // Random beliefs.
    for (int i = 0; i < 10; ++i) {
      std::uniform_real_distribution<double> d(0, 1);
      for (int pl = 0; pl < 2; ++pl) {
        for (int j = 0; j < public_state.beliefs[pl].size(); ++j) {
          public_state.beliefs[pl][j] = d(rnd);
       }
      }
      TestCase(&public_state, oracle_context.get(), approx_context.get());
    }
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::papers_with_code;
  TestTerminalEvaluatorHasSameIterations("kuhn_poker");
  TestTerminalEvaluatorHasSameIterations("leduc_poker");
  TestTerminalEvaluatorHasSameIterations(
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)");

  TestOracleEvaluatorMP();
  TestOracleEvaluator("matrix_mp");
  TestOracleEvaluator("kuhn_poker");
  TestOracleEvaluator(
      "goofspiel(players=2,num_cards=3,imp_info=True,points_order=descending)");
}
