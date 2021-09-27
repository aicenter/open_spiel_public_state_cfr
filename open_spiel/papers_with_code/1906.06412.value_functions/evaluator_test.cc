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

#include "open_spiel/algorithms/infostate_cfr.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/solver.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/evaluator.h"

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


}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  std::vector<std::string> test_games = {
      "kuhn_poker",
      "leduc_poker",
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)",
  };

  for (const std::string& game_name : test_games) {
    open_spiel::papers_with_code::TestTerminalEvaluatorHasSameIterations(
        game_name);
  }
}
