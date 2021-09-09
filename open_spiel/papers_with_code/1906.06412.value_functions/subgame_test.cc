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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

#include <cmath>
#include <iostream>
#include <absl/strings/str_replace.h>

#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/tabular_exploitability.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/games/goofspiel.h"
#include "open_spiel/games/kuhn_poker.h"
#include "open_spiel/games/leduc_poker.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/cfr.h"

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

std::unique_ptr<SubgameSolver> MakeRecursiveDepthLimitedCFR(
    std::shared_ptr<const Game> game, int trunk_depth_limit,
    int subgame_depth_limit) {
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr<PublicStateEvaluator> nonterminal_evaluator =
      MakeDummyEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  // Recursive leaf evaluator.

  auto leaf_evaluator = std::make_shared<CFREvaluator>(
      game, subgame_depth_limit, nonterminal_evaluator,
      terminal_evaluator, public_observer, infostate_observer);
  leaf_evaluator->reset_subgames_on_evaluation = false;  // Needed for test !
  leaf_evaluator->bandit_name = "RegretMatching";
  leaf_evaluator->nonterminal_evaluator = leaf_evaluator;
  leaf_evaluator->num_cfr_iterations = 1;
  leaf_evaluator->save_values_policy = PolicySelection::kCurrentPolicy;

  // Builds the root leaf public states so that we can call the recursive
  // evaluator.
  auto subgame = std::make_shared<Subgame>(game, trunk_depth_limit);
  return std::make_unique<SubgameSolver>(subgame, leaf_evaluator,
                                         terminal_evaluator,
                                         /*rnd_gen=*/nullptr,
                                         "RegretMatching",
                                         PolicySelection::kCurrentPolicy);
}

void TestRecursiveDepthLimitedSolving(const std::string& game_name) {
  // If we make 1 iterations in each of the recursive subgames, it is the same
  // as if we were running CFR in the whole game. Thus we can check that we
  // compute the same regrets as the original implementation.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  const int trunk_iterations = 10;

  for (int trunk_depth_limit = 0; trunk_depth_limit < game->MaxMoveNumber();
       ++trunk_depth_limit) {
    for (int subgame_depth_limit = 1; subgame_depth_limit < 4;
         ++subgame_depth_limit) {

      algorithms::InfostateCFR vec_solver(*game);
      std::unique_ptr<SubgameSolver> dl_solver = MakeRecursiveDepthLimitedCFR(
          game, trunk_depth_limit, subgame_depth_limit);

      std::shared_ptr<Policy> vec_avg = vec_solver.AveragePolicy();
      std::shared_ptr<Policy> dl_avg = dl_solver->AveragePolicy();
      std::shared_ptr<Policy> vec_cur = vec_solver.CurrentPolicy();
      std::shared_ptr<Policy> dl_cur = dl_solver->CurrentPolicy();
      auto trees = dl_solver->subgame()->trees;

      for (int j = 0; j < trunk_iterations; ++j) {
        vec_solver.RunSimultaneousIterations(1);
        dl_solver->RunSimultaneousIterations(1);
        SPIEL_CHECK_FLOAT_NEAR(vec_solver.RootValue(),
                               dl_solver->initial_state().CurrentValue(), 1e-6);
        for (int pl = 0; pl < 2; ++pl) {
          CheckIterationConsistency(*vec_avg, *dl_avg, *trees[pl]);
          CheckIterationConsistency(*vec_cur, *dl_cur, *trees[pl]);
        }
      }
    }
  }
}

void TestMakeAllPublicStates(const std::string& game_name) {
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);

  for (PublicState& s : all->public_states) {
// Debug print:
//    std::cout << "----" << std::endl;
//    std::cout << "Obs: " << s.public_tensor.Tensor() << "\n";
//    for (int pl = 0; pl < 2; ++pl) {
//      std::cout << "Nodes " << pl << ":\n";
//      for (const InfostateNode* node : s.nodes[pl]) {
//        std::cout << "  " << node->TreePath() << "\n";
//        std::cout << "  States:\n";
//        for (const std::unique_ptr<State> & state
//            : node->corresponding_states()) {
//          std::cout << "    " << state->HistoryString() << "\n";
//        }
//      }
//    }

    using History = std::vector<Action>;
    std::unordered_set<History, absl::Hash<History>> state_histories;
    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_FALSE(s.nodes[pl].empty());
      SPIEL_CHECK_FALSE(s.nodes[0][0]->corresponding_states().empty());
      State* a_state = s.nodes[0][0]->corresponding_states()[0].get();

      for (const algorithms::InfostateNode* node : s.nodes[pl]) {
        SPIEL_CHECK_FALSE(node->corresponding_states().empty());
        SPIEL_CHECK_EQ(node->tree().acting_player(), pl);

        for (const std::unique_ptr<State>& state
            : node->corresponding_states()) {
          const auto& h = state->History();
          SPIEL_CHECK_EQ(a_state->MoveNumber(), state->MoveNumber());
          if (pl == 0) {
            SPIEL_CHECK_TRUE(state_histories.find(h) == state_histories.end());
            state_histories.insert(h);
          } else {
            SPIEL_CHECK_TRUE(state_histories.find(h) != state_histories.end());
          }
        }
      }
    }
  }
}

std::unique_ptr<Subgame> Player0lossSubgame() {
  std::shared_ptr<const Game> game = LoadGame(
      "goofspiel(players=2,num_turns=3,num_cards=4,imp_info=True,points_order=descending)");
  std::unique_ptr<PublicStatesInGame> all = MakeAllPublicStates(*game);
  // Pick public state with PL0 loss.
  for (PublicState& s : all->public_states) {
    absl::Span<const float> wins = s.public_tensor.Tensor("win_sequence");
    if (wins == std::vector<float>{0, 1, 0, 0, 0, 0}) {
      SPIEL_CHECK_EQ(s.nodes[0].size(), 3);
      SPIEL_CHECK_EQ(s.nodes[1].size(), 3);
      return MakeSubgame(s);
    }
  }
}

void TestValuesInLimitedGoofspiel() {
  std::shared_ptr<Subgame> subgame = Player0lossSubgame();
  auto solver = std::make_unique<SubgameSolver>(subgame,
                                                /*nonterminal_evaluator=*/nullptr,
                                                MakeTerminalEvaluator(),
                                                /*rnd_gen=*/nullptr,
                                                "PredictiveRegretMatchingPlus");
  subgame->initial_state().beliefs[0] = {0., 0., 0.};  // PL0 doesn't want to play here.
  subgame->initial_state().beliefs[1] = {0., 0., 1.};  // PL1 plays here with the highest card (4).

  solver->RunSimultaneousIterations(10000);
  // 1/3, 1/3, -1/3
  std::cout << subgame->initial_state().values[0] << "\n";
  // 0, 0, 0
  std::cout << subgame->initial_state().values[1] << "\n";
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
    open_spiel::papers_with_code::TestRecursiveDepthLimitedSolving(game_name);
    open_spiel::papers_with_code::TestMakeAllPublicStates(game_name);
  }

  open_spiel::papers_with_code::TestValuesInLimitedGoofspiel();
}
