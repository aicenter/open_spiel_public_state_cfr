// Copybot 2019 DeepMind Technologies Ltd. All bots reserved.
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


#include "open_spiel/algorithms/dispatch_policy.h"
#include "open_spiel/algorithms/ortools/trunk_exploitability.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void CheckSparseTrunksKuhnDepth2(
    const std::vector <std::unique_ptr <SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 3);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    SPIEL_CHECK_EQ(sparse_trunks[i]->fixate_infostates.size(), 1);
    const std::string& eval_infostate = sparse_trunks[i]->fixate_infostates[0];
    // One char, i.e. card 0/1/2
    SPIEL_CHECK_EQ(eval_infostate.size(), 1);
    SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->public_states().size(), 5);

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                         ->leaf_nodes().size(), num_states * 5);
    }
  }
}

void CheckSparseTrunksKuhnDepth3(
    const std::vector <std::unique_ptr <SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 6);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    SPIEL_CHECK_EQ(sparse_trunks[i]->fixate_infostates.size(), 1);
    const std::string& eval_infostate = sparse_trunks[i]->fixate_infostates[0];
    // Two chars, i.e. card 0/1/2 followed by p/b
    SPIEL_CHECK_EQ(eval_infostate.size(), 2);

    for (int pl = 0; pl < 2; ++pl) {
      if (eval_infostate[1] == 'p') {
        // Pass infostate
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                           ->leaf_nodes().size(), num_states * 3);
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->public_states().size(), 3);
      } else if (eval_infostate[1] == 'b') {
        // Bet infostate
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                           ->leaf_nodes().size(), num_states * 2);
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->public_states().size(), 2);
      } else {
        SpielFatalError("Exhausted pattern match!");
      }
    }
  }
}

void CheckSparseTrunksKuhnDepth4(
    const std::vector <std::unique_ptr <SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 3);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    SPIEL_CHECK_EQ(sparse_trunks[i]->fixate_infostates.size(), 1);
    const std::string& eval_infostate = sparse_trunks[i]->fixate_infostates[0];
    // Three chars, i.e. card 0/1/2 followed by pb
    SPIEL_CHECK_EQ(eval_infostate.size(), 3);
    SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->public_states().size(), 2);

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                         ->leaf_nodes().size(), num_states * 2);
    }
  }
}

// Test that sparse trunks are constructed correctly.
void TestMakeSparseTrunks() {
  std::shared_ptr <const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr <Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr <Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr <PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  std::shared_ptr <PublicStateEvaluator> dummy_eval =
      MakeDummyEvaluator();
  std::string bandits_for_cfr = "RegretMatchingPlus";
  std::mt19937 rnd_gen(0);

  // Use any valid number of states.
  // -1 are for full trees test (just as 6) -- there is no sparsification.
  for (int limit = -1; limit <= 6; ++limit) {
    if (limit == 0) continue;

    for (int roots_depth = 2; roots_depth <= 4; ++roots_depth) {
      std::vector <std::unique_ptr <SparseTrunk>> sparse_trunks =
          MakeSparseTrunks(game, infostate_observer, public_observer,
                           roots_depth, /*trunk_depth=*/1000,
                           dummy_eval, terminal_evaluator,
                           limit, bandits_for_cfr, rnd_gen);
      int num_states = limit > 0 ? limit : 6;

      switch (roots_depth) {
        case 2: CheckSparseTrunksKuhnDepth2(sparse_trunks, num_states);
          break;
        case 3: CheckSparseTrunksKuhnDepth3(sparse_trunks, num_states);
          break;
        case 4: CheckSparseTrunksKuhnDepth4(sparse_trunks, num_states);
          break;
      }
    }
  }
}

// Test that we can converge to an equilibrium in a slice of the game that is in
// a support of some NE from the whole game. We make a sparse depth-limited
// subgame rooted in the slice, and call DL-CFR at the leaves of this subgame.
void TestMakeSparseTrunkWithEqSupport() {
  const int no_move_limit = 1000;  // Some big number.
  std::shared_ptr <const Game> game = LoadGame(
      "goofspiel(players=2,num_cards=5,imp_info=True,points_order=descending)");
  std::shared_ptr <Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr <Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr <PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();
  auto cfr_eval = std::make_shared <CFREvaluator>(
      game, no_move_limit, /*leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  std::string bandits_for_cfr = "RegretMatchingPlus";

  algorithms::ortools::SequenceFormLpSpecification whole_game(*game, "CLP");
  auto[eq_policy, game_value] = algorithms::ortools::MakeEquilibriumPolicy(&whole_game);

  const int roots_depth = 2;
  const int trunk_depth = 3;
  double support_threshold = 0.0;
  bool prune_chance_histories = false;

  std::unique_ptr <SparseTrunk> sparse_slice =
      MakeSparseTrunkWithEqSupport(eq_policy, game,
                                   infostate_observer, public_observer,
                                   roots_depth, trunk_depth,  // ! <--
                                   cfr_eval, terminal_evaluator,
                                   bandits_for_cfr,
                                   support_threshold, prune_chance_histories);
  std::shared_ptr <Policy> slice_policy = sparse_slice->dlcfr->AveragePolicy();

  // Make a special dispatch table for exploitability evaluation:
  // The roots of the sparse slice will change with the DL-CFR iterations.
  DispatchPolicy dispatch_policy;
  dispatch_policy.AddDispatch(sparse_slice->fixate_infostates, slice_policy);

  // For evaluation, the sparse trunk is constructed as replacing the players'
  // equilibrium policies as a chance in the upper game. By constructing the
  // trunk with no move limit, we make an evaluation trunk.
  std::unique_ptr <SparseTrunk> eval_trunk =
      MakeSparseTrunkWithEqSupport(eq_policy, game,
                                   infostate_observer, public_observer,
                                   roots_depth, no_move_limit,  // ! <--
                                   cfr_eval, terminal_evaluator,
                                   bandits_for_cfr,
                                   support_threshold, prune_chance_histories);
  algorithms::ortools::SequenceFormLpSpecification eq_fixed_as_chance_lp(
      eval_trunk->dlcfr->trees(), "CLP");

  double expl;
  for (int i = 0; i <= 50; ++i) {
    cfr_eval->num_cfr_iterations = (i + 1) * 5;  // For faster convergence.
    sparse_slice->dlcfr->RunSimultaneousIterations(1);
    if (i % 10 == 0) {
      expl = algorithms::ortools::TrunkExploitability(&eq_fixed_as_chance_lp,
                                                      dispatch_policy);
      std::cout << i << " " << expl << std::endl;
    }
  }
  SPIEL_CHECK_LT(expl, 5e-3);
}

}  // namespace
}  // papers_with_code
}  // open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestMakeSparseTrunks();
  open_spiel::papers_with_code::TestMakeSparseTrunkWithEqSupport();
}
