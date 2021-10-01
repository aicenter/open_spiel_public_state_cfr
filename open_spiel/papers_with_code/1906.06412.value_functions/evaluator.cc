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
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver.h"
#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/bandits_policy.h"


namespace open_spiel {
namespace papers_with_code {

// -- Dummy evaluator ----------------------------------------------------------

// Evaluator that does nothing.
struct DummyEvaluator : public PublicStateEvaluator {
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override { return nullptr; };
  void ResetContext(PublicStateContext* context) const override {};
  void EvaluatePublicState(PublicState* public_state,
                           PublicStateContext* context) const override {};
};

// -- Terminal evaluator -------------------------------------------------------

TerminalPublicStateContext::TerminalPublicStateContext(
    const PublicState& state) {
  SPIEL_CHECK_TRUE(state.IsTerminal());
  auto& leaf_nodes = state.nodes;
  SPIEL_CHECK_EQ(leaf_nodes[0].size(), leaf_nodes[1].size());
  const int num_terminals = leaf_nodes[0].size();
  utilities.reserve(num_terminals);
  permutation.reserve(num_terminals);

  using History = absl::Span<const Action>;
  std::map<History, int> player1_map;
  for (int i = 0; i < num_terminals; ++i) {
    player1_map[leaf_nodes[1][i]->TerminalHistory()] = i;
  }
  SPIEL_CHECK_EQ(player1_map.size(), leaf_nodes[1].size());

  for (int i = 0; i < num_terminals; ++i) {
    const algorithms::InfostateNode* a = leaf_nodes[0][i];
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const algorithms::InfostateNode* b = leaf_nodes[1][permutation_index];
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const algorithms::InfostateNode* leaf = leaf_nodes[0][i];
    const double v = leaf->terminal_utility();
    const double chn = leaf->terminal_chance_reach_prob();
    utilities.push_back(v * chn);
    permutation.push_back(permutation_index);
  }

  // A quick check to see if the permutation is ok
  // by computing the arithmetic sum.
  SPIEL_DCHECK_EQ(
      std::accumulate(permutation.begin(),
                      permutation.end(), 0),
      num_terminals * (num_terminals - 1) / 2);
}

std::unique_ptr<PublicStateContext> TerminalEvaluator::CreateContext(
    const PublicState& state) const {
  return std::make_unique<TerminalPublicStateContext>(state);
}

void TerminalEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  auto* terminal = open_spiel::down_cast<TerminalPublicStateContext*>(context);
  for (int i = 0; i < terminal->utilities.size(); ++i) {
    const int j = terminal->permutation[i];
    state->values[0][i] = terminal->utilities[i] * state->beliefs[1][j];
    state->values[1][j] = - terminal->utilities[i] * state->beliefs[0][i];
  }
}

// -- CFR evaluator ------------------------------------------------------------

struct CFRContext : public PublicStateContext {
  std::unique_ptr<SubgameSolver> dlcfr;
  explicit CFRContext(std::unique_ptr<SubgameSolver> d)
      : dlcfr(std::move(d)) {}
};

void CheckChildPublicStateConsistency(
    const CFRContext& cfr_public_state, const PublicState& leaf_state) {
  SPIEL_CHECK_TRUE(leaf_state.IsLeaf());
  auto trees = cfr_public_state.dlcfr->subgame()->trees;
  for (int pl = 0; pl < 2; ++pl) {
    const algorithms::InfostateNode& root = trees[pl]->root();
    SPIEL_CHECK_EQ(leaf_state.nodes[pl].size(), root.num_children());
    for (int i = 0; i < root.num_children(); ++i) {
      const algorithms::InfostateNode& actual = *root.child_at(i);
      const algorithms::InfostateNode& expected = *leaf_state.nodes[pl][i];
      SPIEL_CHECK_EQ(actual.infostate_string(), expected.infostate_string());
    }
  }
  // All OK.
}

CFREvaluator::CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
                           std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
                           std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
                           std::shared_ptr<Observer> public_observer,
                           std::shared_ptr<Observer> infostate_observer,
                           int cfr_iterations)
    : game(std::move(game)), depth_limit(depth_limit),
      nonterminal_evaluator(std::move(leaf_evaluator)),
      terminal_evaluator(std::move(terminal_evaluator)),
      public_observer(std::move(public_observer)),
      infostate_observer(std::move(infostate_observer)),
      num_cfr_iterations(cfr_iterations) {
  SPIEL_CHECK_GT(depth_limit, 0);
}

std::unique_ptr<PublicStateContext> CFREvaluator::CreateContext(
    const PublicState& state) const {
  if (!state.IsLeaf()) return nullptr;
  SPIEL_CHECK_TRUE(state.IsLeaf());

  auto subgame_trees = std::vector{
      MakeInfostateTree(state.nodes[0],
                        depth_limit, kDlCfrInfostateTreeStorage),
      MakeInfostateTree(state.nodes[1],
                        depth_limit, kDlCfrInfostateTreeStorage)
  };
  auto subgame = std::make_shared<Subgame>(
      game, public_observer, subgame_trees);
  auto solver = std::make_unique<SubgameSolver>(
      subgame, nonterminal_evaluator, terminal_evaluator, /*rnd_gen=*/nullptr,
      bandit_name, save_values_policy);
  auto cfr_public_state = std::make_unique<CFRContext>(std::move(solver));
  SPIEL_DCHECK(CheckChildPublicStateConsistency(*cfr_public_state, state));
  return cfr_public_state;
}

void CFREvaluator::ResetContext(PublicStateContext* context) const {
  auto* cfr_state = open_spiel::down_cast<CFRContext*>(context);
  cfr_state->dlcfr->Reset();
}

void CFREvaluator::EvaluatePublicState(PublicState* state,
                                       PublicStateContext* context) const {
  SPIEL_CHECK_TRUE(state->IsLeaf());
  auto* cfr_context = open_spiel::down_cast<CFRContext*>(context);
  SubgameSolver* solver = cfr_context->dlcfr.get();
  // We pretty much always should. This only to support special test cases.
  if (reset_subgames_on_evaluation) {
    solver->Reset();
  }
  solver->initial_state().SetBeliefs(state->beliefs);
  solver->RunSimultaneousIterations(num_cfr_iterations);
  auto& resulting_values = solver->initial_state().values;
  // Copy the results.
  for (int pl = 0; pl < 2; ++pl) {
    std::copy(resulting_values[pl].begin(), resulting_values[pl].end(),
              state->values[pl].begin());
  }
//  for (const algorithms::InfostateNode* node : state->nodes[0]) {
//    std::cout << node->infostate_string() << "\n";
//  }
//  std::cout << state->public_id << " "
//            << " " << " beliefs: " << state->beliefs[0]
//            << " " << " values: " << state->values[0]  << "\n";
}

// -- Shorthand factories ------------------------------------------------------

std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator() {
  return std::make_shared<TerminalEvaluator>();
}

std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator() {
  return std::make_shared<DummyEvaluator>();
}

std::shared_ptr<PublicStateEvaluator> MakeApproxOracleEvaluator(
    std::shared_ptr<const Game> game, int cfr_iterations) {

  std::shared_ptr<const PublicStateEvaluator> dummy_evaluator =
      MakeDummyEvaluator();
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});

  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
      MakeTerminalEvaluator();

  return std::make_shared<CFREvaluator>(game, algorithms::kNoMoveAheadLimit,
                                        dummy_evaluator, terminal_evaluator,
                                        public_observer, infostate_observer,
                                        cfr_iterations);
}

}  // namespace papers_with_code
}  // namespace open_spiel
