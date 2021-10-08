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
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
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
//  if (!state.IsLeaf()) return nullptr;
//  SPIEL_CHECK_TRUE(state.IsLeaf());

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
  SPIEL_CHECK_TRUE(state);
  SPIEL_CHECK_TRUE(context);

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


// -- Oracle evaluator ---------------------------------------------------------


// Following is an implementation of a counterfactually optimal value function
// (that computes values for counterfactually optimal extensions of trunk
// strategies). See [1] for details.
//
// It uses sequence-form linear program to find the Nash equilibrium strategy
// for each player, one at a time, by reformulating it to be a value-solving
// subgame, with specified beliefs of both players. The resulting strategies
// correspond to mutual best responses.
//
// However, counterfactually optimal extensions must be mutual *counterfactual*
// best responses. This is a subtle, but important distinction. This refinement
// is required because we'd like to use these values for CFR iterations in the
// trunk, and that requires the counterfactually optimal value functions. Thus
// we make post-processing of the SF-LP strategies, as briefly outlined in [1].
//
// Here are the more specific steps for the post-processing procedure:
//
// 1. Compute the mutual best-responding strategies by using the value-solving
//    subgames.
//
// 2. Make a refinement step, where the strategies in unreachable infostates
//    change from fully-uniform to uniform over actions that have the max value.
//
// 3. Compute the counterfactual values using these mutual CBR strategies.
//
// [1] Value Functions for Depth-Limited Solving in Imperfect-Information Games
// Vojtěch Kovařík, Dominik Seitz, Viliam Lisý, Jan Rudolf, Shuo Sun, Karel Ha


struct OracleEvaluator : public PublicStateEvaluator {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;
  OracleEvaluator(std::shared_ptr<const Game> game,
                  std::shared_ptr<Observer> infostate_observer)
    : game(game), infostate_observer(infostate_observer) {};
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(PublicState* s,
                           PublicStateContext* context) const override;
};

struct OraclePublicStateContext : public PublicStateContext {
  std::shared_ptr<Observer> infostate_observer;
  std::vector<const State*> histories;
  std::vector<double> chances;
  std::array<std::vector<int>, 2> belief_indices;  // History -> belief idx
  std::shared_ptr<Subgame> subgame;

 public:
  OraclePublicStateContext(const PublicState& state);

  std::vector<std::shared_ptr<algorithms::InfostateTree>>
    MakeTrees(const std::array<std::vector<double>, 2>& beliefs);
};

OraclePublicStateContext::OraclePublicStateContext(const PublicState& state) {
  subgame = MakeSubgame(state);
  infostate_observer = state.game()->MakeObserver(kInfoStateObsType, {});
  for (int i = 0; i < state.nodes[0].size(); ++i) {
    const auto* node = state.nodes[0][i];
    for (int k = 0; k < node->corresponding_states_size(); ++k) {
      const auto& state = node->corresponding_states()[k];
      const auto& chn = node->corresponding_chance_reach_probs()[k];
      histories.push_back(state.get());
      chances.push_back(chn);
      belief_indices[0].push_back(i);
    }
  }
  belief_indices[1] = std::vector<int>(histories.size(), -1);
  for (int j = 0; j < state.nodes[1].size(); ++j) {
    const auto* node = state.nodes[1][j];
    for (int k = 0; k < node->corresponding_states_size(); ++k) {
      const auto& state = node->corresponding_states()[k];
      const auto& chn = node->corresponding_chance_reach_probs()[k];
      for (int l = 0; l < histories.size(); ++l) {
        if (histories[l]->History() == state->History()) {
          belief_indices[1][l] = j;
          break;
        }
      }
    }
  }
  for (int k = 0; k < histories.size(); ++k) {
    SPIEL_CHECK_NE(belief_indices[1][k], -1);
  }
}

std::vector<std::shared_ptr<algorithms::InfostateTree>>
OraclePublicStateContext::MakeTrees(
    const std::array<std::vector<double>, 2>& beliefs) {
  std::vector<double> subgame_chances = chances;
  double chn_sum = 0.;
  for (int k = 0; k < histories.size(); ++k) {
    for (int pl = 0; pl < 2; ++pl) {
      subgame_chances[k] *= beliefs[pl].at(belief_indices[pl][k]);
    }
    SPIEL_DCHECK_PROB(subgame_chances[k]);
    chn_sum += subgame_chances[k];
  }
  // Normalize chance probs.
  if (chn_sum == 0.) {
    for (int k = 0; k < histories.size(); ++k) {
      subgame_chances[k] = 1. / histories.size();
    }
  } else {
    for (int k = 0; k < histories.size(); ++k) {
      subgame_chances[k] /= chn_sum;
    }
  }
  return algorithms::MakeInfostateTrees(histories, subgame_chances,
                                        infostate_observer);
}

class ResponseBandit : public algorithms::bandits::Bandit {
  double reach_prob_;
  int time_;
  bool unreachable_;
 public:
  ResponseBandit(std::vector<double> init_strategy, bool unreachable)
    : Bandit(std::move(init_strategy)) {
    unreachable_ = unreachable;
    SPIEL_DCHECK_TRUE(IsValidProbDistribution(current_strategy_));
  }
  void ComputeStrategy(size_t current_time, double reach_prob = 1.) override {
    time_ = current_time;
    reach_prob_ = reach_prob;
  }
  void ObserveRewards(absl::Span<const double> rewards) override {
    SPIEL_DCHECK_EQ(rewards.size(), current_strategy_.size());
    if (!unreachable_) return;  // Do not modify strategy in reachable parts.
    if (time_ > 1) return;  // Do not modify on second pass when computing value.

    double max_reward = -std::numeric_limits<double>::infinity();
    int num_max = 0;
    for (const double& reward : rewards) {
      if (reward > max_reward) {
        max_reward = reward;
        num_max = 1;
      } else if (reward == max_reward) {
        ++num_max;
      }
    }
    SPIEL_CHECK_GE(num_max, 1);

    for (int i = 0; i < num_actions(); ++i) {
      if (rewards[i] == max_reward) {
        current_strategy_[i] = 1. / num_max;
      } else {
        current_strategy_[i] = 0.;
      }
    }
    SPIEL_DCHECK_TRUE(IsValidProbDistribution(current_strategy_));
  };
};

std::vector<algorithms::BanditVector> MakeResponseBandits(
    const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees,
    const TabularPolicy& optimal_brs) {
  std::vector<algorithms::BanditVector> out;
  out.reserve(2);
  for (const std::shared_ptr<algorithms::InfostateTree>& tree : trees) {
    algorithms::BanditVector bandits(tree.get());
    for (auto* node: tree->AllDecisionInfostates()) {
      auto optimal_local_policy = optimal_brs.GetStatePolicy(node->infostate_string());
      if (optimal_local_policy.empty()) {
        int num_actions = node->num_children();
        bandits[node->decision_id()] = std::make_unique<ResponseBandit>(
            std::vector<double>(num_actions, 1. / num_actions),
            /*unreachable=*/true);
      } else {
        bandits[node->decision_id()] = std::make_unique<ResponseBandit>(
            GetProbs(optimal_local_policy), /*unreachable=*/false);
      }
    }
    out.push_back(std::move(bandits));
  }
  return out;
}

void OracleEvaluator::EvaluatePublicState(
    PublicState* state, PublicStateContext* context) const {
  SPIEL_CHECK_TRUE(context);
  auto* oracle_context = down_cast<OraclePublicStateContext*>(context);
  auto trees = oracle_context->MakeTrees(state->beliefs);
  SPIEL_CHECK_EQ(trees[0]->root().num_children(), state->nodes[0].size());
  SPIEL_CHECK_EQ(trees[1]->root().num_children(), state->nodes[1].size());

  algorithms::ortools::SequenceFormLpSpecification sf_lp(trees);
  const auto& [optimal_brs, game_value] =
      MakeEquilibriumPolicy(&sf_lp, /*uniform_imputation=*/false);

  SubgameSolver solver(
      oracle_context->subgame,
      MakeResponseBandits(oracle_context->subgame->trees, optimal_brs));
  SPIEL_CHECK_TRUE(state->public_tensor == solver.initial_state().public_tensor);
  solver.initial_state().beliefs = state->beliefs;

  solver.RunSimultaneousIterations(1);
  solver.ResetCumulValues();
  solver.RunSimultaneousIterations(1);

  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(state->nodes[pl].size(), state->values[pl].size());
    for (int i = 0; i < state->values[pl].size(); ++i) {
      auto* node = oracle_context->subgame->trees[pl]->root().child_at(i);
      state->values[pl][i] = node->cumul_value;
    }
  }
}

std::unique_ptr<PublicStateContext> OracleEvaluator::CreateContext(
    const PublicState& state) const {
  return std::make_unique<OraclePublicStateContext>(state);
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

  auto evaluator = std::make_shared<CFREvaluator>(game, algorithms::kNoMoveAheadLimit,
                                        dummy_evaluator, terminal_evaluator,
                                        public_observer, infostate_observer,
                                        cfr_iterations);
  // Use PRM+ since it is typically faster.
  evaluator->bandit_name = "PredictiveRegretMatchingPlus";
  return evaluator;
}

std::shared_ptr<PublicStateEvaluator> MakeOracleEvaluator(
    std::shared_ptr<const Game> game) {
  return std::make_shared<OracleEvaluator>(
      game, game->MakeObserver(kInfoStateObsType, {}));
}

}  // namespace papers_with_code
}  // namespace open_spiel
