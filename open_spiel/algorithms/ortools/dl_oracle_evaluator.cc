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

#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

// This file contains an implementation of a counterfactually optimal
// value function (that computes values for counterfactually optimal extensions
// of trunk strategies). See [1] for details.
//
// It uses sequence-form linear program to find the Nash equilibrium strategy
// for each player, one at a time, by reformulating it to be a value-solving
// subgame. The resulting strategies correspond to mutual best responses.
//
// However, counterfactually optimal extensions must be mutual *counterfactual*
// best responses. This is a subtle, but important distinction. This refinement
// is required because we'd like to use these values for CFR iterations in the
// trunk, and that requires the counterfactually optimal value functions. Thus
// we make post-processing of the SF-LP strategies, as briefly outlined in [1].
//
// Here are the more specific steps for the post-processing procedure:
//
// 0. Compute the mutual best-responding strategies by using the value-solving
// subgames.
//
// 1. For each player compute a "gradient" `g` for the terminals `z` of its
// infostate tree:
//   g(z) = u(z) * pi^{sf_lp}_opponent(z) * range_opponent(z) * pi_chance(z),
//
// i.e. terminal utility multiplied by opponent's reach that includes the
// opponent's range (based on the opponent's realization plan).
//
// 2. For all infostates within infostate-subtrees that have a non-zero range,
// compute behavioral strategies and save them into a policy table. These are
// already valid final strategies and they will not change.
//
// 3. For each root infostate that has a zero range, compute CBR bottom-up based
// on the "gradient" `g`. At each decision infostate choose a pure behavioral
// strategy that maximizes the cfvs of the children nodes, store this strategy
// in the policy table and propagate an updated cfv up the tree.
//
// 4. Now the policy table is completed for the entire infostate tree, and
// consists of two mutually best-responding strategies. We will compute the
// expected utilities of roots of the subgame under this profile.
//
// 5. We compute a weighted sum of the roots utilities according to opponent's
// ranges and chance reach probability (to those roots). This results in the
// final counterfactually optimal values.
//
// [1] Value Functions for Depth-Limited Solving in Imperfect-Information Games
// Vojtěch Kovařík, Dominik Seitz, Viliam Lisý, Jan Rudolf, Shuo Sun, Karel Ha

namespace open_spiel {
namespace algorithms {
namespace ortools {

namespace opres = operations_research;

OracleEvaluator::OracleEvaluator(std::shared_ptr<const Game> game,
                                 std::shared_ptr<Observer> infostate_observer)
    : game(std::move(game)),
      infostate_observer(std::move(infostate_observer)) {}

// Needs to have LP specified everywhere else using standard SF-LP.
void RefineSpecToValueSolvingSubgame(
    const InfostateNode& player_root,
    absl::Span<const double> range,
    std::unordered_map<const InfostateNode*, SolverVariables>& lp_spec) {
  SPIEL_CHECK_EQ(player_root.num_children(), range.size());

  // Set the reach probabilities of empty sequences that would correspond to
  // the trunk's leaf nodes to be equal to the player's range at those nodes.
  for (int i = 0; i < range.size(); ++i) {
    const InfostateNode* root_child_node = player_root.child_at(i);
    const SolverVariables& node_vars = lp_spec.at(root_child_node);
    SPIEL_CHECK_TRUE(node_vars.var_reach_prob);

    opres::MPConstraint* ct = node_vars.ct_parent_reach_prob;
    SPIEL_CHECK_TRUE(ct);
    ct->Clear();
    ct->SetLB(range[i]);
    ct->SetUB(range[i]);
    ct->SetCoefficient(node_vars.var_reach_prob, 1);
  }
}

std::vector<double> ComputeGradient(
    const std::vector<InfostateNode*>& player_terminals,
    const std::map<const InfostateNode*, const InfostateNode*>& terminal_map,
    std::unordered_map<const InfostateNode*, SolverVariables>& opponent_spec) {
  std::vector<double> gradient(player_terminals.size(), 0.);
  for (int i = 0; i < player_terminals.size(); ++i) {
    const InfostateNode* player_node = player_terminals[i];
    const InfostateNode* opponent_node = terminal_map.at(player_node);
    operations_research::MPVariable* opponent_reach =
        opponent_spec.at(opponent_node).var_reach_prob;
    SPIEL_CHECK_TRUE(opponent_reach);
    gradient[i] = player_terminals[i]->terminal_utility()
        * player_terminals[i]->terminal_chance_reach_prob()
          // Includes both pi^{sf_lp}_opponent(z) * range_opponent(z)
        * opponent_reach->solution_value();
  }
  return gradient;
}

DecisionVector<std::vector<double>> ComputeCfBestResponse(
    const InfostateTree& tree, absl::Span<double> cf_gradient,
    std::unordered_map<const InfostateNode*, SolverVariables>& player_spec) {
  DecisionVector<std::vector<double>> strategy(&tree);
  BottomUp(
      tree, cf_gradient,
      [&](DecisionId id, absl::Span<const double> loss) -> void {
          const InfostateNode* node = tree.decision_infostate(id);
          strategy[id] = std::vector(node->num_children(), 0.);
          auto node_reach = player_spec[node].var_reach_prob;
          SPIEL_CHECK_TRUE(node_reach);

          if (node_reach->solution_value() > 0) {
            size_t i = 0;
            for (InfostateNode* child : node->child_iterator()) {
              auto child_reach = player_spec[child].var_reach_prob;
              SPIEL_CHECK_TRUE(child_reach);
              strategy[id][i++] = child_reach->solution_value()
                  / node_reach->solution_value();
            }
          } else {
            auto iter_min = std::min_element(loss.begin(), loss.end());
            const size_t best_response = std::distance(loss.begin(), iter_min);
            strategy[id][best_response] = 1.;
          }
      },
      /*policy_fn=*/[&](DecisionId id) -> std::vector<double> {
          return strategy[id];
      });
  return strategy;
}

double ComputeExpectedUtility(HistoryNode* node, const TabularPolicy& policy) {
  if (node->GetType() == StateType::kTerminal) {
    return node->GetValue();
  }
  if (node->GetType() == StateType::kChance) {
    double value = 0.;
    for (Action a : node->GetChildActions()) {
      const auto& [prob, child] = node->GetChild(a);
      value += prob * ComputeExpectedUtility(child, policy);
    }
    return value;
  }
  if (node->GetType() == StateType::kDecision) {
    double value = 0.;
    auto state_policy = policy.GetStatePolicy(node->GetInfoState());
    for (Action action : node->GetChildActions()) {
      const auto& [fixed_prob, child] = node->GetChild(action);
      SPIEL_CHECK_EQ(fixed_prob, 1.);
      const double prob = GetProb(state_policy, action);
      value += prob * ComputeExpectedUtility(child, policy);
    }
    return value;
  }
  SpielFatalError("Exhausted pattern match!");
}

void OracleEvaluator::EvaluatePublicState(
    dlcfr::LeafPublicState* s, dlcfr::PublicStateContext* context) const {
  SPIEL_CHECK_TRUE(context);
  auto* oracle_context =
      open_spiel::down_cast<OraclePublicStateContext*>(context);
  std::array<SequenceFormLpSolver, 2>& solvers = oracle_context->solvers;

  // Check pointers for the trees are still the same.
  SPIEL_CHECK_EQ(solvers[0].trees()[0].get(), solvers[1].trees()[0].get());
  SPIEL_CHECK_EQ(solvers[0].trees()[1].get(), solvers[1].trees()[1].get());
  const std::vector<std::shared_ptr<InfostateTree>>& trees = solvers[0].trees();

  // Step 0. Construct the value-solving subgame for each player and solve it.
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(s->leaf_nodes[pl].size(), s->ranges[pl].size());
    RefineSpecToValueSolvingSubgame(trees[pl]->root(), s->ranges[pl],
                                    solvers[pl].lp_specification());
    // Run the solver!
    solvers[pl].Solve();
  }

  // Step 1. Compute the gradients of counterfactual values.
  std::vector<std::vector<double>> cf_gradients;
  for (int pl = 0; pl < 2; ++pl) {
    cf_gradients.push_back(ComputeGradient(
        /*player_terminals=*/trees[pl]->leaf_nodes(),
        /*terminal_map=*/solvers[pl].terminal_bijection().association(pl),
        /*opponent_spec=*/solvers[1 - pl].lp_specification()));
  }

  // Step 2. and 3.: Compute the behavioral strategies.

  // Resulting Nash equilibrium refinement which consists of a pair of
  // strategies that are mutual counterfactual best responses.
  TabularPolicy mutual_cbrs;
  for (int pl = 0; pl < 2; ++pl) {
    DecisionVector<std::vector<double>> cf_br = ComputeCfBestResponse(
        /*tree=*/*trees[pl],
        /*cf_gradient=*/absl::MakeSpan(cf_gradients[pl]),
        /*player_spec=*/solvers[pl].lp_specification());
    for (DecisionId id : trees[pl]->AllDecisionIds()) {
      const InfostateNode* node = trees[pl]->decision_infostate(id);
      std::vector<std::pair<Action, double>> policy;
      Zip(node->legal_actions().begin(), node->legal_actions().end(),
          cf_br[id].begin(), policy);
      mutual_cbrs.SetStatePolicy(node->infostate_string(), policy);
    }
  }

  // Step 4. Compute expected utilities of subgame histories (for player 0).
  std::map<std::string, /*utility=*/double> expected_utilities;
  for (HistoryTree& h : oracle_context->subgame_histories) {
    expected_utilities[h.Root()->GetHistory()] =
        ComputeExpectedUtility(h.Root(), mutual_cbrs);
  }

  // Step 5. Compute final counterfactually optimal values.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < s->leaf_nodes[pl].size(); ++i) {
      const InfostateNode* node = s->leaf_nodes[pl][i];
      s->values[pl][i] = 0.;
      for (int k = 0; k < node->corresponding_states_size(); ++k) {
        const std::unique_ptr<State>& state = node->corresponding_states()[k];
        const std::string state_str = state->ToString();
        const size_t j = oracle_context->subgame_ranges.at(state_str)[1 - pl];
        const double opponent_range = s->ranges[1 - pl][j];
        const double chance_range = node->corresponding_chance_reach_probs()[k];
        const double sign = 1 - 2 * pl;
        s->values[pl][i] += expected_utilities.at(state_str)
                          * opponent_range * chance_range * sign;
      }
    }
  }
  return;
}

std::unique_ptr<dlcfr::PublicStateContext> OracleEvaluator::CreateContext(
    const dlcfr::LeafPublicState& leaf_state) const {
  std::vector<std::shared_ptr<InfostateTree>> trees = {
      MakeInfostateTree(leaf_state.leaf_nodes[0]),
      MakeInfostateTree(leaf_state.leaf_nodes[1])
  };
  std::array<SequenceFormLpSolver, 2> solvers = {
      SequenceFormLpSolver(trees),
      SequenceFormLpSolver(trees)
  };
  for (int pl = 0; pl < 2; ++pl) {
    solvers[pl].SpecifyLinearProgram(pl);
  }

  std::vector<HistoryTree> subgame_histories;
  std::map<std::string, std::array<size_t, 2>> subgame_ranges;
  // This is just an estimate of the typical size. Probably bigger.
  subgame_histories.reserve(leaf_state.leaf_nodes[0].size() * 8);
  for (int i = 0; i < leaf_state.leaf_nodes[0].size(); ++i) {
    const InfostateNode* leaf_node = leaf_state.leaf_nodes[0][i];
    for (const std::unique_ptr<State>& s : leaf_node->corresponding_states()) {
      subgame_histories.emplace_back(std::move(s->Clone()), /*player_id=*/0);
      subgame_ranges[s->ToString()][0] = i;
    }
  }
  size_t num_histories = subgame_ranges.size();
  for (int j = 0; j < leaf_state.leaf_nodes[1].size(); ++j) {
    const InfostateNode* leaf_node = leaf_state.leaf_nodes[1][j];
    for (const std::unique_ptr<State>& s : leaf_node->corresponding_states()) {
      subgame_ranges[s->ToString()][1] = j;
      --num_histories;
    }
  }
  SPIEL_DCHECK_EQ(num_histories, 0);

  return std::make_unique<OraclePublicStateContext>(
      std::move(solvers), std::move(subgame_histories), std::move(subgame_ranges));
}

void RecursivelyRefineSpecFixStrategyWithPolicy(
    const InfostateNode* node,
    const Policy& fixed_policy,
    SequenceFormLpSolver* solver) {
  if (node->type() == kDecisionInfostateNode) {
    ActionsAndProbs local_policy =
        fixed_policy.GetStatePolicy(node->infostate_string());
    if (!local_policy.empty()) {  // Fix policy at this node!
      SPIEL_DCHECK_EQ(local_policy.size(), node->num_children());
      std::unordered_map<const InfostateNode*, SolverVariables>& lp_spec =
          solver->lp_specification();

      for (int i = 0; i < node->num_children(); ++i) {
        const InfostateNode* child = node->child_at(i);
        SPIEL_DCHECK_EQ(node->legal_actions()[i], local_policy[i].first);
        const double prob = local_policy[i].second;
        opres::MPConstraint* ct = solver->solver()->MakeRowConstraint(0., 0.);
        ct->SetCoefficient(lp_spec[node].var_reach_prob, prob);
        ct->SetCoefficient(lp_spec[child].var_reach_prob, -1.);
      }
    }
  }

  for (const InfostateNode* child : node->child_iterator()) {
    RecursivelyRefineSpecFixStrategyWithPolicy(child, fixed_policy, solver);
  }
}

double ComputeRootValueWhileFixingStrategy(
    SequenceFormLpSolver* solver, const Policy& fixed_policy,
    Player fixed_player) {
  solver->SpecifyLinearProgram(fixed_player);
  RecursivelyRefineSpecFixStrategyWithPolicy(
      solver->trees()[fixed_player]->mutable_root(), fixed_policy, solver);
  return solver->Solve();
}

double TrunkExploitability(SequenceFormLpSolver* solver,
                           const Policy& trunk_policy) {
  return (- ComputeRootValueWhileFixingStrategy(solver, trunk_policy, 0)
          - ComputeRootValueWhileFixingStrategy(solver, trunk_policy, 1)) / 2.;
}

double TrunkPlayerExploitability(
    SequenceFormLpSolver* solver, const Policy& trunk_policy, Player p,
    absl::optional<double> maybe_game_value) {
  double game_value;
  if (maybe_game_value.has_value()) {
    game_value = maybe_game_value.value();
  } else {
    solver->SpecifyLinearProgram(Player{0});
    game_value = solver->Solve();
  }
  // Switch sign appropriately - game value is defined for player 0!
  const double root_value = game_value * (1 - 2*p);
  const double fixed_value =
      ComputeRootValueWhileFixingStrategy(solver, trunk_policy, p);
  return root_value - fixed_value;
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
