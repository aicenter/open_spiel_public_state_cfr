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

#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

#include <map>
#include <memory>
#include <utility>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "ortools/linear_solver/linear_solver.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

namespace opres = operations_research;

SequenceFormLpSolver::SequenceFormLpSolver(const Game& game)
    : SequenceFormLpSolver({
                               MakeInfostateTree(game, 0),
                               MakeInfostateTree(game, 1),
                           }) {}

SequenceFormLpSolver::SequenceFormLpSolver(
    std::array<std::shared_ptr<InfostateTree>, 2> solver_trees,
    const std::string& solver_id)
    : solver_trees_(std::move(solver_trees)),
      terminal_bijection_(
          ConnectTerminals(*solver_trees_[0], *solver_trees_[1])),
      solver_(MPSolver::CreateSolver(solver_id)), lp_spec_() {
  SPIEL_CHECK_TRUE(solver_);
}


void SequenceFormLpSolver::SpecifyReachProbsConstraints(InfostateNode* node) {
  lp_spec_[node].var_reach_prob = solver_->MakeNumVar(
      /*lb=*/0.0, /*ub=*/1., "");

  if (node->type() == kTerminalInfostateNode)
    return;  // Nothing to do.
  if (node->type() == kObservationInfostateNode) {
    for (InfostateNode* child : node->child_iterator()) {
      SpecifyReachProbsConstraints(child);

      // Equality constraint: parent = child
      opres::MPConstraint* ct
          = lp_spec_[child].ct_parent_reach_prob
          = solver_->MakeRowConstraint(/*lb=*/0, /*ub=*/0, "");
      ct->SetCoefficient(lp_spec_[node].var_reach_prob, -1);
      ct->SetCoefficient(lp_spec_[child].var_reach_prob, 1);
    }
    return;
  }
  if (node->type() == kDecisionInfostateNode) {
    // Equality constraint: parent = sum of children
    opres::MPConstraint* ct
        = lp_spec_[node].ct_child_reach_prob
        = solver_->MakeRowConstraint(/*lb=*/0, /*ub=*/0, "");
    ct->SetCoefficient(lp_spec_[node].var_reach_prob, -1);
    for (InfostateNode* child : node->child_iterator()) {
      SpecifyReachProbsConstraints(child);
      ct->SetCoefficient(lp_spec_[child].var_reach_prob, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void SequenceFormLpSolver::SpecifyCfValuesConstraints(InfostateNode* node) {
  lp_spec_[node].var_cf_value = solver_->MakeNumVar(
      /*lb=*/-opres::MPSolver::infinity(),
      /*ub=*/opres::MPSolver::infinity(), "");

  if (node->type() == kDecisionInfostateNode) {
    for (InfostateNode* child : node->child_iterator()) {
      SpecifyCfValuesConstraints(child);
      opres::MPConstraint* ct
          = lp_spec_[child].ct_parent_cf_value
          = solver_->MakeRowConstraint();
      ct->SetUB(0.);
      ct->SetCoefficient(lp_spec_[node].var_cf_value, -1);
      ct->SetCoefficient(lp_spec_[child].var_cf_value, 1);
    }
    return;
  }

  opres::MPConstraint* ct
      = lp_spec_[node].ct_child_cf_value
      = solver_->MakeRowConstraint();
  ct->SetUB(0.);
  ct->SetCoefficient(lp_spec_[node].var_cf_value, -1);

  if (node->type() == kTerminalInfostateNode) {
    const std::map<const InfostateNode*, const InfostateNode*>& terminal_map =
        terminal_bijection_.association(node->tree().acting_player());
    const InfostateNode* opponent_node = terminal_map.at(node);
    const double value =
        node->terminal_utility() * node->terminal_chance_reach_prob();
    // Terminal value constraint comes from the opponent.
    ct->SetCoefficient(lp_spec_[opponent_node].var_reach_prob, value);
    return;
  }
  if (node->type() == kObservationInfostateNode) {
    // Value constraint: sum of children = parent
    ct->SetLB(0.);
    for (InfostateNode* child : node->child_iterator()) {
      SpecifyCfValuesConstraints(child);
      ct->SetCoefficient(lp_spec_[child].var_cf_value, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void SequenceFormLpSolver::SpecifyRootConstraints(InfostateNode* root_node) {
  SPIEL_CHECK_TRUE(root_node->is_root_node());
  SolverVariables& root_data = lp_spec_.at(root_node);
  root_data.var_reach_prob->SetLB(1.);
  root_data.var_reach_prob->SetUB(1.);
}

void SequenceFormLpSolver::SpecifyObjective(InfostateNode* root_node) {
  opres::MPObjective* const objective = solver_->MutableObjective();
  objective->SetCoefficient(lp_spec_[root_node].var_cf_value, 1);
  objective->SetMinimization();
}

void SequenceFormLpSolver::ClearSpecification() {
  solver_->Clear();
  for (auto&[node, vars] : lp_spec_) {
    vars.var_cf_value = nullptr;
    vars.var_reach_prob = nullptr;
    vars.ct_child_cf_value = nullptr;
    vars.ct_parent_cf_value = nullptr;
    vars.ct_child_reach_prob = nullptr;
    vars.ct_parent_reach_prob = nullptr;
  }
}

void SequenceFormLpSolver::SpecifyLinearProgram(Player pl) {
  SPIEL_CHECK_TRUE(pl == 0 || pl == 1);
  ClearSpecification();
  SpecifyReachProbsConstraints(solver_trees_[pl]->mutable_root());
  SpecifyRootConstraints(solver_trees_[pl]->mutable_root());
  SpecifyCfValuesConstraints(solver_trees_[1 - pl]->mutable_root());
  SpecifyObjective(solver_trees_[1 - pl]->mutable_root());
}

double SequenceFormLpSolver::Solve() {
  opres::MPSolver::ResultStatus status = solver_->Solve();
  SPIEL_CHECK_EQ(status, opres::MPSolver::ResultStatus::OPTIMAL);
  return -solver_->Objective().Value();
}

TabularPolicy SequenceFormLpSolver::OptimalPolicy(Player for_player) {
  SPIEL_CHECK_TRUE(for_player == 0 || for_player == 1);
  const InfostateTree* tree = solver_trees_[for_player].get();
  TabularPolicy policy;
  for (DecisionId id : tree->AllDecisionIds()) {
    const InfostateNode* node = tree->decision_infostate(id);
    absl::Span<const Action> actions = node->legal_actions();
    SPIEL_CHECK_EQ(actions.size(), node->num_children());
    ActionsAndProbs state_policy;
    state_policy.reserve(node->num_children());
    double rp_sum = 0.;
    for (int i = 0; i < actions.size(); ++i) {
      rp_sum += lp_spec_[node->child_at(i)].var_reach_prob->solution_value();
    }
    for (int i = 0; i < actions.size(); ++i) {
      double prob;
      if (rp_sum) {
        prob = lp_spec_[node->child_at(i)].var_reach_prob->solution_value()
             / rp_sum;
      } else {
        // If the infostate is unreachable, the strategy is not defined.
        // However some code in the library may require having the strategy,
        // so we just put an uniform strategy here.
        prob = 1. / actions.size();
      }
      state_policy.push_back({actions[i], prob});
    }
    policy.SetStatePolicy(node->infostate_string(), state_policy);
  }
  return policy;
}

SfStrategy SequenceFormLpSolver::OptimalSfStrategy(
    Player for_player) {
  SPIEL_CHECK_TRUE(for_player == 0 || for_player == 1);
  const InfostateTree* tree = solver_trees_[for_player].get();
  SfStrategy strategy(tree);
  for (SequenceId id : tree->AllSequenceIds()) {
    const InfostateNode* node = tree->observation_infostate(id);
    strategy[id] = lp_spec_[node].var_reach_prob->solution_value();
  }
  return strategy;
}

BijectiveContainer<const InfostateNode*>
ConnectTerminals(const InfostateTree& tree_a, const InfostateTree& tree_b) {
  BijectiveContainer<const InfostateNode*> out;

  using History = absl::Span<const Action>;
  std::map<History, const InfostateNode*> history_map;
  for (InfostateNode* node_b : tree_b.leaf_nodes()) {
    history_map[node_b->TerminalHistory()] = node_b;
  }

  for (InfostateNode* node_a : tree_a.leaf_nodes()) {
    const InfostateNode* node_b = history_map[node_a->TerminalHistory()];
    out.put({node_a, node_b});
  }
  return out;
}


void SequenceFormLpSolver::PrintProblemSpecification() {
  const std::vector<opres::MPVariable*>& variables = solver_->variables();
  const std::vector<opres::MPConstraint*>& constraints = solver_->constraints();
  const opres::MPObjective& objective = solver_->Objective();

  std::cout << "Objective:" << std::endl;
  if (objective.maximization()) std::cout << "max ";
  else std::cout << "min ";
  bool first_obj = true;
  for (int i = 0; i < variables.size(); ++i) {
    const double coef = objective.GetCoefficient(variables[i]);
    if (coef) {
      if (!first_obj) std::cout << "+ ";
      std::cout << coef << "*x" << i << " ";
      first_obj = false;
    }
  }
  std::cout << std::endl;

  std::cout << "Constraints:" << std::endl;
  for (auto& ct : solver_->constraints()) {
    std::cout << ct->lb() << " <= ";
    bool first_ct = true;
    for (int i = 0; i < variables.size(); ++i) {
      const double coef = ct->GetCoefficient(variables[i]);
      if (coef) {
        if (!first_ct) std::cout << "+ ";
        std::cout << coef << "*x" << i << " ";
        first_ct = false;
      }
    }
    std::cout << "<= " << ct->ub() << " (" << ct->name() << ")" << std::endl;
  }

  std::cout << "Variables:" << std::endl;
  for (int i = 0; i < variables.size(); i++) {
    const auto& var = variables[i];
    std::cout << var->lb() << " <= " << "x" << i << " <= " << var->ub()
              << " (" << var->name() << ")" << std::endl;
  }
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
