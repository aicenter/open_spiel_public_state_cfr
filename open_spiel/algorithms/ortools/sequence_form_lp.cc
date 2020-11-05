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

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "ortools/linear_solver/linear_solver.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

namespace opres = operations_research;

namespace {

constexpr double kErrorTolerance = 1e-10;

class SolverNode;
using SolverTree = InfostateTree<SolverNode>;

class SolverNode : public InfostateNode</*Self=*/SolverNode> {
 public:
  SolverNode(const SolverTree& tree, SolverNode* parent, int incoming_index,
          InfostateNodeType type, const std::string& infostate_string,
          double terminal_utility, double terminal_chn_reach_prob,
          const State* originating_state)
    : InfostateNode<SolverNode>(
      tree, parent, incoming_index, type, infostate_string,
      terminal_utility, terminal_chn_reach_prob, originating_state)  {}
};

struct SolverData {
  // Variables needed for solving the LP. We will create these later.
  // Doing this in the constructor would require dependecy injection of the
  // solver through the tree, which is unnecessarily complicated.
  // The pointers may become obsolete by design: they should be accessed only
  // during the computation of the solution.
  operations_research::MPVariable* var_cf_value_ = nullptr;
  operations_research::MPVariable* var_reach_prob_ = nullptr;
  // Store the solver solutions for use once the solver goes out of scope.
  double sol_cf_value_;
  double sol_reach_prob_;
};

template<class T>
struct BijectiveContainer {
  std::map<T, T> x2y;
  std::map<T, T> y2x;

  void put(std::pair<T, T> xy) {
    const T& x = xy.first;
    const T& y = xy.second;
    SPIEL_CHECK_TRUE(x2y.find(x) == x2y.end());
    SPIEL_CHECK_TRUE(y2x.find(y) == y2x.end());
    x2y[x] = y;
    y2x[y] = x;
  }

  const std::map<T, T>& association(int direction) const {
    SPIEL_CHECK_TRUE(direction == 0 || direction == 1);
    if (direction == 0) return x2y;
    else return y2x;
  }
};

template<class Node>
BijectiveContainer<const Node*> ConnectTerminals(
    const InfostateTree<Node>& tree_a, const InfostateTree<Node>& tree_b) {
  BijectiveContainer<const Node*> out;

  using History = absl::Span<const Action>;
  std::map<History, const Node*> history_map;
  for (const Node& node_b : tree_b.leaves_iterator()) {
    history_map[node_b.TerminalHistory()] = &node_b;
  }

  for (const Node& node_a : tree_a.leaves_iterator()) {
    const Node* node_b = history_map[node_a.TerminalHistory()];
    out.put({&node_a, node_b});
  }
  return out;
}

void SpecifyReachProbs(opres::MPSolver* solver, std::unordered_map<const SolverNode*, SolverData>& data_table, SolverNode* node) {
  data_table[node].var_reach_prob_ = solver->MakeNumVar(
      /*lb=*/0.0, /*ub=*/1.0, absl::StrCat("rp_", node->ToString()));

  if (node->type() == kTerminalInfostateNode)
    return;  // Nothing to do.
  if (node->type() == kObservationInfostateNode) {
    for (SolverNode& child : node->child_iterator()) {
      SpecifyReachProbs(solver, data_table, &child);

      // Equality constraint: parent = child
      opres::MPConstraint* ct = solver->MakeRowConstraint(
          /*lb=*/0, /*ub=*/0,
          absl::StrCat("rp_", node->ToString(), "_", child.ToString()));
      ct->SetCoefficient(data_table[node].var_reach_prob_, -1);
      ct->SetCoefficient(data_table[&child].var_reach_prob_, 1);
    }
    return;
  }
  if (node->type() == kDecisionInfostateNode) {
    // Equality constraint: parent = sum of children
    opres::MPConstraint* ct = solver->MakeRowConstraint(
        /*lb=*/0, /*ub=*/0, absl::StrCat("rp_", node->ToString()));
    ct->SetCoefficient(data_table[node].var_reach_prob_, -1);
    for (SolverNode& child : node->child_iterator()) {
      SpecifyReachProbs(solver, data_table, &child);
      ct->SetCoefficient(data_table[&child].var_reach_prob_, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void SpecifyCfValues(
    opres::MPSolver* solver, std::unordered_map<const SolverNode*, SolverData>& data_table, SolverNode* node,
    const std::map<const SolverNode*, const SolverNode*>& terminal_map) {
  data_table[node].var_cf_value_ = solver->MakeNumVar(
      /*lb=*/-opres::MPSolver::infinity(),
      /*ub=*/opres::MPSolver::infinity(),
             absl::StrCat("cf_", node->ToString()));

  if (node->type() == kDecisionInfostateNode) {
    for (SolverNode& child : node->child_iterator()) {
      SpecifyCfValues(solver, data_table, &child, terminal_map);
      opres::MPConstraint* ct = solver->MakeRowConstraint(
          absl::StrCat("cf_", node->ToString(), "_", child.ToString()));
      ct->SetUB(0.);
      ct->SetCoefficient(data_table[node].var_cf_value_, -1);
      ct->SetCoefficient(data_table[&child].var_cf_value_, 1);
    }
    return;
  }

  opres::MPConstraint* ct = solver->MakeRowConstraint(
      absl::StrCat("cf_", node->ToString()));
  ct->SetUB(0.);
  ct->SetCoefficient(data_table[node].var_cf_value_, -1);

  if (node->type() == kTerminalInfostateNode) {
    const SolverNode* opponent_node = terminal_map.at(node);
    const double value =
        node->terminal_utility() * node->terminal_chance_reach_prob();
    // Terminal value constraint comes from the opponent.
    ct->SetCoefficient(data_table[opponent_node].var_reach_prob_, value);
    return;
  }
  if (node->type() == kObservationInfostateNode) {
    // Value constraint: sum of children = parent
    ct->SetLB(0.);
    for (SolverNode& child : node->child_iterator()) {
      SpecifyCfValues(solver, data_table, &child, terminal_map);
      ct->SetCoefficient(data_table[&child].var_cf_value_, 1);
    }
    return;
  }

  SpielFatalError("Exhausted pattern match!");
}

void CollectReachProbsSolutions(std::unordered_map<const SolverNode*, SolverData>& data_table, SolverNode* node) {
  data_table[node].sol_reach_prob_ = data_table[node].var_reach_prob_->solution_value();
  for (SolverNode& child : node->child_iterator()) {
    CollectReachProbsSolutions(data_table, &child);
  }
}

void CollectCfValuesSolutions(std::unordered_map<const SolverNode*, SolverData>& data_table, SolverNode* node) {
  data_table[node].sol_cf_value_ = data_table[node].var_cf_value_->solution_value();
  for (SolverNode& child : node->child_iterator()) {
    CollectCfValuesSolutions(data_table, &child);
  }
}

// Useful for debugging.
void PrintProblemSpecification(const opres::MPSolver& solver) {
  const std::vector<opres::MPVariable*>& variables = solver.variables();
  const std::vector<opres::MPConstraint*>& constraints = solver.constraints();
  const opres::MPObjective& objective = solver.Objective();

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
  for (auto& ct : solver.constraints()) {
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

void SolveForPlayer(
    Player pl,
    const std::array<std::unique_ptr<SolverTree>, 2>& solver_trees,
    std::unordered_map<const SolverNode*, SolverData>& data_table,
    const std::map<const SolverNode*, const SolverNode*>& terminal_map) {
  // 1. Create the linear solver (with the GLOP backend).
  opres::MPSolver solver("sf_lp", opres::MPSolver::GLOP_LINEAR_PROGRAMMING);

  // 2. Recursively create variables and constraints.
  SpecifyReachProbs(&solver, data_table, solver_trees[pl]->mutable_root());
  SpecifyCfValues(&solver, data_table, solver_trees[1 - pl]->mutable_root(),
                  terminal_map);
  // Add constraints for root node.
  SolverNode* root_node = solver_trees[pl]->mutable_root();
  SolverData& root_data = data_table.at(root_node);
  root_data.var_reach_prob_->SetLB(1.);
  root_data.var_reach_prob_->SetUB(1.);

  // 3. Solve the problem.
  opres::MPObjective* const objective = solver.MutableObjective();
  objective->SetCoefficient(
      data_table[solver_trees[1 - pl]->mutable_root()].var_cf_value_, 1);
  objective->SetMinimization();

  // Keeping this around for debugging.
//  PrintProblemSpecification(solver);
  opres::MPSolver::ResultStatus status = solver.Solve();
  SPIEL_CHECK_EQ(status, opres::MPSolver::ResultStatus::OPTIMAL);

  // 4. Save the solved values - solver will go out of scope.
  CollectReachProbsSolutions(data_table, solver_trees[pl]->mutable_root());
  CollectCfValuesSolutions(data_table, solver_trees[1 - pl]->mutable_root());
  SPIEL_CHECK_EQ(
      objective->Value(),
      data_table[solver_trees[1 - pl]->mutable_root()].sol_cf_value_);
}

void CollectTabularPolicy(TabularPolicy* policy, std::unordered_map<const SolverNode*, SolverData>& data_table, const SolverNode& node) {
  if (node.type() == kDecisionInfostateNode) {
    absl::Span<const Action> actions = node.legal_actions();
    SPIEL_CHECK_EQ(actions.size(), node.num_children());
    ActionsAndProbs state_policy;
    state_policy.reserve(node.num_children());
    double rp_sum = 0.;
    for (int i = 0; i < actions.size(); ++i) {
      rp_sum += data_table[node.child_at(i)].sol_reach_prob_;
    }
    for (int i = 0; i < actions.size(); ++i) {
      double prob;
      if (rp_sum) {
        prob = data_table[node.child_at(i)].sol_reach_prob_ / rp_sum;
      } else {
        // If the infostate is unreachable, the strategy is not defined.
        // However some code in the library may require having the strategy,
        // so we just put an uniform strategy here.
        prob = 1. / actions.size();
      }
      state_policy.push_back({actions[i], prob});
    }
    policy->SetStatePolicy(node.infostate_string(), state_policy);
  }

  for (const SolverNode& child : node.child_iterator()) {
    CollectTabularPolicy(policy, data_table, child);
  }
}

}  // namespace

std::unique_ptr<ZeroSumSequentialGameSolution> SolveZeroSumSequentialGame(
    std::shared_ptr<Observer> infostate_observer,
    const std::vector<const State*>& start_states,
    const std::vector<float>& chance_reach_probs,
    std::optional<int> solve_only_player,
    bool collect_tabular_policy) {

  // 1. Construct infoset trees for the game.
  std::array<std::unique_ptr<SolverTree>, 2> solver_trees;
  for (int pl = 0; pl < 2; ++pl) {
    solver_trees[pl] = std::make_unique<SolverTree>(
        start_states, chance_reach_probs, infostate_observer, pl);
  }
  std::unordered_map<const SolverNode*, SolverData> data_table;

  // 2. Connect the terminals - now we can go from one tree to the other
  //    via pointers.
  BijectiveContainer<const SolverNode*> terminal_map = ConnectTerminals(
      *solver_trees[0], *solver_trees[1]);

  // 3. Solve for players.
  if (solve_only_player) {
    int pl = solve_only_player.value();
    SolveForPlayer(pl, solver_trees, data_table, terminal_map.association(1 - pl));
  } else {
    for (int pl = 0; pl < 2; ++pl) {
      SolveForPlayer(pl, solver_trees, data_table, terminal_map.association(1 - pl));
    }
    // Check zero-sum-ness of the root values.
    SPIEL_CHECK_FLOAT_NEAR(
        data_table[solver_trees[0]->mutable_root()].sol_cf_value_,
        -data_table[solver_trees[1]->mutable_root()].sol_cf_value_,
        kErrorTolerance);
  }

  // 4. Collect the requested results.
  auto sol = std::make_unique<ZeroSumSequentialGameSolution>();
  // Always collect game value.
  sol->game_value = data_table[solver_trees[0]->mutable_root()].sol_cf_value_;
  if (collect_tabular_policy) {
    if (solve_only_player) {
      int pl = solve_only_player.value();
      CollectTabularPolicy(&sol->policy, data_table, solver_trees[pl]->root());
    } else {
      for (int pl = 0; pl < 2; ++pl) {
        CollectTabularPolicy(&sol->policy, data_table, solver_trees[pl]->root());
      }
    }
  }
  return sol;
}

std::unique_ptr<ZeroSumSequentialGameSolution> SolveZeroSumSequentialGame(
    const Game& game) {
  std::unique_ptr<State> state = game.NewInitialState();
  std::vector<const State*> starting_states;
  starting_states.push_back(state.get());
  std::vector<float> chance_reach_probs = {1.};

  return SolveZeroSumSequentialGame(game.MakeObserver(kInfoStateObsType, {}),
                                    starting_states, chance_reach_probs);
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
