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
// subgame, with a specified range of the opponent. The resulting strategies
// correspond to mutual best responses.
//
// However, counterfactually optimal extensions must be mutual *counterfactual*
// best responses. This is a subtle, but important distinction. This refinement
// is required because we'd like to use these values for CFR iterations in the
// trunk, and that requires the counterfactually optimal value functions. Thus
// we make post-processing of the SF-LP strategies, as briefly outlined in [1].
//
// Note that throughout the comments / implementation a "player" refers refers
// to some player `i`, while "opponent" is the player `1-i`.
//
// Here are the more specific steps for the post-processing procedure:
//
// 0. Compute the mutual best-responding strategies by using the value-solving
// subgames, with a specified range of the opponent and the player playing into
// the subgame with probabilities (range) 1.
//
// 1. For each player compute a "gradient" `g` for the terminals `z` of its
// (full) infostate tree, i.e. the counterfactual values:
//
//   g(z) = u(z) * pi^{sf_lp}_opponent(z) * range_opponent(z) * pi_chance(z),
//
// i.e. terminal utility multiplied by chance, opponent's reach (based on the
// opponent's realization plan from the previously solved LP) and the opponent's
// range.
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
// consists of two mutually counterfactually-best-responding strategies. We will
// compute the expected utilities of roots of the subgame under this profile.
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

void RecursivelyUpdateTerminalUtilityConstraints(
    const InfostateNode* opponent_node, double opponent_range,
    std::unordered_map<const InfostateNode*, NodeSpecification>& node_spec,
    const std::map<const InfostateNode*, const InfostateNode*>& opponent_terminal_map) {
  if (opponent_node->type() == kTerminalInfostateNode) {
    const InfostateNode* player_node = opponent_terminal_map.at(opponent_node);
    SPIEL_CHECK_EQ(player_node->type(), kTerminalInfostateNode);
    SPIEL_CHECK_EQ(opponent_node->type(), kTerminalInfostateNode);
    SPIEL_CHECK_EQ(opponent_node->TerminalHistory(),
                   player_node->TerminalHistory());
    SPIEL_CHECK_EQ(opponent_node->terminal_chance_reach_prob(),
                   player_node->terminal_chance_reach_prob());

    opres::MPConstraint* ct = node_spec[opponent_node].ct_child_cf_value;
    SPIEL_CHECK_TRUE(ct);
    SPIEL_CHECK_EQ(ct->ub(), 0.);
    SPIEL_CHECK_TRUE(node_spec[opponent_node].var_cf_value);
    SPIEL_CHECK_EQ(ct->GetCoefficient(node_spec[opponent_node].var_cf_value), -1);
    SPIEL_CHECK_TRUE(node_spec[player_node].var_reach_prob);

    // Take care when changing this code: it is shared with ComputeGradient!
    const double value_weighted_by_opp_range
        = opponent_node->terminal_utility()
        * opponent_node->terminal_chance_reach_prob()
        * opponent_range;
    ct->SetCoefficient(node_spec[player_node].var_reach_prob,
                       value_weighted_by_opp_range);
    return;
  }

  for (const InfostateNode* opponent_child : opponent_node->child_iterator()) {
    RecursivelyUpdateTerminalUtilityConstraints(
        opponent_child, opponent_range, node_spec, opponent_terminal_map);
  }
}

void RefineSpecToValueSolvingSubgame(
    const InfostateNode* opponent_root,
    absl::Span<const double> opponent_range,
    const std::map<const InfostateNode*, const InfostateNode*>& opponent_terminal_map,
    std::unordered_map<const InfostateNode*, NodeSpecification>& player_spec) {
  SPIEL_CHECK_EQ(opponent_root->num_children(), opponent_range.size());

  // Set the reach probabilities of empty sequences that would correspond to
  // the trunk's leaf nodes to be equal to the player's range at those nodes.
  for (int i = 0; i < opponent_range.size(); ++i) {
    RecursivelyUpdateTerminalUtilityConstraints(
        opponent_root->child_at(i), opponent_range[i],
        player_spec, opponent_terminal_map);
  }
}

std::vector<double> ComputeGradient(
    const std::vector<InfostateNode*>& player_terminals,
    const std::map<const InfostateNode*, const InfostateNode*>& player_terminal_map,
    // We use player spec to get utility_weighted_by_opp_range.
    std::unordered_map<const InfostateNode*, NodeSpecification>& player_spec,
    std::unordered_map<const InfostateNode*, NodeSpecification>& opponent_spec) {
  std::vector<double> gradient(player_terminals.size(), 0.);
  for (int i = 0; i < player_terminals.size(); ++i) {
    const InfostateNode* player_node = player_terminals[i];
    const InfostateNode* opponent_node = player_terminal_map.at(player_node);

    operations_research::MPVariable* opponent_reach =
        opponent_spec.at(opponent_node).var_reach_prob;
    SPIEL_CHECK_TRUE(opponent_reach);

    opres::MPConstraint* ct = player_spec.at(opponent_node).ct_child_cf_value;
    SPIEL_CHECK_TRUE(ct);
    SPIEL_CHECK_TRUE(player_spec.at(player_node).var_reach_prob);

    // Avoid recalculating this again. This corresponds to:
    // = opponent_node->terminal_utility()
    // * opponent_node->terminal_chance_reach_prob()
    // * opponent_range;
    const double utility_weighted_by_opp_range =
        ct->GetCoefficient(player_spec[player_node].var_reach_prob);

    // As we compute best response for the player, we need to flip the sign,
    // as the utility was originally for the opponent.
    gradient[i] = - utility_weighted_by_opp_range
                * opponent_reach->solution_value();
  }
  return gradient;
}

// TODO: gradient as move
DecisionVector<std::vector<double>> RefineBestResponseToCfBestResponse(
    const InfostateTree& player_tree, absl::Span<double> player_cf_gradient,
    std::unordered_map<const InfostateNode*, NodeSpecification>& player_spec) {
  std::mt19937 mt;
  DecisionVector<std::vector<double>> strategy(&player_tree);
  BottomUp(
    player_tree, player_cf_gradient,
    /*observe_rewards_fn=*/[&](DecisionId id, absl::Span<const double> rewards)
    {
      const InfostateNode* node = player_tree.decision_infostate(id);
      SPIEL_CHECK_EQ(rewards.size(), node->num_children());
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
        auto iter_max = std::max_element(rewards.begin(), rewards.end());
        const double max_reward = *iter_max;
        std::vector<size_t> max_indices;
        for (int i = 0; i < rewards.size(); i++) {
          if (fabs(max_reward - rewards[i]) < 1e-10) max_indices.push_back(i);
        }
        std::uniform_int_distribution<int> dist(0, max_indices.size() - 1);
        const int resp_idx = dist(mt);
        strategy[id][resp_idx] = 1.;
      }
      return strategy[id];
    });
  return strategy;
}

double ComputeExpectedUtility(const HistoryNode* node,
                              const TabularPolicy& policy, Player for_player) {
  if (node->GetType() == StateType::kTerminal) {
    return node->GetUtility(for_player);
  }
  if (node->GetType() == StateType::kChance) {
    double value = 0.;
    for (Action a : node->GetChildActions()) {
      const auto& [chance_prob, child] = node->GetChild(a);
      value += chance_prob * ComputeExpectedUtility(child, policy, for_player);
    }
    return value;
  }
  if (node->GetType() == StateType::kDecision) {
    if (node->GetState()->IsSimultaneousNode()) {
      const int num_players = node->GetState()->NumPlayers();
      std::vector<ActionsAndProbs> state_policies;
      state_policies.reserve(num_players);
      for (int pl = 0; pl < num_players; ++pl) {
        state_policies.push_back(policy.GetStatePolicy(node->GetInfoState(pl)));
      }

      double value = 0.;
      for (Action flat_joint_action : node->GetChildActions()) {
        const auto&[chance_prob, child] = node->GetChild(flat_joint_action);
        SPIEL_CHECK_EQ(chance_prob, 1.);
        double prob = 1.;
        std::vector<Action> player_actions =
            node->action_view().FlatJointActionToActions(flat_joint_action);
        for (int pl = 0; pl < num_players; ++pl) {
          prob *= GetProb(state_policies[pl], player_actions[pl]);
        }
        value += prob * ComputeExpectedUtility(child, policy, for_player);
      }
      return value;

    } else {
      double value = 0.;
      auto state_policy = policy.GetStatePolicy(node->GetInfoState());
      for (Action action : node->GetChildActions()) {
        const auto&[chance_prob, child] = node->GetChild(action);
        SPIEL_CHECK_EQ(chance_prob, 1.);
        const double prob = GetProb(state_policy, action);
        value += prob * ComputeExpectedUtility(child, policy, for_player);
      }
      return value;
    }
  }
  SpielFatalError("Exhausted pattern match!");
}

void OracleEvaluator::EvaluatePublicState(
    dlcfr::LeafPublicState* s, dlcfr::PublicStateContext* context) const {
  SPIEL_CHECK_TRUE(context);
  auto* oracle_context =
      open_spiel::down_cast<OraclePublicStateContext*>(context);
  std::array<SequenceFormLpSpecification, 2>& specs =
      oracle_context->specifications;

  // Check pointers for the trees are still the same.
  SPIEL_CHECK_EQ(specs[0].trees()[0].get(), specs[1].trees()[0].get());
  SPIEL_CHECK_EQ(specs[0].trees()[1].get(), specs[1].trees()[1].get());
  SPIEL_DCHECK_EQ(specs[0].terminal_bijection().association(0),
                  specs[1].terminal_bijection().association(0));
  SPIEL_DCHECK_EQ(specs[0].terminal_bijection().association(1),
                  specs[1].terminal_bijection().association(1));
  const std::vector<std::shared_ptr<InfostateTree>>& trees = specs[0].trees();
  const BijectiveContainer<const InfostateNode*>& terminal_bijection =
      specs[0].terminal_bijection();

  // Step 0. Construct the value-solving subgame for each player and solve it.
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(s->leaf_nodes[pl].size(), s->ranges[pl].size());
    RefineSpecToValueSolvingSubgame(
        /*opponent_root=*/trees[1 - pl]->mutable_root(),
        /*opponent_range=*/s->ranges[1 - pl],
        /*opponent_terminal_map=*/terminal_bijection.association(1 - pl),
        /*player_spec=*/specs[pl].node_spec());
    // Run the solver!
    specs[pl].Solve();
  }

  // Step 1. Compute the "gradients" of counterfactual values.
  std::vector<std::vector<double>> cf_gradients;
  for (int pl = 0; pl < 2; ++pl) {
    cf_gradients.push_back(ComputeGradient(
        /*player_terminals=*/trees[pl]->leaf_nodes(),
        /*player_terminal_map=*/terminal_bijection.association(pl),
        /*player_spec=*/specs[pl].node_spec(),
        /*opponent_spec=*/specs[1 - pl].node_spec()));
  }

  // Step 2. and 3.: Compute the behavioral strategies.

  // Resulting Nash equilibrium refinement which consists of a pair of
  // strategies that are mutual counterfactual best responses.
  TabularPolicy mutual_cbrs;
  for (int pl = 0; pl < 2; ++pl) {
    DecisionVector<std::vector<double>> cf_br =
        RefineBestResponseToCfBestResponse(
            /*player_tree=*/*trees[pl],
            /*player_cf_gradient=*/absl::MakeSpan(cf_gradients[pl]),
            /*player_spec=*/specs[pl].node_spec());

    for (DecisionId id : trees[pl]->AllDecisionIds()) {
      const InfostateNode* player_node = trees[pl]->decision_infostate(id);
      std::vector<std::pair<Action, double>> policy;
      Zip(player_node->legal_actions().begin(),
          player_node->legal_actions().end(),
          cf_br[id].begin(), policy);
      mutual_cbrs.SetStatePolicy(player_node->infostate_string(), policy);
    }
  }

  // Step 4. Compute expected utilities of subgame histories.
  std::map<std::string, /*utility=*/double> expected_utilities;
  for (const HistoryTree& h : oracle_context->subgame_root_histories) {
    expected_utilities[h.Root().GetHistory()] =
        ComputeExpectedUtility(&h.Root(), mutual_cbrs, /*for_player*/0);
  }

  // Step 5. Compute final counterfactually optimal values.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < s->leaf_nodes[pl].size(); ++i) {
      const InfostateNode* player_node = s->leaf_nodes[pl][i];
      s->values[pl][i] = 0.;
      for (int k = 0; k < player_node->corresponding_states_size(); ++k) {
        const std::unique_ptr<State>& state =
            player_node->corresponding_states()[k];
        const std::string history_str = state->HistoryString();
        const size_t j =
            oracle_context->subgame_range_indexing.at(history_str)[1 - pl];
        const double opponent_range = s->ranges[1 - pl][j];
        const double chance_reach =
            player_node->corresponding_chance_reach_probs()[k];
        // We must change the sign appropriately because expected_utilities
        // are computed for player 0.
        const double sign = 1 - 2 * pl;
        s->values[pl][i] += sign * expected_utilities.at(history_str)
                          * opponent_range * chance_reach;
      }
    }
  }

//  std::cout << "public_id " << s->public_id << ": "
//            << "ranges: " << s->ranges[0][0] << " "
//            << "values: " << s->values[0][0] << "," << s->values[1][0] << " \n"
//            << "tensor: " << s->public_tensor << "\n"
//            ;
//  std::cout << mutual_cbrs.PolicyTable() << "\n";
  return;
}

std::unique_ptr<dlcfr::PublicStateContext> OracleEvaluator::CreateContext(
    const dlcfr::LeafPublicState& leaf_state) const {
  std::vector<std::shared_ptr<InfostateTree>> trees = {
      MakeInfostateTree(leaf_state.leaf_nodes[0]),
      MakeInfostateTree(leaf_state.leaf_nodes[1])
  };
  std::array<SequenceFormLpSpecification, 2> specifications = {
      SequenceFormLpSpecification(trees),
      SequenceFormLpSpecification(trees)
  };
  for (int pl = 0; pl < 2; ++pl) {
    specifications[pl].SpecifyLinearProgram(pl);
  }

  std::vector<HistoryTree> subgame_histories;
  std::map<std::string, std::array<size_t, 2>> subgame_ranges;
  // This is just an estimate of the typical size. Probably bigger.
  subgame_histories.reserve(leaf_state.leaf_nodes[0].size() * 8);
  for (int i = 0; i < leaf_state.leaf_nodes[0].size(); ++i) {
    const InfostateNode* leaf_node = leaf_state.leaf_nodes[0][i];
    for (const std::unique_ptr<State>& s : leaf_node->corresponding_states()) {
      SPIEL_CHECK_FALSE(s->IsTerminal());
      subgame_histories.emplace_back(s->Clone());
      subgame_ranges[s->HistoryString()][0] = i;
    }
  }
  size_t num_histories = subgame_ranges.size();
  for (int j = 0; j < leaf_state.leaf_nodes[1].size(); ++j) {
    const InfostateNode* leaf_node = leaf_state.leaf_nodes[1][j];
    for (const std::unique_ptr<State>& s : leaf_node->corresponding_states()) {
      subgame_ranges[s->HistoryString()][1] = j;
      --num_histories;
    }
  }
  SPIEL_DCHECK_EQ(num_histories, 0);

  return std::make_unique<OraclePublicStateContext>(
      std::move(specifications), std::move(subgame_histories), std::move(subgame_ranges));
}

void RecursivelyRefineSpecFixStrategyWithPolicy(
    const InfostateNode* player_node,
    const Policy& fixed_policy,
    SequenceFormLpSpecification* specification) {
  if (player_node->type() == kDecisionInfostateNode) {
    ActionsAndProbs local_policy =
        fixed_policy.GetStatePolicy(player_node->infostate_string());
    if (!local_policy.empty()) {  // Fix policy at this node!
      SPIEL_DCHECK_EQ(local_policy.size(), player_node->num_children());
      SPIEL_DCHECK_TRUE(IsValidProbDistribution(local_policy));
      std::unordered_map<const InfostateNode*, NodeSpecification>& node_spec =
          specification->node_spec();

      for (int i = 0; i < player_node->num_children(); ++i) {
        const InfostateNode* player_child = player_node->child_at(i);
        SPIEL_DCHECK_EQ(player_node->legal_actions()[i], local_policy[i].first);
        SPIEL_DCHECK_TRUE(node_spec[player_node].var_reach_prob);
        SPIEL_DCHECK_TRUE(node_spec[player_child].var_reach_prob);

        const double prob = local_policy[i].second;
        // Creates a constraint: prob * r(parent) = r(child)
        opres::MPConstraint* ct = specification->solver()->MakeRowConstraint(0., 0.);
        ct->SetCoefficient(node_spec[player_node].var_reach_prob, prob);
        ct->SetCoefficient(node_spec[player_child].var_reach_prob, -1.);
      }
    }
  }

  for (const InfostateNode* player_child : player_node->child_iterator()) {
    RecursivelyRefineSpecFixStrategyWithPolicy(
        player_child, fixed_policy, specification);
  }
}

double ComputeRootValueWhileFixingStrategy(
    SequenceFormLpSpecification* specification, const Policy& fixed_policy,
    Player fixed_player) {
  specification->SpecifyLinearProgram(fixed_player);
  RecursivelyRefineSpecFixStrategyWithPolicy(
      specification->trees()[fixed_player]->mutable_root(), fixed_policy, specification);
  return specification->Solve();
}

double TrunkExploitability(SequenceFormLpSpecification* spec,
                           const Policy& trunk_policy) {
  return (- ComputeRootValueWhileFixingStrategy(spec, trunk_policy, 0)
          - ComputeRootValueWhileFixingStrategy(spec, trunk_policy, 1)) / 2.;
}

double TrunkPlayerExploitability(
    SequenceFormLpSpecification* spec, const Policy& trunk_policy,
    Player p, absl::optional<double> maybe_game_value) {
  double game_value;
  if (maybe_game_value.has_value()) {
    game_value = maybe_game_value.value();
  } else {
    spec->SpecifyLinearProgram(Player{0});
    game_value = spec->Solve();
  }
  // Switch sign appropriately - game value is defined for player 0!
  const double root_value = game_value * (1 - 2*p);
  const double fixed_value =
      ComputeRootValueWhileFixingStrategy(spec, trunk_policy, p);
  return root_value - fixed_value;
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
