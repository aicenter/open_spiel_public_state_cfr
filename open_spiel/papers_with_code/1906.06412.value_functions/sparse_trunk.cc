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


#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;
using namespace algorithms::dlcfr;

// Note these are shared between players, that's why we loop only
// over one player.
std::vector<const State*> CollectStates(const PublicState& public_state) {
  std::vector<const State*> states;
  // Just some estimate.
  states.reserve(8 * public_state.infostate_nodes[0].size());
  for (const InfostateNode* node : public_state.infostate_nodes[0]) {
    for (int i = 0; i < node->corresponding_states_size(); ++i) {
      states.push_back(node->corresponding_states()[i].get());
    }
  }
  return states;
}

std::vector<double> CollectChances(const PublicState& public_state) {
  std::vector<double> chances;
  // Just some estimate.
  chances.reserve(8 * public_state.infostate_nodes[0].size());
  for (const InfostateNode* node : public_state.infostate_nodes[0]) {
    for (int i = 0; i < node->corresponding_states_size(); ++i) {
      chances.push_back(node->corresponding_chance_reach_probs()[i]);
    }
  }
  return chances;
}

std::vector<int> PickIndices(std::vector<int>& public_state_perm,
                             const std::vector<const State*>& states,
                             const InfostateNode* node,
                             int limit_initial_states,
                             std::mt19937& rnd_gen) {

  if (limit_initial_states < 0) {  // Pick all indices.
    return public_state_perm;
  }

  // Pick one starting state from the infostate node.
  int infostate_pick = std::uniform_int_distribution<>(
                0, node->corresponding_states_size() - 1)(rnd_gen);
  // Find the index of the state.
  const State* picked_state = node->corresponding_states()[infostate_pick].get();
  int picked_index = -1;
  for (int i = 0; i < states.size(); ++i) {
    if (states[i]->FullHistory() == picked_state->FullHistory()) {
      picked_index = i;
    }
  }
  if (picked_index == -1) SpielFatalError("Could not find the picked state!");

  // Shuffle all of the state indices.
  std::shuffle(public_state_perm.begin(),
               public_state_perm.end(), rnd_gen);

  // Make sure there is always one state from the infostate.
  std::vector<int> picks;
  picks.push_back(picked_index);

  for (int pick_rnd_idx : public_state_perm) {
    if (pick_rnd_idx == picked_index) continue;  // Do not repeat selection.
    if (picks.size() >= limit_initial_states) break;
    picks.push_back(pick_rnd_idx);
  }

  SPIEL_DCHECK_EQ(picks.size(),
                  std::min((int) states.size(), limit_initial_states));

  return picks;
}

std::vector<std::unique_ptr<SparseTrunk>> MakeSparseTrunks(
    std::shared_ptr<const Game> game,
    std::shared_ptr<Observer> infostate_observer,
    std::shared_ptr<Observer> public_observer,
    int roots_depth, int trunk_depth,
    std::shared_ptr<const PublicStateEvaluator> net_evaluator,
    std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
    int limit_initial_states,
    const std::string& bandits_for_cfr, std::mt19937& rnd_gen) {
  // Must have a defined start.
  SPIEL_CHECK_GT(roots_depth, 0);
  // Without this the trunks would not contain any states.
  SPIEL_CHECK_LT(roots_depth, trunk_depth);
  // Negative values means "all states", and positive values must be >= 1.
  SPIEL_CHECK_NE(limit_initial_states, 0);

  // 1. First of all, build a temporary full trunk, so we can identify all
  //    public states, infostates and associated histories.
  std::shared_ptr<PublicStateEvaluator> dummy_eval = MakeDummyEvaluator();
  DepthLimitedCFR temp_trunk(game, roots_depth, dummy_eval, terminal_evaluator);
//  std::cout << "# Temp trunk public states stats: \n";
//  PrintPublicStatesStats(temp_trunk.public_states());

  // 2. For each decision infostate, make a random subset of histories
  //    within the public state + 1 special history that surely belongs
  //    to that infostate.
  int num_sparse_trunks = 0;
  for (const PublicState& public_state: temp_trunk.public_states()) {
    for (int pl = 0; pl < 2; ++pl) {
      for (const InfostateNode* node : public_state.infostate_nodes[pl]) {
        // We are interested only in sparsification for decision infostates.
        if (node->type() != kDecisionInfostateNode) continue;
        num_sparse_trunks++;
      }
    }
  }
//  std::cout << "# Making " << num_sparse_trunks << " sparse trunks\n";
  std::vector<std::unique_ptr<SparseTrunk>> sparse_trunks;
  sparse_trunks.reserve(num_sparse_trunks);

  for (const PublicState& public_state: temp_trunk.public_states()) {
    std::vector<const State*> states = CollectStates(public_state);
    std::vector<double> chances = CollectChances(public_state);
    SPIEL_CHECK_EQ(states.size(), chances.size());

    std::vector<int> public_state_perm(states.size());
    std::iota(public_state_perm.begin(), public_state_perm.end(), 0);

    for (int pl = 0; pl < 2; ++pl) {
      for (const InfostateNode* node : public_state.infostate_nodes[pl]) {
        // We are interested only in sparsification for decision infostates.
        if (node->type() != kDecisionInfostateNode) continue;

        std::vector<int> pick_indices = PickIndices(public_state_perm, states,
                                                    node, limit_initial_states,
                                                    rnd_gen);
        std::vector<const State*> start_states;
        start_states.reserve(pick_indices.size());
        std::vector<double> start_chances;
        start_chances.reserve(pick_indices.size());
        for (int pick_idx : pick_indices) {
          start_states.push_back(states[pick_idx]);
          start_chances.push_back(chances[pick_idx]);
        }

        const int move_lim = trunk_depth - roots_depth;
        std::vector<std::shared_ptr<InfostateTree>> sparse_trees = {
            MakeInfostateTree(start_states, start_chances, infostate_observer,
                              /*pl=*/0, /*max_move_ahead_limit=*/move_lim),
            MakeInfostateTree(start_states, start_chances, infostate_observer,
                              /*pl=*/1, /*max_move_ahead_limit=*/move_lim)
        };
        auto sparse_trunk = std::make_unique<SparseTrunk>();
        sparse_trunk->dlcfr = std::make_unique<DepthLimitedCFR>(
            game, sparse_trees, net_evaluator, terminal_evaluator,
            public_observer, MakeBanditVectors(sparse_trees, bandits_for_cfr));
        sparse_trunk->fixate_infostates = {node->infostate_string()};
        sparse_trunks.push_back(std::move(sparse_trunk));
      }
    }
  }

  return sparse_trunks;
}

constexpr double kSupportThreshold = 1e-5;
//constexpr double kSupportThreshold = -1;

// Return set of actions in support for each player.
std::vector<ActionsAndProbs> GetActionsInSupport(
    const State& s, const Observer& infostate_observer,
    const TabularPolicy& eq_policies,
    double support_threshold) {
  SPIEL_CHECK_TRUE(s.IsPlayerNode() || s.IsSimultaneousNode());
  std::vector<ActionsAndProbs> actions_in_support;
  actions_in_support.reserve(s.NumPlayers());

  for (int pl = 0; pl < s.NumPlayers(); ++pl) {
    if (s.IsPlayerActing(pl)) {
      const std::string infostate = infostate_observer.StringFrom(s, pl);
      ActionsAndProbs local_policy = eq_policies.GetStatePolicy(infostate);
      SPIEL_CHECK_FALSE(local_policy.empty());
      ActionsAndProbs play_actions;
      play_actions.reserve(local_policy.size());
      for (const auto& [action, prob] : local_policy) {
        if (prob > support_threshold) {
          play_actions.push_back({action, prob});
        }
      }
      actions_in_support.push_back(play_actions);
    } else {
      actions_in_support.push_back({{kInvalidAction, 1.}});
    }
  }

  return actions_in_support;
}

void GenerateHistoriesInSupportAtDepth(
    std::unique_ptr<State> s, double chn,
    const Observer& infostate_observer,
    const TabularPolicy& eq_policies,
    int depth,
    std::vector<std::unique_ptr<State>>& states_out,
    std::vector<double>& chances_out,
    double support_threshold) {

  if (s->IsTerminal()) {
    return;
  }

  if (s->MoveNumber() == depth) {
    states_out.push_back(std::move(s));
    chances_out.push_back(chn);
    return;
  }

  if (s->IsChanceNode()) {
    for (const auto& [action, prob] : s->ChanceOutcomes()) {
      std::unique_ptr<State> child = s->Clone();
      child->ApplyAction(action);
      GenerateHistoriesInSupportAtDepth(
          std::move(child), chn * prob, infostate_observer, eq_policies,
          depth, states_out, chances_out, support_threshold);
    }
    return;
  }

  if (s->IsSimultaneousNode()) {
    std::vector<ActionsAndProbs> actions_in_support =
        GetActionsInSupport(*s, infostate_observer, eq_policies,
                            support_threshold);

    for (auto& [action0, prob0] : actions_in_support[0]) {
      for (auto& [action1, prob1] : actions_in_support[1]) {
        std::unique_ptr<State> child = s->Clone();
        child->ApplyActions({action0, action1});
        GenerateHistoriesInSupportAtDepth(
            std::move(child), chn * prob0 * prob1, infostate_observer, eq_policies,
            depth, states_out, chances_out, support_threshold);
      }
    }
    return;
  }

  SPIEL_CHECK_TRUE(s->IsPlayerNode());
  std::vector<ActionsAndProbs> actions_in_support =
      GetActionsInSupport(*s, infostate_observer, eq_policies,
                          support_threshold);
  for (const auto& [action, prob] :  actions_in_support[s->CurrentPlayer()]) {
    if (action == kInvalidAction) continue;

    std::unique_ptr<State> child = s->Clone();
    child->ApplyAction(action);
    GenerateHistoriesInSupportAtDepth(
        std::move(child), chn * prob, infostate_observer, eq_policies,
        depth, states_out, chances_out, support_threshold);
  }
}


std::unique_ptr<SparseTrunk> MakeSparseTrunkWithEqSupport(
    const TabularPolicy& eq_policies,
    std::shared_ptr<const Game> game,
    std::shared_ptr<Observer> infostate_observer,
    std::shared_ptr<Observer> public_observer,
    int roots_depth, int trunk_depth,
    std::shared_ptr<const algorithms::dlcfr::PublicStateEvaluator> leaf_evaluator,
    std::shared_ptr<const algorithms::dlcfr::PublicStateEvaluator> terminal_evaluator,
    const std::string& bandits_for_cfr,
    double support_threshold,
    bool prune_chance_histories) {

  std::vector<std::unique_ptr<State>> all_start_states;
  std::vector<double> all_start_chances;
  GenerateHistoriesInSupportAtDepth(game->NewInitialState(), /*chn=*/1.,
                                    *infostate_observer, eq_policies, roots_depth,
                                    all_start_states, all_start_chances, -1);
  SPIEL_CHECK_EQ(all_start_states.size(), all_start_chances.size());

  // Prune states based on reach prob.
  std::vector<const State*> start_states_ptrs;
  std::vector<double> start_chances;
  std::vector<std::string> fixate_infostates;
  std::vector<std::string> uniform_infostates;
  int num_hist_nonzero_reach = 0;
  for (int i = 0; i < all_start_chances.size(); ++i) {
    // Print the chance dist
    const bool in_support = all_start_chances[i] > support_threshold;
    if (all_start_chances[i] > 0) num_hist_nonzero_reach++;

    std::unique_ptr<State>& s = all_start_states[i];
    if (s->IsPlayerNode()) {
//      std::cout << "# " << all_start_chances[i] << std::endl;
    }

    if (in_support && (!prune_chance_histories || s->IsPlayerNode())) {
      start_states_ptrs.push_back(s.get());
      start_chances.push_back(all_start_chances[i]);
    }

    for (int pl = 0; pl < 2; ++pl) {
      if (s->IsPlayerActing(pl)) {
        std::string info_state = infostate_observer->StringFrom(*s, pl);
        const auto it_fixate = std::find(fixate_infostates.begin(),
                                         fixate_infostates.end(),
                                         info_state);
        const auto it_uniform = std::find(uniform_infostates.begin(),
                                          uniform_infostates.end(),
                                          info_state);

        if (in_support) {
          if (it_fixate == fixate_infostates.end()) {
            fixate_infostates.push_back(info_state);
          }
          // Remove from uniform, as we have now added the infostate
          // to the fixated infostates.
          if (it_uniform != uniform_infostates.end()) {
            uniform_infostates.erase(it_uniform);
          }
        } else {
          if (it_fixate == fixate_infostates.end()
              && it_uniform == uniform_infostates.end()) {
            uniform_infostates.push_back(info_state);
          }
        }
      }
    }
  }

//  std::cout << "# All histories: " << all_start_states.size() << "\n";
//  std::cout << "# Histories with non-zero reach: " << num_hist_nonzero_reach << "\n";
//  std::cout << "# Selected starting histories: " << start_states_ptrs.size() << "\n";
//  std::cout << "# Fixate infostates: " << fixate_infostates.size() << "\n";
//  std::cout << "# Uniform infostates: " << uniform_infostates.size() << "\n";

  const int move_lim = trunk_depth - roots_depth;
  auto sparse_trunk = std::make_unique<SparseTrunk>();
  if (!fixate_infostates.empty()) {
    std::vector<std::shared_ptr<InfostateTree>> sparse_trees = {
        MakeInfostateTree(start_states_ptrs, start_chances, infostate_observer,
            /*pl=*/0, /*max_move_ahead_limit=*/move_lim),
        MakeInfostateTree(start_states_ptrs, start_chances, infostate_observer,
            /*pl=*/1, /*max_move_ahead_limit=*/move_lim)
    };
    sparse_trunk->dlcfr = std::make_unique<DepthLimitedCFR>(
        game, sparse_trees, leaf_evaluator, terminal_evaluator,
        public_observer, MakeBanditVectors(sparse_trees, bandits_for_cfr));
  }
  sparse_trunk->fixate_infostates = fixate_infostates;
  sparse_trunk->uniform_infostates = uniform_infostates;

  return sparse_trunk;
}

}  // papers_with_code
}  // open_spiel



