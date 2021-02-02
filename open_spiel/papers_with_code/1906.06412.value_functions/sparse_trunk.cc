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
std::vector<const State*> CollectStates(const LeafPublicState& public_state) {
  std::vector<const State*> states;
  // Just some estimate.
  states.reserve(8 * public_state.leaf_nodes[0].size());
  for (const InfostateNode* node : public_state.leaf_nodes[0]) {
    for (int i = 0; i < node->corresponding_states_size(); ++i) {
      states.push_back(node->corresponding_states()[i].get());
    }
  }
  return states;
}

std::vector<double> CollectChances(const LeafPublicState& public_state) {
  std::vector<double> chances;
  // Just some estimate.
  chances.reserve(8 * public_state.leaf_nodes[0].size());
  for (const InfostateNode* node : public_state.leaf_nodes[0]) {
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
    std::shared_ptr<const LeafEvaluator> net_evaluator,
    std::shared_ptr<const LeafEvaluator> terminal_evaluator,
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
  DepthLimitedCFR temp_trunk(game, roots_depth,
                             net_evaluator, terminal_evaluator);

  // 2. For each decision infostate, make a random subset of histories
  //    within the public state + 1 special history that surely belongs
  //    to that infostate.
  std::vector<std::unique_ptr<SparseTrunk>> sparse_trunks;
  sparse_trunks.reserve(temp_trunk.public_leaves().size() * 8);

  for (const LeafPublicState& public_state: temp_trunk.public_leaves()) {
    std::vector<const State*> states = CollectStates(public_state);
    std::vector<double> chances = CollectChances(public_state);
    SPIEL_CHECK_EQ(states.size(), chances.size());

    std::vector<int> public_state_perm(states.size());
    std::iota(public_state_perm.begin(), public_state_perm.end(), 0);

    for (int pl = 0; pl < 2; ++pl) {
      for (const InfostateNode* node : public_state.leaf_nodes[pl]) {
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
        sparse_trunk->eval_infostate = node->infostate_string();
        sparse_trunks.push_back(std::move(sparse_trunk));
      }
    }
  }

  return sparse_trunks;
}

}  // papers_with_code
}  // open_spiel



