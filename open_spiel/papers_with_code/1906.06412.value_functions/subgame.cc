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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

std::unique_ptr<Subgame> SubgameFactory::MakeTrunk(
    std::shared_ptr<const LeafEvaluator> custom_leaf_evaluator,
    std::string custom_bandits_for_cfr) const {
  std::vector<std::unique_ptr<State>> start_states {};
  start_states.push_back(game->NewInitialState());
  std::vector<double> chance_reach_probs {1.};
  std::vector<std::shared_ptr<InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(
        start_states, chance_reach_probs,
        infostate_observer, pl, max_move_ahead_limit,
        algorithms::dlcfr::kDlCfrInfostateTreeStorage
    ));
  }
  auto out = std::make_unique<Subgame>(
      game, trees, custom_leaf_evaluator, terminal_evaluator,
      public_observer, MakeBanditVectors(trees, custom_bandits_for_cfr));
  return out;
}

std::unique_ptr<Subgame> SubgameFactory::MakeSubgame(
    const ParticleSet& set,
    int custom_move_ahead_limit,
    std::shared_ptr<const PublicStateEvaluator> custom_leaf_evaluator
) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  int depth = custom_move_ahead_limit ? custom_move_ahead_limit
                                      : max_move_ahead_limit;
  auto trees = MakeSubgameInfostateTrees(set, depth);
  auto evaluator = custom_leaf_evaluator ? custom_leaf_evaluator
                                         : leaf_evaluator;
  auto out = std::make_unique<Subgame>(
      game, trees, evaluator, terminal_evaluator,
      public_observer, MakeBanditVectors(trees, use_bandits_for_cfr));
  set.AssignBeliefs(out->initial_state());
  out->SetBeliefs(out->initial_state().beliefs);
  return out;
}

std::unique_ptr<Subgame> SubgameFactory::MakeSubgame(
    const PublicState& state,
    int custom_move_ahead_limit,
    std::shared_ptr<const PublicStateEvaluator> custom_leaf_evaluator
) const {
  std::vector<std::shared_ptr<InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(
        state.nodes[pl],
        custom_move_ahead_limit ? custom_move_ahead_limit
                                : max_move_ahead_limit,
        algorithms::dlcfr::kDlCfrInfostateTreeStorage
    ));
  }
  auto evaluator = custom_leaf_evaluator ? custom_leaf_evaluator
                                         : leaf_evaluator;
  auto out = std::make_unique<Subgame>(
      game, trees, evaluator, terminal_evaluator,
      public_observer, MakeBanditVectors(trees, use_bandits_for_cfr));
  out->SetBeliefs(state.beliefs);
  return out;
}

std::vector<std::shared_ptr<InfostateTree>>
SubgameFactory::MakeSubgameInfostateTrees(const ParticleSet& set, int depth) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, public_observer,
                                           hand_observer, set));
  std::vector<std::unique_ptr<State>> root_histories;
  std::vector<double> chance_reach_probs;
  for (const Particle& particle : set.particles) {
    root_histories.push_back(particle.MakeState(*game));
    chance_reach_probs.push_back(particle.chance_reach);
  }
  std::vector<std::shared_ptr<InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(
        root_histories, chance_reach_probs,
        infostate_observer, pl, depth,
        algorithms::dlcfr::kDlCfrInfostateTreeStorage
    ));
  }
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, infostate_observer,
                                           {trees[0]->root().children(),
                                            trees[1]->root().children()}, set));
  return trees;
}


}  // papers_with_code
}  // open_spiel
