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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"

namespace open_spiel {
namespace papers_with_code {

std::shared_ptr<Subgame> SubgameFactory::MakeTrunk(int trunk_depth) const {
  std::vector<std::unique_ptr<State>> start_states {};
  start_states.push_back(game->NewInitialState());
  std::vector<double> chance_reach_probs {1.};
  int depth = trunk_depth > 0 ? trunk_depth
                              : max_trunk_depth;
  auto trees = algorithms::MakeInfostateTrees(
      start_states, chance_reach_probs, infostate_observer,
      depth, kDlCfrInfostateTreeStorage);
  return std::make_shared<Subgame>(game, public_observer, trees);
}

std::shared_ptr<Subgame> SubgameFactory::MakeSubgame(
    const ParticleSet& set, int custom_move_ahead_limit) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  int depth = custom_move_ahead_limit > 0 ? custom_move_ahead_limit
                                          : max_move_ahead_limit;
  auto trees = MakeSubgameInfostateTrees(set, depth);
  auto out = std::make_unique<Subgame>(game, public_observer, trees);
  set.AssignBeliefs(out->initial_state());  // Compute initial beliefs..
  return out;
}

std::shared_ptr<Subgame> SubgameFactory::MakeSubgame(
    const PublicState& state, int custom_move_ahead_limit) const {
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(
        state.nodes[pl],
        custom_move_ahead_limit > 0 ? custom_move_ahead_limit
                                     : max_move_ahead_limit,
        kDlCfrInfostateTreeStorage
    ));
  }
  auto out = std::make_unique<Subgame>(game, public_observer, trees);
  out->initial_state().SetBeliefs(state.beliefs);
  return out;
}

std::vector<std::shared_ptr<algorithms::InfostateTree>>
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
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(algorithms::MakeInfostateTree(
        root_histories, chance_reach_probs,
        infostate_observer, pl, depth,
        kDlCfrInfostateTreeStorage
    ));
  }
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, infostate_observer,
                                           {trees[0]->root().children(),
                                            trees[1]->root().children()}, set));
  return trees;
}


}  // papers_with_code
}  // open_spiel
