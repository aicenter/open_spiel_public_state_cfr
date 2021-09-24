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

#include <utility>

namespace open_spiel {
namespace papers_with_code {

std::shared_ptr<Subgame> SubgameFactory::MakeTrunk(int trunk_depth) const {
  std::vector<std::unique_ptr<State>> start_states{};
  start_states.push_back(game->NewInitialState());
  std::vector<double> chance_reach_probs{1.};
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
  set.AssignBeliefsTo(&out->initial_state());  // Compute initial beliefs..
  return out;
}

std::shared_ptr<Subgame> SubgameFactory::MakeSubgameSafeResolving(
    const ParticleSet& set, int player,
    std::unordered_map<std::string, double> opponent_CFVs,
    int custom_move_ahead_limit) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  int depth = custom_move_ahead_limit > 0 ? custom_move_ahead_limit
                                          : max_move_ahead_limit;
  auto trees = MakeSubgameResolvingInfostateTrees(
      set, depth, player, std::move(opponent_CFVs),
      use_max_cfv_in_missing_infostates);
  auto out = std::make_unique<Subgame>(game, public_observer, trees);
  set.AssignBeliefsTo(&out->initial_state());  // Compute initial beliefs.
  return out;
}

std::shared_ptr<Subgame> SubgameFactory::MakeSubgame(
    const PublicState& state, int custom_move_ahead_limit) const {
  std::unique_ptr<Subgame> out =
      open_spiel::papers_with_code::MakeSubgame(
          state, game, public_observer,
          custom_move_ahead_limit > 0 ? custom_move_ahead_limit
                                      : max_move_ahead_limit);
  out->initial_state().SetBeliefs(state.beliefs);
  return out;
}

std::vector<std::shared_ptr<algorithms::InfostateTree>>
SubgameFactory::MakeSubgameInfostateTrees(const ParticleSet& set,
                                          int depth) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, public_observer,
                                           infostate_observer, set));
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

std::vector<std::shared_ptr<algorithms::InfostateTree>>
SubgameFactory::MakeSubgameResolvingInfostateTrees(
    const ParticleSet& set, int depth, int player,
    std::unordered_map<std::string, double> opponent_CFVs,
    bool use_max_cfv_in_missing_infostates) const {
  SPIEL_CHECK_LE(set.particles.size(), max_particles);
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, public_observer,
                                           hand_observer, set));
  SPIEL_CHECK_FALSE(opponent_CFVs.empty());

  int opponent = 1 - player;

  std::vector<std::unique_ptr<State>> root_histories;
  std::vector<double> chance_reach_probs;
  std::unordered_map<std::string, double> info_state_reaches;

  for (const Particle& particle : set.particles) {
    double resolving_reach = particle.chance_reach
                           * particle.player_reach[player];
    if (resolving_reach == 0) continue;

    auto state = particle.MakeState(*game);
    std::string info_state = state->InformationStateString(opponent);
    if (info_state_reaches.find(info_state) == info_state_reaches.end()) {
      info_state_reaches.emplace(info_state, 0.);
    }

    root_histories.push_back(std::move(state));
    chance_reach_probs.push_back(resolving_reach);
    info_state_reaches[info_state] += resolving_reach;
  }

  double max_cfv = std::numeric_limits<double>::lowest();
  for(const auto&[is, cfv] : opponent_CFVs) {
    max_cfv = std::max(max_cfv, cfv);
  }
  SPIEL_CHECK_GT(max_cfv, std::numeric_limits<double>::lowest());  // We found some.

  // Normalize the CFVs based on sum of reach probs.
  for (const auto&[infostate, resolving_reach] : info_state_reaches) {
    if (resolving_reach > 0) {
      auto it = opponent_CFVs.find(infostate);
      if (it == opponent_CFVs.end()) {
        if (use_max_cfv_in_missing_infostates) {
          // If the infostate does not exist, offer the opponent
          // the highest existing cf value.
          opponent_CFVs[infostate] = max_cfv / resolving_reach;
        } else {
          SpielFatalError(absl::StrCat(
              "The infostate '", infostate,
              "' does not have an opponent counterfactual value defined."));
        }
      } else {
        it->second /= resolving_reach;
      }
    }
  }
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees =
      algorithms::MakeResolvingInfostateTrees(
          root_histories, chance_reach_probs, infostate_observer,
          1 - player, opponent_CFVs, depth, kDlCfrInfostateTreeStorage);
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, infostate_observer,
                                           {trees[0]->root().children(),
                                            trees[1]->root().children()}, set));
  return trees;
}

}  // papers_with_code
}  // open_spiel
