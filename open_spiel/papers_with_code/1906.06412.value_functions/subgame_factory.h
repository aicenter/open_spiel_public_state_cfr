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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_FACTORY_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_FACTORY_


#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

namespace open_spiel {
namespace papers_with_code {

constexpr int kDefaultMaxTrunkDepth = 3;
constexpr int kDefaultMaxMoveAheadLimit = 2;
constexpr int kDefaultMaxParticles = 1000;

// A (depth-limited) subgame rooted at some perfect-information histories,
// that have belief distribution over the infostates induced by those histories.
using SequenceFormLpSpecification = algorithms::ortools::SequenceFormLpSpecification;

// Produce a subgame given some input.
struct SubgameFactory {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;  // For infostate strings.
  std::shared_ptr<Observer> public_observer;     // For public tensor.
  std::shared_ptr<Observer> hand_observer;       // For hand tensor.
  std::shared_ptr<ParticleGenerator> particle_generator;

  int max_trunk_depth = kDefaultMaxTrunkDepth;
  int max_move_ahead_limit = kDefaultMaxMoveAheadLimit;
  int max_particles = kDefaultMaxParticles;  // Must be always >= 1.
  double particle_epsilon = 0.1;

  // Subgame from game's initial state.
  std::shared_ptr<Subgame> MakeTrunk(int trunk_depth = -1) const;
  std::shared_ptr<Subgame> MakeSubgame(const ParticleSet& set,
                                       int custom_move_ahead_limit = -1) const;
  std::shared_ptr<Subgame> MakeSubgameSafeResolving(
      const ParticleSet& set, int player,
      std::unordered_map<std::string, double> CFVs,
      int custom_move_ahead_limit = -1) const;
  std::shared_ptr<Subgame> MakeSubgame(const PublicState& state,
                                       int custom_move_ahead_limit = -1) const;

 protected:
  std::vector<std::shared_ptr<algorithms::InfostateTree>>
  MakeSubgameInfostateTrees(const ParticleSet& set, int depth) const;
  std::vector<std::shared_ptr<algorithms::InfostateTree>>
  MakeSubgameResolvingInfostateTrees(
      const ParticleSet& set, int depth, int player,
      std::unordered_map<std::string, double> CFVs) const;

};

} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_FACTORY_

