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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

struct Particle {
  // Flat joint actions.
  std::vector<Action> history;
  // Cumulative chance reach prob.
  float chance_reach;
  // Individual player reach probs.
  std::array<float, 2> player_reach;

  // Rollout a state based on particle history.
  std::unique_ptr<State> MakeState(const Game& game) const;
};

// Set of particles that are used to construct a public belief state.
// All of the particles must share the same public state!
struct ParticleSet {
  std::vector<Particle> particles;
  // Partitions the set of particles to their corresponding infostates for each
  // player. Stored numbers are indices within the particles vector.
  std::array<std::vector<std::vector<int>>, 2> partition;

  // Aggregate player reach probs over individual particles to compute
  // player beliefs over the infostate partition.
  std::array<std::vector<float>, 2> ComputeBeliefs() const;
};

// Check internal observation consistency of the particle set.
void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> public_observer,
                                 std::shared_ptr<Observer> hand_observer,
                                 const ParticleSet& set);
// Check consistency with infostate nodes -- the roots of the infostate tree.
void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> infostate_observer,
                                 std::array<std::vector<const algorithms::InfostateNode*>, 2> infostate_nodes,
                                 const ParticleSet& set);

} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_
