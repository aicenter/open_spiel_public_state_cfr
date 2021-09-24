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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

struct Particle {
  // Flat joint actions.
  std::vector<Action> history;
  // Cumulative chance reach prob.
  float chance_reach = 0;
  // Individual player reach probs.
  std::array<float, 2> player_reach = {0., 0.};
  Particle(std::vector<Action> history) : history(std::move(history)) {}

  // Full reach prob.
  float reach() const {
    return chance_reach * player_reach[0] * player_reach[1];
  }
  // Rollout a state based on particle history.
  std::unique_ptr<State> MakeState(const Game& game) const;
};

// Set of particles that are used to construct a public belief state.
// All of the particles must share the same public state!
struct ParticleSet {
  std::vector<Particle> particles;

  int index_of(const std::vector<Action>& history) const;
  Particle& at(const std::vector<Action>& history);
  const Particle& at(const std::vector<Action>& history) const;
  Particle& add(const std::vector<Action>& history);
  bool has(const std::vector<Action>& history) const;
  int size() const { return particles.size(); }

  void ImportSet(const ParticleSet& other) {
    for (const Particle& candidate : other.particles) {
      if (!has(candidate.history)) particles.push_back(candidate);
    }
  }
  void AssignBeliefsTo(PublicState* state) const;
  void ComputeBeliefs(const Game& game, const TabularPolicy& policy,
                      const Observer& infostate_observer);
};


std::unique_ptr<ParticleSet> PickParticlesBasedOnReach(const PublicState& state,
                                                       int max_particles);
std::unique_ptr<ParticleSet> PickParticlesBasedOnQvalues(const PublicState& state,
                                                         int max_particles);

// Check internal observation consistency of the particle set.
void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> public_observer,
                                 std::shared_ptr<Observer> hand_observer,
                                 const ParticleSet& set);
// Check consistency with infostate nodes -- the roots of the infostate tree.
void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> infostate_observer,
                                 std::vector<std::vector<algorithms::InfostateNode*>> infostate_nodes,
                                 const ParticleSet& set);

std::vector<std::unique_ptr<State>> GetStatesInSupport(const Game& game,
                                                       const Policy& policy);

} // namespace papers_with_code

std::ostream& operator<<(std::ostream& os,
                         const papers_with_code::Particle& particle);

} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_
