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

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"

namespace open_spiel {
namespace papers_with_code {

std::unique_ptr<State> Particle::MakeState(const Game& game) const {
  std::unique_ptr<State> s = game.NewInitialState();
  for (Action a : history) s->ApplyAction(a);
  return  s;
}

void CheckObservation(const Observation& actual, const Observation& expected) {
  SPIEL_CHECK_EQ(actual.Tensor(), expected.Tensor());
  SPIEL_CHECK_EQ(actual.tensor_info(), expected.tensor_info());
}

void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> public_observer,
                                 std::shared_ptr<Observer> hand_observer,
                                 const ParticleSet& set) {
  SPIEL_CHECK_FALSE(set.particles.empty());
  SPIEL_CHECK_FALSE(set.partition.empty());

  std::vector<std::unique_ptr<State>> histories;
  for (const Particle& particle : set.particles) {
    histories.push_back(particle.MakeState(game));
  }

  // Check public observations
  Observation expected_public(game, public_observer);
  Observation actual_public = expected_public;
  expected_public.SetFrom(*histories[0], kDefaultPlayerId);
  for (const std::unique_ptr<State>& history : histories) {
    actual_public.SetFrom(*history, kDefaultPlayerId);
    CheckObservation(actual_public, expected_public);
  }

  // Check partition
  Observation expected_hand(game, hand_observer);
  Observation actual_hand = expected_hand;
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_FALSE(set.partition[pl].empty());
    for (const std::vector<int>& partition_element : set.partition[pl]) {
      SPIEL_CHECK_FALSE(partition_element.empty());
      expected_hand.SetFrom(*histories[partition_element[0]], pl);
      for (int particle_idx : partition_element) {
        actual_hand.SetFrom(*histories[particle_idx], pl);
        CheckObservation(actual_hand, expected_hand);
      }
    }
  }
}

}  // papers_with_code
}  // open_spiel
