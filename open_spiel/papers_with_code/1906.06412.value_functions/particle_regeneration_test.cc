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

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void TestGenerateParticles() {
  std::shared_ptr<const Game> game = LoadGame(
      "goofspiel(imp_info=true,players=2,points_order=descending,num_cards=3)");
  auto observer = game->MakeObserver(kPublicStateObsType, {});
  auto observation = Observation(*game, observer);
  auto rnd_gen = std::make_shared<std::mt19937>();
  ParticleGenerator generator(
      std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game), rnd_gen);

  auto check_particle_count = [&](const State& s, int expected_count) {
    observation.SetFrom(s, 0);
    generator.SetPublicState(observation);
    auto particle_set = generator.GenerateParticles(1000, 1000);
    SPIEL_CHECK_EQ(particle_set->particles.size(), expected_count);
  };

  {  // A tie.
    auto state = game->NewInitialState();
    state->ApplyActions({1, 1});
    check_particle_count(*state, 3);
  }
  {  // A win.
    auto state = game->NewInitialState();
    state->ApplyActions({2, 1});
    check_particle_count(*state, 3);
  }
  {  // A loss.
    auto state = game->NewInitialState();
    state->ApplyActions({1, 2});
    check_particle_count(*state, 3);
  }
}

// Print out bunch of particles to see how diverse they are.
// Not a test yet.
void ShowParticleDiversity() {
  std::shared_ptr <const Game> game = LoadGame(
      "goofspiel(imp_info=true,players=2,points_order=descending,num_cards=13)");
  auto infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  auto infostate_observation = Observation(*game, infostate_observer);
  auto public_observer = game->MakeObserver(kPublicStateObsType, {});
  auto public_observation = Observation(*game, public_observer);
  const int player = 0;
  auto rnd_gen = std::make_shared<std::mt19937>();
  ParticleGenerator generator(
      std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game), rnd_gen);

  auto state = game->NewInitialState();
  state->ApplyActions({1, 2});
  state->ApplyActions({2, 1});
  state->ApplyActions({4, 3});
  state->ApplyActions({7, 10});
  state->ApplyActions({12, 0});
  state->ApplyActions({8, 4});
  infostate_observation.SetFrom(*state, player);
  public_observation.SetFrom(*state, 0);

  const absl::Time start = absl::Now();
  generator.SetInfoState(infostate_observation, player);
  auto particle_set = generator.GenerateParticles(1000, 1000);
  const absl::Time end = absl::Now();
  const double milis = absl::ToDoubleMilliseconds(end - start);

  auto particle_observation = public_observation;
  for (auto& particle: particle_set->particles) {
    std::cout << particle << "\n";
    auto particle_state = particle.MakeState(*game);
    particle_observation.SetFrom(*particle_state, 0);
    SPIEL_CHECK_TRUE(particle_observation == public_observation);
  }
  std::cout << "Generated " << particle_set->particles.size()
            << " particles in " << milis << "ms";
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestGenerateParticles();
  open_spiel::papers_with_code::ShowParticleDiversity();
}

