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


#include "open_spiel/papers_with_code/1906.06412.value_functions/ismcts_playthroughs.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"
#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void TestSamplingPlaythroughs() {
  std::mt19937 rnd_gen(0);
  auto game =  LoadGame("goofspiel("
                          "players=2,"
                          "num_cards=3,"
                          "imp_info=True,"
                          "points_order=descending"
                        ")");
  IsmctsPlaythroughs playthroughs;
  playthroughs.num_matches = 10;
  playthroughs.max_simulations = 10;
  auto turn_based = ConvertToTurnBased(*game);
  playthroughs.MakeBot(rnd_gen);
  playthroughs.GenerateNodes(*turn_based, rnd_gen);
  SPIEL_CHECK_FLOAT_NEAR((--playthroughs.cdf.end())->first, 1., 1e-8);
  Observation& infostate = playthroughs.SampleInfostate(rnd_gen);
  auto particle_set = GenerateParticles(infostate, 0, 1000, 1000, 0, rnd_gen);
  SPIEL_CHECK_FALSE(particle_set->particles.empty());
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestSamplingPlaythroughs();
}

