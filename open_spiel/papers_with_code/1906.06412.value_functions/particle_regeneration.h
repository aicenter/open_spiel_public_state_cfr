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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_REGENERATION_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_REGENERATION_

#include "ortools/sat/cp_model.h"

#include "open_spiel/games/goofspiel.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"

namespace open_spiel {
namespace papers_with_code {

namespace opr = operations_research;

// Implemented only for imp. info GoofSpiel.
class ParticleGenerator {
  std::shared_ptr<const goofspiel::GoofspielGame> game_;
  const opr::Domain cards_;
  std::mt19937& rnd_gen_;

  std::unique_ptr<opr::sat::CpModelBuilder> cp_model_ = nullptr;
  std::array<std::vector<opr::sat::IntVar>, 2> played_;
  int num_bets_;
  Player current_player_;
 public:
  ParticleGenerator(std::shared_ptr<const goofspiel::GoofspielGame> game,
                    std::mt19937& rnd_gen)
      : game_(game), cards_(0, game->NumCards() - 1), rnd_gen_(rnd_gen) {};

  //  0: a tie
  // -1: a loss (of player 0)
  //  1: a win  (of player 0)
  void SetPublicOutcomes(const std::vector<int>& outcomes);
  void SetInfoState(const Observation& infostate, Player player_hand);
  void SetPublicState(const Observation& public_state);

  std::unique_ptr<ParticleSet> GenerateParticles(int max_particles,
                                                 int max_rejection_cnt);

 private:
  void ResetModel(int num_bets);
};

} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARTICLE_REGENERATION_
