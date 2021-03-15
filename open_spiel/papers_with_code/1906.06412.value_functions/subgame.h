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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"


namespace open_spiel {
namespace papers_with_code {

struct Subgame {
  std::unique_ptr<algorithms::dlcfr::DepthLimitedCFR> solver;

  // A pointer to experience replay where the solution should be written.
  // All of the inputs are prepared by the SubgameFactory; only outputs should
  // be written by using the CopySolution() method.
  ParticleDataPoint solution;

  Subgame(std::unique_ptr<algorithms::dlcfr::DepthLimitedCFR> solver,
          const ParticleDataPoint& solution)
      : solver(std::move(solver)), solution(solution) {}

  void CopySolution();
};

// Produce a subgame given a particle set.
struct SubgameFactory {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;  // For infostate strings.
  std::shared_ptr<Observer> public_observer;     // For public tensor.
  std::shared_ptr<Observer> hand_observer;       // For hand tensor.

  std::string use_bandits_for_cfr;
  int max_move_ahead_limit;
  int max_particles;
  ParticleDims dims;
  TreeMap<Action, std::unique_ptr<State>> history_cache;

  std::shared_ptr<const algorithms::dlcfr::LeafEvaluator> terminal_evaluator;
  std::shared_ptr<algorithms::dlcfr::LeafEvaluator> leaf_evaluator;

  std::unique_ptr<Subgame> MakeSubgame(const ParticleSet& set,
                                       ParticleDataPoint write_solution_to);
  std::unique_ptr<ParticleSet> SampleParticleSet(const Subgame& subgame,
                                                 std::mt19937& rnd_gen);
};


} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_

