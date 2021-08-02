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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EXPERIENCE_REPLAY_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EXPERIENCE_REPLAY_

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/reusable_structs.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"

namespace open_spiel {
namespace papers_with_code {

// Experience replay - a circular buffer.
class ExperienceReplay : public BatchData {
  size_t head_ = 0;
  // Track how many times the whole buffer has been rewritten.
  size_t overflow_cnt_ = 0;
  // Track how many times each experience has been sampled.
  std::vector<int> visit_cnt_;
 public:
  ExperienceReplay(int buffer_size, int input_size, int output_size)
    : BatchData(buffer_size, input_size, output_size),
      visit_cnt_(buffer_size, 0) {}

  // Return a data point that can be written to.
  ParticleDataPoint AddExperience(const ParticleDims& dims);
  PositionalDataPoint AddExperience(const PositionalDims& dims);

  // Fill batch with randomly sampled data points.
  void SampleBatch(BatchData* batch, std::mt19937& rnd_gen);
  bool IsFilled() const { return overflow_cnt_ > 0; }
  bool IsAtBeginning() const { return head_ == 0; }
  void ResetHead() { head_ = 0; }
  size_t head() const { return head_; }
  const std::vector<int>& visit_cnt() const { return visit_cnt_; }

 protected:
  void AdvanceHead();
};

enum ReplayFillerPolicy {
  kNothing,
  kTrunkDlcfr,
  kTrunkRandom,
  kPbsRandom,
  kSparsePbsRandom,
  kIigsKnPbsRandom,
  kBootstrap,
  kIsmctsBootstrap
};
ReplayFillerPolicy GetReplayFillerPolicy(const std::string& s);  // From string.

// Helper struct so we don't need to pass so many parameters
// for bandit randomization.
struct StrategyRandomizer {
  std::mt19937* rnd_gen;
  double prob_pure_strat = 0.1;
  double prob_fully_mixed = 0.05;
  double prob_benford_dist = 0.0;

  void Randomize(std::vector<algorithms::BanditVector>& bandits);
  void Randomize(const Game& game, ParticleSet* set,
                 std::shared_ptr<Observer> infostate_observer);
};

// Helper struct so we don't need to pass so many parameters
// for adding replay experiences.
struct ReplayFiller {
  // All of these must be supplied.
  ExperienceReplay* replay;
  SubgameFactory* subgame_factory;
  SolverFactory* solver_factory;
  BasicDims* dims;
  StrategyRandomizer* randomizer;
  ReusableStructures* reuse;

  // Optional bootstrapping.
  std::unique_ptr<ExperienceReplay> bootstrap = nullptr;
  // Bootstrapping starts at the maximum move number - 1 (skips terminals), and
  // uses the neural network to incrementally generate target values for
  // learning, as we move from deep public states to the shallow ones, until
  // root (bootstrap_move_number=0) is reached. The initial value assigned will
  // be assigned automatically based on the game.
   int bootstrap_move_number;

  // Params.
  NetArchitecture arch = NetArchitecture::kParticle;
  double sparse_epsilon = 0.;
  std::vector<int> eval_iters;
  int max_rejection_cnt = 1000;
  int infostate_particles = 1;
  bool normalize_beliefs = false;

  void CreateExperiences(ReplayFillerPolicy fill_policy, int num_experiences);

 protected:
  void AddTrunkRandomPbsSolution();
  void AddRandomPbsSolution();
  void AddRandomSparsePbsSolution();
  void AddIigsKnRandomPbsSolution();
  void AddBootstrappedSolution();
  void AddIsmctsBootstrapedSolution();
  void FillReplayWithTrunkDlCfrPbsSolutions();

  std::unique_ptr<ParticleSet> PickParticleSet(int at_depth = -1);
  std::unique_ptr<ParticleSet> PickIsmctsParticleSet(int at_depth = -1);

  void AddExperience(const PublicState& state, const NetContext* net_context);
  void AddExperiencesFromPublicStates(const std::vector<PublicState>& states);

  void AddParticleExperience(const PublicState& leaf, ExperienceReplay* buffer);
  void AddPositionalExperience(const PublicState& leaf,
                               const NetContext& net_context,
                               ExperienceReplay* buffer);
};

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EXPERIENCE_REPLAY_
