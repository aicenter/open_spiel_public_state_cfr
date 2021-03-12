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

#include "open_spiel/algorithms/infostate_dl_cfr.h"


#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"

namespace open_spiel {
namespace papers_with_code {

// Experience replay - a circular buffer.
class ExperienceReplay : public BatchData {
  size_t head_ = 0;
  // Track how many times the whole buffer has been rewritten.
  size_t overflow_cnt_ = 0;
 public:
  ExperienceReplay(int buffer_size, int input_size, int output_size)
    : BatchData(buffer_size, input_size, output_size) {}
  // Return a data point that can be written to.
  ParticleDataPoint AddExperience(const ParticleDims& dims);
  PositionalData AddExperience(const PositionalDims& dims);

  // Fill batch with randomly sampled data points.
  void SampleBatch(BatchData* batch, std::mt19937& rnd_gen) const;
 protected:
  void AdvanceHead();
};

enum ExpReplayInitPolicy {
  kGenerateDlcfrIterations,
  kGenerateRandomRangesAndSubgameValues,
};
ExpReplayInitPolicy GetInitPolicy(const std::string& s);  // Enum from string.

void AddExperiencesFromTrunk(
    const std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves,
    const std::vector<NetContext*>& net_contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    std::mt19937& rnd_gen, bool shuffle_input_output);

void GenerateDataRandomRanges(
    Trunk* trunk, const std::vector<NetContext*>& contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    double prob_pure_strat, double prob_fully_mixed,
    std::mt19937& rnd_gen, bool shuffle_input_output);

void GenerateDataDLCfrIterations(
    Trunk* trunk, const std::vector<NetContext*>& contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    int trunk_iters, std::function<void(/*trunk_iter=*/int)> monitor_fn,
    std::mt19937& rnd_gen, bool shuffle_input_output);

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EXPERIENCE_REPLAY_
