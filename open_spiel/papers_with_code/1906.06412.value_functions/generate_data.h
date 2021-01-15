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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_

#include <map>

#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"

namespace open_spiel {
namespace papers_with_code {

void RandomizeStrategy(std::vector<algorithms::BanditVector>& bandits,
                       std::mt19937& rnd_gen,
                       double prob_pure_strat = 0.0,
                       double prob_fully_mixed = 0.05);

void GenerateDataRandomRanges(Trunk* trunk, ExperienceReplay* replay,
                              std::mt19937& rnd_gen);

void GenerateDataDLCfrIterations(
    Trunk* trunk, ExperienceReplay* replay, int trunk_iters,
    std::function<void(/*trunk_iter=*/int)> monitor_fn);

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_
