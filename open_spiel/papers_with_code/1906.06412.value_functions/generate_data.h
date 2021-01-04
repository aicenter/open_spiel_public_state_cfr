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

#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_batch.h"

namespace open_spiel {
namespace papers_with_code {

using float_net = float;    // Floats used in the neural network.
using float_tree = double;  // Floats used in the cfr computation.

// Copy non-contiguous vectors using a permutation map.
// This also converts float <-> double as needed.
template<typename From, typename To>
void PlacementCopy(absl::Span<const From> from, absl::Span<To> to,
                   std::map<size_t, size_t> from_to) {
  for (const auto&[f, t] : from_to) {
    to[t] = from[f];
  }
}

void RandomizeTrunkStrategy(std::vector<algorithms::BanditVector>& bandits,
                            std::mt19937& rnd_gen, double prob_pure_strat);

void GenerateData(const std::vector<algorithms::dlcfr::RangeTable>& tables,
                  algorithms::dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                  std::mt19937& rnd_gen, bool verbose = false);

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_
