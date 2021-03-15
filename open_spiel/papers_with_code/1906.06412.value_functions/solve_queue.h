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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVE_QUEUE_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVE_QUEUE_

#include <deque>

#include "open_spiel/spiel.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"


namespace open_spiel {
namespace papers_with_code {

// Store particle sets that should be solved next.
class SolveQueue {
  // Future implementation might compress the particle sets.
  std::deque<ParticleSet> queue_;
 public:
  void Add(const ParticleSet& set);
  const ParticleSet Get();
};

} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SOLVE_QUEUE_
