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

#include "open_spiel/papers_with_code/1906.06412.value_functions/solve_queue.h"

namespace open_spiel {
namespace papers_with_code {

void SolveQueue::Add(const ParticleSet& set) {
  queue_.push_back(set);
}

const ParticleSet SolveQueue::Get() {
  ParticleSet set = queue_.front();
  queue_.pop_front();
  return set;
}

}  // papers_with_code
}  // open_spiel
