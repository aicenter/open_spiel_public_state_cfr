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

#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"


namespace open_spiel {
namespace papers_with_code {

void ExperienceReplay::SelectRandomExperience(BatchData* write_to,
                                              std::mt19937& rnd_gen) {
  SPIEL_CHECK_FALSE(buffer.empty());
  const int num_batches = buffer.size();
  int index = std::uniform_int_distribution<>(0, num_batches - 1)(rnd_gen);
  write_to->CopyFrom(buffer[index]);
}

}  // papers_with_code
}  // open_spiel
