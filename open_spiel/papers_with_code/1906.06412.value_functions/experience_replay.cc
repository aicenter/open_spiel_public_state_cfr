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

PositionalData ExperienceReplay::AddExperience(const PositionalDataDims& dims) {
  PositionalData point = point_at(head_, dims);
  AdvanceHead();
  return point;
}

void ExperienceReplay::AdvanceHead() {
  ++head_;
  if (head_ >= size()) {
    head_ = 0;
    ++overflow_cnt_;
  }
}

void ExperienceReplay::SampleBatch(BatchData* batch,
                                   std::mt19937& rnd_gen) const {
  // Do not sample non-filled experiences.
  const int n = overflow_cnt_ == 0 ? head_ : size();
  const int k = batch->size();
  SPIEL_CHECK_GE(n, k);

  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rnd_gen);

  for (int i = 0; i < k; ++i) {
    batch->data[i].copy_(data[perm[i]]);
    batch->target[i].copy_(target[perm[i]]);
  }
}

}  // papers_with_code
}  // open_spiel
