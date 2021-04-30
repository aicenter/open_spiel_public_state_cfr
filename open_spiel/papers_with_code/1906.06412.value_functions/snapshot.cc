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

#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"

namespace open_spiel {
namespace papers_with_code {

void SaveNetSnapshot(std::shared_ptr<ValueNet> model, const std::string& path) {
  torch::save(model, path);
}

std::shared_ptr<ValueNet> LoadNetSnapshot(const std::string& path) {
  return nullptr; // TODO
}

}  // namespace papers_with_code
}  // namespace open_spiel


