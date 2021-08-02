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

#include <filesystem>
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"

namespace open_spiel {
namespace papers_with_code {

void SaveNetSnapshot(std::shared_ptr<ValueNet> model, const std::string& path) {
  torch::save(model, path);
}

void LoadNetSnapshot(std::shared_ptr<ValueNet> model, const std::string& path,
                     BatchData* try_inference) {

  try {
    torch::load(model, path);
    if (try_inference) {
      // If the dimensions are not compatible, the program will fail here.
      try_inference->data.zero_();
      model->forward(try_inference->data);
    }
  } catch (std::exception e) {
    std::cerr << "Could not load the network snapshot! At "
              << path << std::endl;
    throw std::move(e);
  }
}

std::string FindSnapshot(const std::string& snapshot_dir) {
  // Search for all files that end with kModelExt, and select
  // a file with the highest loop number.

  namespace fs = std::filesystem;
  int max_loop = -1;
  for (const auto& entry : fs::directory_iterator(snapshot_dir)) {
    if(entry.path().extension() != kModelExt) continue;
    size_t num_chars;
    int loop = std::stoi(entry.path().filename(), &num_chars);
    if (num_chars) max_loop = std::max(max_loop, loop);
  }

  if (max_loop == -1) return "";  // No snapshot found.

  std::string snapshot = snapshot_dir + "/" + std::to_string(max_loop) + kModelExt;
  SPIEL_CHECK_TRUE(fs::exists(fs::path(snapshot)));
  return snapshot;
}

}  // namespace papers_with_code
}  // namespace open_spiel


