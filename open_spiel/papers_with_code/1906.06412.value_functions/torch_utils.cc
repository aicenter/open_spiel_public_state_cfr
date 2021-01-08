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

#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"


#include "torch/torch.h"

namespace open_spiel {
namespace papers_with_code {

torch::Device FindDevice() {
  if (torch::cuda::is_available()) {
    std::cout << "# CUDA available! Training on GPU." << std::endl;
    return torch::Device(torch::kCUDA);
  } else {
    std::cout << "# Training on CPU." << std::endl;
    return torch::Device(torch::kCPU);
  }
}

}  // papers_with_code
}  // open_spiel
