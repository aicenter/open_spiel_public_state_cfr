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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_

#include "torch/torch.h"

namespace open_spiel {
namespace papers_with_code {

// A name-holder for various possible value network architectures.
struct ValueNet : public torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// A simple 3-layer MLP neural network.
struct PositionalValueNet final : public ValueNet {
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
  PositionalValueNet(size_t inputs_size, size_t outputs_size,
                     size_t hidden_size) :
      fc1(inputs_size, hidden_size),
      fc2(hidden_size, hidden_size),
      fc3(hidden_size, outputs_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::sigmoid(fc1->forward(x));
    x = torch::sigmoid(fc2->forward(x));
    return torch::sigmoid(fc3->forward(x));
  }
};

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
