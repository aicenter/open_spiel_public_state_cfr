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
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

// A name-holder for various possible value network architectures.
struct ValueNet : public torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
};

// A simple MLP neural network.
struct PositionalValueNet final : public ValueNet {
  std::vector<torch::nn::Linear> fc_layers;

  enum ActivationFunction {
    kNone, kRelu, kLeakyRelu, kSigmoid
  };

  ActivationFunction activation_fn;

  PositionalValueNet(size_t inputs_size, size_t outputs_size,
                     size_t hidden_size, size_t num_layers = 3,
                     ActivationFunction activation = kRelu)
      : activation_fn(activation) {
    SPIEL_CHECK_GE(num_layers, 1);
    for (int i = 0; i < num_layers; ++i) {
      size_t layer_input = i == 0 ? inputs_size : hidden_size;
      size_t layer_output = i == num_layers - 1 ? outputs_size : hidden_size;
      fc_layers.emplace_back(layer_input, layer_output);
      register_module(absl::StrCat("fc", i), fc_layers.back());
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    int num_layers_with_activation = fc_layers.size() - 1;
    for (int i = 0; i < num_layers_with_activation; ++i) {
      x = fc_layers[i]->forward(x);
      switch (activation_fn) {
        case kNone:
          break;
        case kRelu:
          x = torch::relu(x);
          break;
        case kLeakyRelu:
          x = torch::leaky_relu(x);
          break;
        case kSigmoid:
          x = torch::sigmoid(x);
          break;
      }
    }
    // Last layer has no activation function -- just linear output.
    return fc_layers.back()->forward(x);
  }
};

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
