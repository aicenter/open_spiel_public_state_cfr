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
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"

namespace open_spiel {
namespace papers_with_code {

// A base class for various possible value network architectures.
struct ValueNet : public torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
  virtual torch::Tensor PrepareTarget(BatchData* batch) = 0;

  void MakeLayers(std::vector<torch::nn::Linear>& layers, int num_layers,
                  int inputs_size, int hidden_size, int outputs_size);

  void RegisterLayers(const std::vector<torch::nn::Linear>& layers,
                      const std::string& layer_name);
};

enum ActivationFunction { kNone, kRelu, kLeakyRelu, kSigmoid };

inline torch::Tensor Activation(ActivationFunction f, torch::Tensor x) {
  switch (f) {
    case kNone:
      return x;
    case kRelu:
      return torch::relu(x);
    case kLeakyRelu:
      return torch::leaky_relu(x);
    case kSigmoid:
      return torch::sigmoid(x);
  }
}

// A simple MLP neural network.
struct PositionalValueNet final : public ValueNet {
  std::vector<torch::nn::Linear> fc_layers;
  ActivationFunction activation_fn;

  PositionalValueNet(size_t inputs_size, size_t outputs_size,
                     size_t hidden_size, size_t num_layers = 3,
                     ActivationFunction activation = kRelu);
  torch::Tensor forward(torch::Tensor x) override;
  torch::Tensor PrepareTarget(BatchData* batch) override {
      SpielFatalError("Not implemented!");
  };
};


struct ParticleValueNet final : public ValueNet {
  ParticleDims* dims;
  ActivationFunction activation_fn;
  std::vector<torch::nn::Linear> fc_context;
  std::vector<torch::nn::Linear> fc_basis;
  int limit_particle_count = -1;

  ParticleValueNet(ParticleDims* particle_dims,
                   ActivationFunction activation = kRelu);
  torch::Tensor forward(torch::Tensor xss) override;
  torch::Tensor PrepareTarget(BatchData* batch) override;

  torch::Tensor change_of_basis(torch::Tensor fs);
  torch::Tensor base_coordinates(torch::Tensor bs, torch::Tensor scales);
  torch::Tensor pool(torch::Tensor xs);
  torch::Tensor regression(torch::Tensor xs);

  int context_size() { return dims->max_particles; }
  int regression_size() { return dims->max_particles; }
};

#undef _

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
