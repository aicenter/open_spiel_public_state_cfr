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

constexpr const char* kDefaultArch = "particle_vf";

// A base class for various possible value network architectures.
struct ValueNet : public torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
  virtual torch::Tensor PrepareTarget(BatchData* batch) = 0;
  virtual NetArchitecture architecture() const = 0;

  void MakeLayers(std::vector<torch::nn::Linear>& layers, int num_layers,
                  int inputs_size, int hidden_size, int outputs_size);
  void RegisterLayers(const std::vector<torch::nn::Linear>& layers,
                      const std::string& layer_name);
};

enum ActivationFunction { kNone, kRelu, kLeakyRelu, kSigmoid };
torch::Tensor Activation(ActivationFunction f, torch::Tensor x);

// A simple MLP neural network.
struct PositionalValueNet final : public ValueNet {
  std::vector<torch::nn::Linear> fc_regression;
  ActivationFunction activation_fn;

  PositionalValueNet(std::shared_ptr<PositionalDims> positional_dims,
                     size_t num_layers_regression, size_t num_width_regression,
                     ActivationFunction activation = kRelu);
  torch::Tensor forward(torch::Tensor x) override;
  torch::Tensor PrepareTarget(BatchData* batch) override {
    return batch->target;
  };
  NetArchitecture architecture() const override {
    return NetArchitecture::kPositional;
  }
};

// A particle neural network that uses MLPs for regression and change of basis.
struct ParticleValueNet final : public ValueNet {
  std::shared_ptr<ParticleDims> dims;
  bool zero_sum_regression;
  ActivationFunction activation_fn;
  std::vector<torch::nn::Linear> fc_regression;
  std::vector<torch::nn::Linear> fc_basis;
  size_t num_inputs_regression;

  ParticleValueNet(std::shared_ptr<ParticleDims> particle_dims,
                   size_t num_layers_regression,
                   size_t num_width_regression,
                   size_t num_inputs_regression,
                   bool zero_sum_regression,
                   ActivationFunction activation = kRelu);
  torch::Tensor forward(torch::Tensor xss) override;
  torch::Tensor PrepareTarget(BatchData* batch) override;
  NetArchitecture architecture() const override {
    return NetArchitecture::kParticle;
  }

  torch::Tensor change_of_basis(torch::Tensor fs);
  torch::Tensor base_coordinates(torch::Tensor bs, torch::Tensor scales);
  torch::Tensor pool(torch::Tensor xs);
  torch::Tensor regression(torch::Tensor xs);

  int context_size() { return pooled_size() + dims->public_features_size; }
  int pooled_size() { return num_inputs_regression; }
  int regression_size() { return num_inputs_regression; }
};

void InitWeights(torch::nn::Module& m);

std::shared_ptr<ValueNet> MakeModel(
    NetArchitecture arch, std::shared_ptr<BasicDims> dims,
    int num_layers_regression, int num_width_regression,
    int num_inputs_regression, bool zero_sum_regression);


}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
