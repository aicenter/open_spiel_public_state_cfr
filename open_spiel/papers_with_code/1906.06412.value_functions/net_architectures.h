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

#define _ -1  // A shape placeholder.
#ifndef NDEBUG
inline void CHECK_SHAPE(const torch::Tensor& tensor,
                 std::initializer_list<int64_t> shape) {
  const std::vector<int64_t> expected_shape(shape);
  SPIEL_DCHECK_EQ(tensor.dim(), expected_shape.size());
  for (int i = 0; i < expected_shape.size(); i++) {
    if (expected_shape[i] == _) continue;
    if (tensor.sizes().at(i) != expected_shape[i]) {
      std::string actual_str = absl::StrJoin(tensor.sizes().vec(), ",");
      std::string expected_str = absl::StrJoin(expected_shape, ",");
      SpielFatalError(absl::StrCat(
          "CHECK_SHAPE: ",
          tensor.sizes().at(i), " != ", expected_shape[i], " at index ", i,
          " -- full shapes: actual ", actual_str, " expected ", expected_str));
    }
  }
}
#else
inline void CHECK_SHAPE(const torch::Tensor& tensor,
                        std::initializer_list<int64_t> shape) {}
#endif


// A name-holder for various possible value network architectures.
struct ValueNet : public torch::nn::Module {
  virtual torch::Tensor forward(torch::Tensor x) = 0;
};

enum ActivationFunction {
  kNone, kRelu, kLeakyRelu, kSigmoid
};

inline void MakeLayers(std::vector<torch::nn::Linear>& layers, int num_layers,
                       int inputs_size, int hidden_size, int outputs_size) {
  for (int i = 0; i < num_layers; ++i) {
    size_t layer_input = i == 0 ? inputs_size : hidden_size;
    size_t layer_output = i == num_layers - 1 ? outputs_size : hidden_size;
    layers.emplace_back(layer_input, layer_output);
  }
}

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
                     ActivationFunction activation = kRelu)
      : activation_fn(activation) {
    SPIEL_CHECK_GE(num_layers, 1);
    MakeLayers(fc_layers, num_layers, inputs_size, hidden_size, outputs_size);
    for (int i = 0; i < num_layers; ++i) {
      register_module(absl::StrCat("fc_", i), fc_layers[i]);
    }
  }


  torch::Tensor forward(torch::Tensor x) {
    int num_layers_with_activation = fc_layers.size() - 1;
    for (int i = 0; i < num_layers_with_activation; ++i) {
      x = fc_layers[i]->forward(x);
      x = Activation(activation_fn, x);
    }
    // Last layer has no activation function -- just linear output.
    return fc_layers.back()->forward(x);
  }
};


struct ParticleValueNet final : public ValueNet {
  ParticleDims* dims;
  ActivationFunction activation_fn;
  std::vector<torch::nn::Linear> fc_context;
  std::vector<torch::nn::Linear> fc_kernel;

  int context_size() { return dims->max_particles * 2; }
  int positional_output() { return dims->max_particles * 2; }

  ParticleValueNet(ParticleDims* particle_dims,
                   ActivationFunction activation = kRelu)
      : dims(particle_dims), activation_fn(activation) {
    int dim_context = context_size();
    int num_layers_kernel = 4;
    int num_layers_context = 3;

    MakeLayers(fc_kernel, num_layers_kernel,
               dims->particle_size(), dims->particle_size() * 3, dim_context);
    MakeLayers(fc_context, num_layers_context,
               dim_context, dim_context * 3, dims->max_particles);
    RegisterLayers(fc_kernel, "fc_kernel_");
    RegisterLayers(fc_context, "fc_context_");
  }

  void RegisterLayers(const std::vector<torch::nn::Linear>& layers,
                      const std::string& layer_name) {
    for (int i = 0; i < layers.size(); ++i) {
      register_module(absl::StrCat(layer_name, i), layers[i]);
    }
  }

  torch::Tensor kernel(torch::Tensor xs) {                                      CHECK_SHAPE(xs, {_, dims->particle_size()});
    for (int i = 0; i < fc_kernel.size() - 1; ++i) {
      xs = fc_kernel[i]->forward(xs);
      xs = Activation(activation_fn, xs);
    }
    xs = fc_kernel.back()->forward(xs);                                         CHECK_SHAPE(xs, {_, context_size()});
    return xs;
  }

  torch::Tensor pool(torch::Tensor xs) {                                        CHECK_SHAPE(xs, {_, context_size()});
    torch::Tensor context = torch::sum(xs, {0});                                CHECK_SHAPE(xs, {context_size()});
    return context;
  }

  torch::Tensor regression(torch::Tensor xs) {
    for (int i = 0; i < fc_context.size() - 1; ++i) {
      xs = fc_kernel[i]->forward(xs);
      xs = Activation(activation_fn, xs);
    }
    xs = fc_kernel.back()->forward(xs);                                         CHECK_SHAPE(xs, {_, positional_output()});
    return xs;
  }

  torch::Tensor forward(torch::Tensor xss) {
    CHECK_SHAPE(xss, {_, dims->point_input_size()});

    std::vector<torch::Tensor> contexts;
    for (torch::Tensor xs : xss.split(/*split_size=*/1, /*dim=*/0)) {           CHECK_SHAPE(xs, {dims->point_input_size()});
      // Convert to int -- for our small numbers this is ok.
      int num_particles = xs[0].item<float_net>();

      torch::Tensor particles = xs
          .slice(/*dim=*/0,
                 /*start*/1,  // Skip the num_particles item.
                 /*end=*/dims->particle_size() * num_particles + 1, /*step=*/1)
          .reshape({num_particles, dims->particle_size()});                     CHECK_SHAPE(particles, {num_particles, dims->particle_size()});
      torch::Tensor context = pool(kernel(particles));                          CHECK_SHAPE(context, {context_size()});
      contexts.push_back(context);
    }

    torch::Tensor batches = torch::cat(contexts, /*dim=*/0);                    CHECK_SHAPE(batches, {_, context_size()});
    torch::Tensor ys = regression(batches);                                     CHECK_SHAPE(ys, {_, positional_output()});
    return ys;
  }
};

#undef _

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NEURAL_NETS_
