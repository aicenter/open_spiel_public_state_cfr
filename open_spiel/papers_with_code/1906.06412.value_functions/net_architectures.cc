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

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"

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

// Load all of the Slice, Ellipsis, etc.
using namespace torch::indexing;

void ValueNet::MakeLayers(std::vector<torch::nn::Linear>& layers, int num_layers,
                     int inputs_size, int hidden_size, int outputs_size) {
  for (int i = 0; i < num_layers; ++i) {
    size_t layer_input = i == 0 ? inputs_size : hidden_size;
    size_t layer_output = i == num_layers - 1 ? outputs_size : hidden_size;
    layers.emplace_back(layer_input, layer_output);
  }
}

void ValueNet::RegisterLayers(const std::vector<torch::nn::Linear>& layers,
                              const std::string& layer_name) {
  for (int i = 0; i < layers.size(); ++i) {
    register_module(absl::StrCat(layer_name, i), layers[i]);
  }
}

// -- PositionalValueNet -------------------------------------------------------

PositionalValueNet::PositionalValueNet(size_t inputs_size, size_t outputs_size,
                                       size_t hidden_size, size_t num_layers,
                                       ActivationFunction activation)
    : activation_fn(activation) {
  SPIEL_CHECK_GE(num_layers, 1);
  MakeLayers(fc_layers, num_layers, inputs_size, hidden_size, outputs_size);
  RegisterLayers(fc_layers, "fc_");
}

torch::Tensor PositionalValueNet::forward(torch::Tensor x) {
  int num_layers_with_activation = fc_layers.size() - 1;
  for (int i = 0; i < num_layers_with_activation; ++i) {
    x = fc_layers[i]->forward(x);
    x = Activation(activation_fn, x);
  }
  // Last layer has no activation function -- just linear output.
  return fc_layers.back()->forward(x);
}

// -- ParticleValueNet ---------------------------------------------------------

ParticleValueNet::ParticleValueNet(ParticleDims* particle_dims,
                                   ActivationFunction activation)
    : dims(particle_dims), activation_fn(activation) {
  int dim_context = context_size();
  int num_layers_kernel = 4;
  int num_layers_context = 3;

  MakeLayers(fc_kernel, num_layers_kernel,
             dims->particle_size() - 1, dims->particle_size(), dim_context);
  MakeLayers(fc_context, num_layers_context,
             dim_context, dim_context * 3, dims->max_particles);
  RegisterLayers(fc_kernel, "fc_kernel_");
  RegisterLayers(fc_context, "fc_context_");
}

torch::Tensor ParticleValueNet::kernel(torch::Tensor xs) {                      CHECK_SHAPE(xs, {_, dims->particle_size()});
  torch::Tensor coords = xs.index({  // Skip the range input.
      Slice(), Slice(0, dims->particle_size() - 1)});                           CHECK_SHAPE(coords, {_, dims->particle_size() - 1});
  torch::Tensor ranges = xs.index({  // Skip all features.
      Slice(), Slice(dims->particle_size() - 1, dims->particle_size())});       CHECK_SHAPE(ranges, {_, 1});

  for (int i = 0; i < fc_kernel.size() - 1; ++i) {
    coords = fc_kernel[i]->forward(coords);
    coords = Activation(activation_fn, coords);
  }
  coords = fc_kernel.back()->forward(coords);                                   CHECK_SHAPE(coords, {_, context_size()});
  torch::Tensor ks = coords.mul(ranges);                                        CHECK_SHAPE(ks, {_, context_size()});
  return ks;
}

torch::Tensor ParticleValueNet::pool(torch::Tensor xs) {                        CHECK_SHAPE(xs, {_, context_size()});
  torch::Tensor context = torch::sum(xs, {0});                                  CHECK_SHAPE(context, {context_size()});
  return context;
}

torch::Tensor ParticleValueNet::regression(torch::Tensor xs) {
  for (int i = 0; i < fc_context.size() - 1; ++i) {
    xs = fc_context[i]->forward(xs);
    xs = Activation(activation_fn, xs);
  }
  xs = fc_context.back()->forward(xs);                                          CHECK_SHAPE(xs, {_, positional_output()});
  return xs;
}

torch::Tensor ParticleValueNet::forward(torch::Tensor xss) {
  int batch_size = xss.size(0);
  CHECK_SHAPE(xss, {batch_size, dims->point_input_size()});

  std::vector<torch::Tensor> contexts;
  contexts.reserve(batch_size);
  for (torch::Tensor xs : xss.split(/*split_size=*/1, /*dim=*/0)) {             CHECK_SHAPE(xs, {1, dims->point_input_size()});
    xs = xs.squeeze(/*dim=*/0);                                                 CHECK_SHAPE(xs, {dims->point_input_size()});
    // Convert fp32 to int -- for our small numbers < 1e7 this is ok.
    int num_particles = xs[0].item<float_net>();                                SPIEL_CHECK_GT(num_particles, 0);
    SPIEL_CHECK_LT(num_particles, 1e7);
    const int num_particles_offset = 1;
    torch::Tensor particles = xs
        // Skip the num_particles item.
        .index({ Slice(1, num_particles_offset + num_particles * dims->particle_size()) })
        // Rearrange into particles.
        .reshape({num_particles, dims->particle_size()});                       CHECK_SHAPE(particles, {num_particles, dims->particle_size()});
    torch::Tensor context = pool(kernel(particles)).unsqueeze(/*dim=*/0);       CHECK_SHAPE(context, {1, context_size()});
    contexts.push_back(context);
  }

  torch::Tensor batches = torch::cat(contexts, /*dim=*/0);                      CHECK_SHAPE(batches, {batch_size, context_size()});
  torch::Tensor ys = regression(batches);                                       CHECK_SHAPE(ys, {batch_size, positional_output()});
  return ys;
}

}  // namespace papers_with_code
}  // namespace open_spiel


