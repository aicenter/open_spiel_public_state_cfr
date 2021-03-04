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

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

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
                                   size_t num_layers_regression,
                                   size_t num_width_regression,
                                   ActivationFunction activation)
    : dims(particle_dims), activation_fn(activation) {
  int dim_context = context_size();
  int num_layers_kernel = 4;

  MakeLayers(fc_basis, num_layers_kernel,
             dims->particle_size() - 1, dims->particle_size(), dim_context);
  MakeLayers(fc_regression, num_layers_regression, dim_context,
             dim_context * num_width_regression, regression_size());
  RegisterLayers(fc_basis, "fc_basis_");
  RegisterLayers(fc_regression, "fc_regression_");
}

torch::Tensor ParticleValueNet::change_of_basis(torch::Tensor fs) {             CHECK_SHAPE(fs, {_, dims->features_size()});
  for (int i = 0; i < fc_basis.size() - 1; ++i) {
    fs = fc_basis[i]->forward(fs);
    fs = Activation(activation_fn, fs);
  }
  torch::Tensor bs = fc_basis.back()->forward(fs);                              CHECK_SHAPE(bs, {_, context_size()});
  return bs;
}

torch::Tensor ParticleValueNet::base_coordinates(torch::Tensor bs,
                                                 torch::Tensor scales) {
  CHECK_SHAPE(bs, {_, context_size()});
  CHECK_SHAPE(scales, {_, 1});
  torch::Tensor cs = bs.mul(scales);                                            CHECK_SHAPE(cs, {_, context_size()});
  return cs;
}

torch::Tensor ParticleValueNet::pool(torch::Tensor cs) {                        CHECK_SHAPE(cs, {_, context_size()});
  torch::Tensor context = torch::sum(cs, {0});                                  CHECK_SHAPE(context, {context_size()});
  return context;
}

torch::Tensor ParticleValueNet::regression(torch::Tensor xs) {                  CHECK_SHAPE(xs, {1, context_size()});
  for (int i = 0; i < fc_regression.size() - 1; ++i) {
    xs = fc_regression[i]->forward(xs);
    xs = Activation(activation_fn, xs);
  }
  xs = fc_regression.back()->forward(xs);                                          CHECK_SHAPE(xs, {1, regression_size()});
  return xs;
}

torch::Tensor ParticleValueNet::forward(torch::Tensor xss) {
  int batch_size = xss.size(0);
  CHECK_SHAPE(xss, {batch_size, dims->point_input_size()});

  std::vector<torch::Tensor> out;
  out.reserve(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    torch::Tensor xs = xss[i];                                                  CHECK_SHAPE(xs, {dims->point_input_size()});
    // Convert fp32 to int -- for our small numbers < 1e7 this is ok.
    int num_particles = xs[0].item<float_net>();                                SPIEL_CHECK_GT(num_particles, 0);
                                                                                SPIEL_CHECK_LT(num_particles, 1e7);
    // Take only an ordered subset of particles, as specified by the amount.
    if (is_training() && limit_particle_count > 0) {
      num_particles = std::min(num_particles, limit_particle_count);
    }

    const int num_particles_offset = 1;
    torch::Tensor particles = xs
        // Skip the num_particles item.
        .index({ Slice(1, num_particles_offset
                          + num_particles * dims->particle_size()) })
        // Rearrange into particles.
        .reshape({num_particles, dims->particle_size()});                       CHECK_SHAPE(particles, {num_particles, dims->particle_size()});

    torch::Tensor fs = particles  // Skip the range input (as the last value).
        .index({Slice(), Slice(0, dims->features_size())});                     CHECK_SHAPE(fs, {num_particles, dims->features_size()});
    torch::Tensor ranges = particles  // Skip all features.
        .index({Slice(), Slice(dims->features_size(), dims->particle_size())}); CHECK_SHAPE(ranges, {num_particles, 1});

    torch::Tensor bs = change_of_basis(fs);                                     CHECK_SHAPE(bs, {num_particles, context_size()});
    torch::Tensor cs = base_coordinates(bs, ranges);                            CHECK_SHAPE(cs, {num_particles, context_size()});
    torch::Tensor context = pool(cs).unsqueeze(/*dim=*/0);                      CHECK_SHAPE(context, {1, context_size()});
    torch::Tensor ys = regression(context).expand({num_particles, -1});         CHECK_SHAPE(ys, {num_particles, regression_size()});
    torch::Tensor proj = (ys * bs).sum(/*dim=*/1).unsqueeze(0);                 CHECK_SHAPE(proj, {1, num_particles});
    out.push_back(proj);
  }
  return torch::cat(out, /*dim=*/1);
}

torch::Tensor ParticleValueNet::PrepareTarget(BatchData* batch) {
  torch::Tensor particles_data = batch->data.index({Slice(), 0});

  std::vector<torch::Tensor> target_slices;
  target_slices.reserve(batch->size());
  for (int i = 0; i < batch->size(); ++i) {
    int num_particles = particles_data[i].item<int>();
    if (limit_particle_count > 0) {
      num_particles = std::min(num_particles, limit_particle_count);
    }
    target_slices.push_back(batch->target.index({i, Slice(0, num_particles)}));
  }
  return torch::cat(target_slices).unsqueeze(/*dim=*/0);
}

}  // namespace papers_with_code
}  // namespace open_spiel


