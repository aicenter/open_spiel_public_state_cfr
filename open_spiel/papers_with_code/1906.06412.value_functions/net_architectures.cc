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

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

#define _ -1  // A shape placeholder.
#ifndef NDEBUG
void CHECK_SHAPE(const torch::Tensor& tensor,
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
void CHECK_SHAPE(const torch::Tensor& tensor,
                 std::initializer_list<int64_t> shape) {}
#endif

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

PositionalValueNet::PositionalValueNet(std::shared_ptr<PositionalDims> positional_dims,
                                       size_t num_width_regression,
                                       size_t num_layers_regression,
                                       ActivationFunction activation)
    : activation_fn(activation) {
  MakeLayers(fc_regression, num_layers_regression,
             positional_dims->point_input_size(),
             positional_dims->point_input_size() * num_width_regression,
             positional_dims->point_output_size());
  RegisterLayers(fc_regression, "fc_");
}

torch::Tensor PositionalValueNet::forward(torch::Tensor x) {
  int num_layers_with_activation = fc_regression.size() - 1;
  for (int i = 0; i < num_layers_with_activation; ++i) {
    x = fc_regression[i]->forward(x);
    x = Activation(activation_fn, x);
  }
  // Last layer has no activation function -- just linear output.
  return fc_regression.back()->forward(x);
}

// -- ParticleValueNet ---------------------------------------------------------

ParticleValueNet::ParticleValueNet(std::shared_ptr<ParticleDims> particle_dims,
                                   size_t num_layers_regression,
                                   size_t num_width_regression,
                                   size_t num_inputs_regression,
                                   bool zero_sum_regression,
                                   ActivationFunction activation)
    : dims(particle_dims),
      zero_sum_regression(zero_sum_regression),
      activation_fn(activation),
      num_inputs_regression(num_inputs_regression) {
  int num_layers_kernel = 4;

  MakeLayers(fc_basis, num_layers_kernel,
             dims->full_features_size(),
             (dims->full_features_size() + 1) * 3,
             pooled_size());
  MakeLayers(fc_regression, num_layers_regression,
             context_size(),
             context_size() * num_width_regression,
             regression_size());
  RegisterLayers(fc_basis, "fc_basis_");
  RegisterLayers(fc_regression, "fc_regression_");
}

torch::Tensor ParticleValueNet::change_of_basis(torch::Tensor fs) {             CHECK_SHAPE(fs, {_, dims->full_features_size()});
  for (int i = 0; i < fc_basis.size() - 1; ++i) {
    fs = fc_basis[i]->forward(fs);
    fs = Activation(activation_fn, fs);
  }
  torch::Tensor bs = fc_basis.back()->forward(fs);                              CHECK_SHAPE(bs, {_, pooled_size()});
  return bs;
}

torch::Tensor ParticleValueNet::base_coordinates(torch::Tensor bs,
                                                 torch::Tensor scales) {
  CHECK_SHAPE(bs, {_, pooled_size()});
  CHECK_SHAPE(scales, {_, 1});
  torch::Tensor cs = bs.mul(scales);                                            CHECK_SHAPE(cs, {_, pooled_size()});
  return cs;
}

torch::Tensor ParticleValueNet::pool(torch::Tensor cs) {                        CHECK_SHAPE(cs, {_, pooled_size()});
  torch::Tensor context = torch::sum(cs, {0});                                  CHECK_SHAPE(context, {pooled_size()});
  return context;
}

torch::Tensor ParticleValueNet::regression(torch::Tensor xs) {                  CHECK_SHAPE(xs, {1, context_size()});
  for (int i = 0; i < fc_regression.size() - 1; ++i) {
    xs = fc_regression[i]->forward(xs);
    xs = Activation(activation_fn, xs);
  }
  xs = fc_regression.back()->forward(xs);                                       CHECK_SHAPE(xs, {1, regression_size()});
  return xs;
}

torch::Tensor ParticleValueNet::forward(torch::Tensor xss) {
  int batch_size = xss.size(0);
  CHECK_SHAPE(xss, {batch_size, dims->point_input_size()});

  std::vector<torch::Tensor> out;
  out.reserve(batch_size);
  int total_batch_parviews = 0;
  for (int i = 0; i < batch_size; ++i) {
    torch::Tensor xs = xss[i];                                                  CHECK_SHAPE(xs, {dims->point_input_size()});
    // Convert fp32 to int -- for our small numbers < 1e7 this is ok.
    std::array<int, 2> player_parviews = {xs[0].item<int>(),
                                          xs[1].item<int>()};
    SPIEL_DCHECK({
      for (int pl = 0; pl < 2; ++pl) {
        SPIEL_CHECK_GT(player_parviews[pl], 0);
        SPIEL_CHECK_LT(player_parviews[pl], 1e7);
      }
    });
    int num_parviews = player_parviews[0] + player_parviews[1];

    // FIXME: nice offsets!
    const int pub_features_offset = 2;
    torch::Tensor public_features = xs
        // Skip the num_parviews item.
        .index({
          Slice(pub_features_offset,
                pub_features_offset + dims->public_features_size)
        })
        .reshape({1, dims->public_features_size});                              CHECK_SHAPE(public_features, {1, dims->public_features_size});

    const int num_parviews_offset = 2 + dims->public_features_size;
    torch::Tensor parviews = xs
        // Skip the num_parviews + public features item.
        .index({
          Slice(num_parviews_offset,
                num_parviews_offset + num_parviews * dims->parview_size())
        })
        // Rearrange into parviews.
        .reshape({num_parviews, dims->parview_size()});                         CHECK_SHAPE(parviews, {num_parviews,
                                                                                                       dims->parview_size()});

    torch::Tensor hand_fs = parviews  // Skip the range input (as the last value).
        .index({Slice(), Slice(0, dims->features_size())});                     CHECK_SHAPE(hand_fs, {num_parviews, dims->features_size()});
    torch::Tensor fs = torch::cat({
        public_features.expand({num_parviews, -1}), hand_fs}, /*dim=*/1);       CHECK_SHAPE(fs, {num_parviews, dims->full_features_size()});
    torch::Tensor ranges = parviews  // Skip all features.
        .index({Slice(), Slice(dims->features_size(),
                               dims->parview_size())});                         CHECK_SHAPE(ranges, {num_parviews, 1});

    torch::Tensor bs = change_of_basis(fs);                                     CHECK_SHAPE(bs, {num_parviews, pooled_size()});
    torch::Tensor cs = base_coordinates(bs, ranges);                            CHECK_SHAPE(cs, {num_parviews, pooled_size()});
    torch::Tensor pooled = pool(cs).unsqueeze(/*dim=*/0);                       CHECK_SHAPE(pooled, {1, pooled_size()});
    torch::Tensor context = torch::cat({pooled, public_features}, /*dim=*/1);   CHECK_SHAPE(context, {1, context_size()});
    torch::Tensor ys = regression(context).expand({num_parviews, -1});          CHECK_SHAPE(ys, {num_parviews, regression_size()});
    torch::Tensor proj = (ys * bs).sum(/*dim=*/1);                              CHECK_SHAPE(proj, {num_parviews});
    if (zero_sum_regression) {
      torch::Tensor ranges1d = ranges.squeeze(/*dim=*/1);                       CHECK_SHAPE(ranges1d, {num_parviews});
      if (ranges1d.sum().item<float>() == 0) {
        proj.zero_();
      } else {
        // beliefs * values = 0 (because game is zero-sum) and vectors are
        // therefore perpendicular. If values are off and are not zero-sum,
        // we project them to the beliefs plane (beliefs are the normal vector).
        torch::Tensor proj_error = (proj.dot(ranges1d)) / (ranges1d.dot(ranges1d));
        proj = proj - proj_error * ranges1d;
      }
      SPIEL_DCHECK_FLOAT_NEAR((proj.dot(ranges1d)).sum().item<float>(), 0., 1e-6);
    }
    SPIEL_CHECK_FALSE(torch::isfinite(proj).logical_not().any().item<bool>());
    out.push_back(proj);
    total_batch_parviews += num_parviews;
  }
  torch::Tensor y = torch::cat(out);                                            CHECK_SHAPE(y, {total_batch_parviews});
  return y;
}

torch::Tensor ParticleValueNet::PrepareTarget(BatchData* batch) {
  torch::Tensor parviews_counts =
      batch->data.index({Slice(), Slice(0, 2)}).sum({1});

  std::vector<torch::Tensor> target_slices;
  target_slices.reserve(batch->size());
  for (int i = 0; i < batch->size(); ++i) {
    int num_parviews = parviews_counts[i].item<int>();
    target_slices.push_back(batch->target.index({i, Slice(0, num_parviews)}));
  }
  return torch::cat(target_slices).unsqueeze(/*dim=*/0);
}

// TODO: test that this indeed initializes weights differently each call.
void InitWeights(torch::nn::Module& m) {
  auto p = m.named_parameters(false);
  auto w = p.find("weight");
  auto b = p.find("bias");
  if (w != nullptr) {
    torch::nn::init::xavier_uniform_(*w);
  }
  if (b != nullptr) {
    torch::nn::init::constant_(*b, 0.01);
  }
}

NetArchitecture GetArchitecture(const std::string& arch) {
  if (arch == "particle_vf") {
    return NetArchitecture::kParticle;
  } else  if (arch == "positional_vf") {
    return NetArchitecture::kPositional;
  } else {
    SpielFatalError("Exhausted pattern match! Architecture not recognized.");
  }
}

torch::Tensor Activation(ActivationFunction f, torch::Tensor x) {
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
std::shared_ptr<ValueNet> MakeModel(
    NetArchitecture arch,
    std::shared_ptr<BasicDims> dims,
    int num_layers_regression,
    int num_width_regression,
    int num_inputs_regression,
    bool zero_sum_regression
) {
  SPIEL_CHECK_GE(num_layers_regression, 1);
  SPIEL_CHECK_GE(num_width_regression, 1);
  switch (arch) {
    case NetArchitecture::kParticle: {
      auto particle_dims = std::dynamic_pointer_cast<ParticleDims>(dims);
      auto model = std::make_shared<ParticleValueNet>(
          particle_dims,
          num_layers_regression, num_width_regression, num_inputs_regression,
          zero_sum_regression,
          ActivationFunction::kRelu);
      return model;
    }
    case NetArchitecture::kPositional: {
      auto positional_dims = std::dynamic_pointer_cast<PositionalDims>(dims);
      return std::make_shared<PositionalValueNet>(
          positional_dims, num_layers_regression, num_width_regression,
          ActivationFunction::kRelu);
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel


