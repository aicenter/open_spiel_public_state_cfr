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
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"

namespace open_spiel {
namespace papers_with_code {

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

bool ValueNet::isfinite() const {
  for (const std::shared_ptr<torch::nn::Module>& m : modules()) {
      auto p = m->named_parameters(false);
      auto w = p.find("weight");
      auto b = p.find("bias");
      if (w != nullptr) {
        if (!w->isfinite().all().item<bool>()) return false;
      }
      if (b != nullptr) {
        if (!b->isfinite().all().item<bool>()) return false;
      }
  }
  return true;
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
                                   bool normalize_beliefs,
                                   ActivationFunction activation)
    : dims(particle_dims),
      zero_sum_regression(zero_sum_regression),
      normalize_beliefs(normalize_beliefs),
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

torch::Tensor ParticleValueNet::change_of_basis(torch::Tensor fs) {
  const int batch_size = fs.size(0);
  const int num_parviews = dims->max_parviews;
  const int big_batch = batch_size * num_parviews;                              CHECK_SHAPE(fs, {batch_size, num_parviews, dims->full_features_size()});

  // Turn batch of parviews into a big batch.
  torch::Tensor big_batch_fs = fs.view({-1, dims->full_features_size()});       CHECK_SHAPE(big_batch_fs, {big_batch, dims->full_features_size()});

  for (int i = 0; i < fc_basis.size() - 1; ++i) {
    big_batch_fs = fc_basis[i]->forward(big_batch_fs);
    big_batch_fs = Activation(activation_fn, big_batch_fs);
  }
  torch::Tensor big_batch_bs = fc_basis.back()->forward(big_batch_fs);          CHECK_SHAPE(big_batch_bs, {big_batch, pooled_size()});
  torch::Tensor bs =
      big_batch_bs.view({batch_size, num_parviews, pooled_size()});             CHECK_SHAPE(bs, {batch_size, num_parviews, pooled_size()});
  return bs;
}

torch::Tensor ParticleValueNet::base_coordinates(torch::Tensor bs,
                                                 torch::Tensor scales) {
  const int batch_size = bs.size(0);
  const int num_parviews = dims->max_parviews;

  CHECK_SHAPE(bs, {batch_size, num_parviews, pooled_size()});
  CHECK_SHAPE(scales, {batch_size, num_parviews, 1});
  torch::Tensor cs = bs.mul(scales);                                            CHECK_SHAPE(cs, {batch_size, num_parviews, pooled_size()});
  return cs;
}

torch::Tensor ParticleValueNet::pool(torch::Tensor cs) {
  const int batch_size = cs.size(0);
  const int num_parviews = dims->max_parviews;

  CHECK_SHAPE(cs, {batch_size, num_parviews, pooled_size()});
  torch::Tensor context = torch::sum(cs, {1});                                  CHECK_SHAPE(context, {batch_size, pooled_size()});
  return context;
}

torch::Tensor ParticleValueNet::regression(torch::Tensor xs) {                  CHECK_SHAPE(xs, {_, context_size()});
  for (int i = 0; i < fc_regression.size() - 1; ++i) {
    xs = fc_regression[i]->forward(xs);
    xs = Activation(activation_fn, xs);
  }
  xs = fc_regression.back()->forward(xs);                                       CHECK_SHAPE(xs, {_, regression_size()});
  return xs;
}

torch::Tensor ParticleValueNet::forward(torch::Tensor xss) {
  const torch::Device device = xss.device();
  const int batch_size = xss.size(0);
  const int max_parviews = dims->max_parviews;
  // FIXME: offsets coming from basic structures!
  const int pub_features_offset = 2;
  const int num_parviews_offset = 2 + dims->public_features_size;
  const auto Batch = Slice();
  const auto Parviews = Slice();
  CHECK_SHAPE(xss, {batch_size, dims->point_input_size()});
  SPIEL_DCHECK_TRUE(torch::isfinite(xss).all().item<bool>());

  torch::Tensor public_features = xss.index({Batch,
      // Skip the num_parviews item.
      Slice(pub_features_offset,
            pub_features_offset + dims->public_features_size)});                CHECK_SHAPE(public_features, {batch_size, dims->public_features_size});

  torch::Tensor parviews = xss.index({Batch,
      // Skip the num_parviews + public features item.
      Slice(num_parviews_offset,
            num_parviews_offset + max_parviews * dims->parview_size())
    // Rearrange into parviews.
    }).view({batch_size, max_parviews, dims->parview_size()});                  CHECK_SHAPE(parviews, {batch_size, max_parviews, dims->parview_size()});

  torch::Tensor hand_fs = parviews
      // Skip the range input (as the last value).
      .index({Batch, Parviews, Slice(0, dims->features_size())});               CHECK_SHAPE(hand_fs, {batch_size, max_parviews, dims->features_size()});
  torch::Tensor beliefs = parviews  // Skip all features.
      .index({Batch, Parviews,
              Slice(dims->features_size(), dims->parview_size())});             CHECK_SHAPE(beliefs, {batch_size, max_parviews, 1});

  // Construct full features by concatenating hands with public features.
  torch::Tensor pf_per_parview = public_features
      .expand({max_parviews, -1, -1}).permute({1, 0, 2});                       CHECK_SHAPE(pf_per_parview, {batch_size, max_parviews, dims->public_features_size});
  torch::Tensor infostate_fs =
      torch::cat({pf_per_parview, hand_fs}, /*dim=*/2);                         CHECK_SHAPE(infostate_fs, {batch_size, max_parviews, dims->full_features_size()});

  // Zero-out public state features for non-full sets.
  torch::Tensor parview_counts =
      xss.index({Batch, Slice(0, pub_features_offset)});                        CHECK_SHAPE(parview_counts, {batch_size, 2});
  torch::Tensor parview_sum = parview_counts.sum(/*dim=*/1);                    CHECK_SHAPE(parview_sum, {batch_size});
  for (int i = 0; i < batch_size; ++i) {
    infostate_fs.index_put_(
        {i, Slice(parview_sum[i].item<int>(), max_parviews), Slice()}, 0);
  }

  torch::Tensor bs = change_of_basis(infostate_fs);                             CHECK_SHAPE(bs, {batch_size, max_parviews, pooled_size()});
  torch::Tensor cs = base_coordinates(bs, beliefs);                             CHECK_SHAPE(cs, {batch_size, max_parviews, pooled_size()});
  torch::Tensor pooled = pool(cs);                                              CHECK_SHAPE(pooled, {batch_size, pooled_size()});
  torch::Tensor context = torch::cat({pooled, public_features}, /*dim=*/1);     CHECK_SHAPE(context, {batch_size, context_size()});
  torch::Tensor ys = regression(context)
      .expand({max_parviews, -1, -1}).permute({1, 0, 2});                       CHECK_SHAPE(ys, {batch_size, max_parviews, regression_size()});
  torch::Tensor proj = (ys * bs).sum(/*dim=*/2);                                CHECK_SHAPE(proj, {batch_size, max_parviews});
  SPIEL_DCHECK_TRUE(torch::isfinite(proj).all().item<bool>());

  if (zero_sum_regression) {
    // beliefs * values = 0 (because game is zero-sum) and vectors are
    // therefore perpendicular. If values are off and are not zero-sum,
    // we project them to the beliefs plane (beliefs are the normal vector).
    torch::Tensor batch_beliefs = beliefs.squeeze(/*dim=*/2);                   CHECK_SHAPE(batch_beliefs, {batch_size, max_parviews});
    torch::Tensor numer = (proj * batch_beliefs).sum(/*dim=*/1);                CHECK_SHAPE(numer, {batch_size});
    torch::Tensor denum = (batch_beliefs * batch_beliefs).sum(/*dim=*/1);       CHECK_SHAPE(denum, {batch_size});
    torch::Tensor proj_error = torch::zeros({batch_size}).to(device);
    proj_error.index_put_(
        {denum != 0},
        numer.index({denum != 0}).div(denum.index({denum != 0})));              CHECK_SHAPE(proj_error, {batch_size});

    torch::Tensor expanded_proj_error =
        proj_error.expand({max_parviews, -1}).permute({1, 0});                  CHECK_SHAPE(expanded_proj_error, {batch_size, max_parviews});
    proj = proj - expanded_proj_error * batch_beliefs;
  }

  // No weird values anywhere.
  SPIEL_DCHECK_TRUE(torch::isfinite(proj).all().item<bool>());
  return proj;
}

torch::Tensor ParticleValueNet::PrepareTarget(BatchData* batch) {
  return batch->target;
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
    bool zero_sum_regression,
    bool normalize_beliefs
) {
  SPIEL_CHECK_GE(num_layers_regression, 1);
  SPIEL_CHECK_GE(num_width_regression, 1);
  switch (arch) {
    case NetArchitecture::kParticle: {
      auto particle_dims = std::dynamic_pointer_cast<ParticleDims>(dims);
      auto model = std::make_shared<ParticleValueNet>(
          particle_dims,
          num_layers_regression, num_width_regression, num_inputs_regression,
          zero_sum_regression, normalize_beliefs,
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


