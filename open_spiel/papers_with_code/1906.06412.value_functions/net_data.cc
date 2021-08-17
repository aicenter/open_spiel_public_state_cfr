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

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"

namespace open_spiel {
namespace papers_with_code {

std::shared_ptr<BasicDims> DeduceBasicDims(
    NetArchitecture arch,
    const Game& game,
    const std::shared_ptr<Observer>& public_observer,
    const std::shared_ptr<Observer>& hand_observer
) {

  std::shared_ptr<BasicDims> dims;
  switch (arch) {
    case NetArchitecture::kParticle:
      dims = std::make_shared<ParticleDims>();
      break;
    case NetArchitecture::kPositional:
      dims = std::make_shared<PositionalDims>();
      break;
  }

  Observation public_observation(game, public_observer);
  Observation hand_observation(game, hand_observer);
  dims->public_features_size = public_observation.Tensor().size();
  dims->hand_features_size = hand_observation.Tensor().size();
  return dims;
}

const torch::TensorOptions kDataPointOptions = torch::TensorOptions()
    .dtype<float_net>();

void DataPoint::Reset() {
  data.zero_();
  target.zero_();
}

bool DataPoint::is_valid_view() const {
  return data.is_view()
      && target.is_view()
      && data.is_contiguous()
      && target.is_contiguous()
      && data.dim() == 1
      && target.dim() == 1;
}

ParviewDataPoint ParticleDataPoint::parview_at(int parview_index) {
  SPIEL_CHECK_LT(parview_index, dims.max_parviews);
  const int offset = parviews_storage_offset();
  return ParviewDataPoint(
      dims,
      data.slice(
          /*dim=*/0,
          /*start=*/offset + dims.parview_size() * parview_index,
          /*end=*/offset + dims.parview_size() * (parview_index + 1),
          /*step=*/1),
      // Technically, this is just one float, but let's pass a slice,
      // so that it is a proper view with # of dimensions = 1
      // (and not 0 for scalar). This makes it easy to reuse the code.
      target.slice(/*dim=*/0, /*start=*/parview_index,
                   /*end=*/parview_index + 1, /*step=*/1));
}

std::array<torch::Tensor, 2> ParticleDataPoint::beliefs() {
  const int max_parviews = dims.max_parviews;

  torch::Tensor parviews = data.index({
    Slice(parviews_storage_offset(),
          parviews_storage_offset() + max_parviews * dims.parview_size())
    // Rearrange into parviews.
  }).view({max_parviews, dims.parview_size()});                                 CHECK_SHAPE(parviews, {max_parviews, dims.parview_size()});

  torch::Tensor beliefs = parviews  // Skip all features.
      .index({Slice(), Slice(dims.features_size(), dims.parview_size())});      CHECK_SHAPE(beliefs, {max_parviews, 1});

  return {
    beliefs.index({Slice(0, num_parviews(0))}),
    beliefs.index({Slice(num_parviews(0), total_parviews())}),
  };
}
std::array<torch::Tensor, 2> ParticleDataPoint::values() {
  const int max_parviews = dims.max_parviews;

  torch::Tensor parviews = data.index({
    Slice(parviews_storage_offset(),
          parviews_storage_offset() + max_parviews * dims.parview_size())
    // Rearrange into parviews.
  }).view({max_parviews, dims.parview_size()});                                 CHECK_SHAPE(parviews, {max_parviews, dims.parview_size()});

  return {
    target.index({Slice(0, num_parviews(0))}),
    target.index({Slice(num_parviews(0), total_parviews())}),
  };
}

std::array<float, 2> ParticleDataPoint::NormalizeBeliefsAndValues() {
  std::array<torch::Tensor, 2> player_beliefs = beliefs();
  std::array<torch::Tensor, 2> player_values = values();
  std::array<float, 2> belief_normalizers = {player_beliefs[0].sum().item<float>(),
                                             player_beliefs[1].sum().item<float>()};

  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_TRUE(std::isfinite(belief_normalizers[pl]));

    WITH_FLOAT_ERRORS_DISABLED({
      // In-place modifications!
      player_beliefs[pl].div_(belief_normalizers[pl]).nan_to_num_(
          /*nan=*/0., /*posinf=*/0., /*neginf=*/0.);
      // Important: Opposite beliefs for values !!!
      player_values[pl].div_(belief_normalizers[1 - pl]).nan_to_num_(
          /*nan=*/0., /*posinf=*/0., /*neginf=*/0.);
    })
    SPIEL_CHECK_TRUE(torch::isfinite(player_beliefs[pl]).all().item<bool>());
    SPIEL_CHECK_TRUE(torch::isfinite(player_values[pl]).all().item<bool>());
  }
  return belief_normalizers;
}

void ParticleDataPoint::DenormalizeValues(
    const std::array<float, 2>& belief_normalizers) {
  std::array<torch::Tensor, 2> player_values = values();
  // In-place modification!
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_TRUE(std::isfinite(belief_normalizers[pl]));

    // Important: Opposite beliefs for values !!!
    player_values[pl].mul_(belief_normalizers[1 - pl]);
    SPIEL_CHECK_TRUE(torch::isfinite(player_values[pl]).all().item<bool>());
  }
}

absl::Span<float_net> ParviewDataPoint::hand_features() {
  return absl::MakeSpan(&data_ptr()[hand_features_offset()],
                        dims.hand_features_size);
}
absl::Span<float_net> ParviewDataPoint::player_features() {
  return absl::MakeSpan(&data_ptr()[player_offset()],
                        dims.player_features_size);
}
float_net& ParviewDataPoint::range() {
  return data_ptr()[range_offset()];
}
float_net& ParviewDataPoint::value() {
  return target_ptr()[0];
}

ParviewDataPoint::ParviewDataPoint(const ParticleDims& particle_dims,
                                   torch::Tensor data, torch::Tensor target)
    : DataPoint(data, target), dims(particle_dims) {
  SPIEL_DCHECK_EQ(data.size(0), dims.parview_size());
  SPIEL_DCHECK_EQ(target.size(0), 1);
}

ParticleDataPoint::ParticleDataPoint(const ParticleDims& particle_dims,
                                     torch::Tensor data, torch::Tensor target)
    : DataPoint(data, target), dims(particle_dims) {}

float_net& ParticleDataPoint::num_parviews(Player pl) {
  return data_ptr()[num_parviews_offset() + pl];
}

int ParticleDataPoint::total_parviews() {
  return num_parviews(0) + num_parviews(1);
}

absl::Span<float_net> ParticleDataPoint::public_features() {
  return absl::MakeSpan(&data_ptr()[public_features_offset()],
                        dims.public_features_size);
}

PositionalDataPoint::PositionalDataPoint(const PositionalDims& positional_dims,
                                         torch::Tensor data, torch::Tensor target)
  : DataPoint(data, target), dims(positional_dims) {
  SPIEL_DCHECK_EQ(data.size(0), dims.point_input_size());
  SPIEL_DCHECK_EQ(target.size(0), dims.point_output_size());
}
absl::Span<float_net> PositionalDataPoint::public_features() {
  return absl::MakeSpan(&data_ptr()[public_features_offset()],
                        dims.public_features_size);
}
float_net& PositionalDataPoint::range_at(Player pl, int index) {
  return data_ptr()[ranges_offset(pl) + index];
}
float_net& PositionalDataPoint::value_at(Player pl, int index) {
  return target_ptr()[values_offset(pl) + index];
}

BatchData::BatchData(int batch_size, int input_size, int output_size)
    : // Pre-allocate all tensors.
      data(torch::empty({batch_size, input_size}, kDataPointOptions)),
      target(torch::empty({batch_size, output_size}, kDataPointOptions)) {
  SPIEL_CHECK_GT(batch_size, 0);
  SPIEL_CHECK_GT(input_size, 0);
  SPIEL_CHECK_GT(output_size, 0);
}

ParticleDataPoint BatchData::point_at(int index,
                                      const ParticleDims& particle_dims) {
  return ParticleDataPoint(particle_dims, data[index], target[index]);
}

PositionalDataPoint BatchData::point_at(int index,
                                        const PositionalDims& positional_dims) {
  return PositionalDataPoint(positional_dims, data[index], target[index]);
}

void BatchData::Reset() {
  data.zero_();
  target.zero_();
}

int BatchData::size() const {
  SPIEL_CHECK_EQ(data.size(0), target.size(0));
  return target.size(0);
}

}  // namespace papers_with_code
}  // namespace open_spiel
