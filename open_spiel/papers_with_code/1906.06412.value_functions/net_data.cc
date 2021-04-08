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

namespace open_spiel {
namespace papers_with_code {

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

float_net& ParticleDataPoint::num_parviews() {
  return data_ptr()[num_parviews_offset()];
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
