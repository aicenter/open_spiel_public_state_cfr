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

ParticleData ParticlesInContext::particle_at(const ParticleDims& dims,
                                             int particle_index) {
  SPIEL_CHECK_LT(particle_index, dims.max_particles);
  const int offset = particle_storage_offset();
  return ParticleData(
      dims,
      data.slice(
          /*dim=*/0,
          /*start=*/offset + dims.particle_size() * particle_index,
          /*end=*/offset + dims.particle_size() * (particle_index + 1),
          /*step=*/1),
      // Technically, this is just one float, but let's pass a slice,
      // so that it is a proper view with # of dimensions = 1
      // (and not 0 for scalar). This makes it easy to reuse the code.
      target.slice(/*dim=*/0, /*start=*/particle_index,
                   /*end=*/particle_index + 1, /*step=*/1));
}


absl::Span<float_net> ParticleData::public_features() {
  return absl::MakeSpan(&data_ptr()[public_features_offset()],
                        dims.public_features_size);
}
absl::Span<float_net> ParticleData::hand_features() {
  return absl::MakeSpan(&data_ptr()[hand_features_offset()],
                        dims.hand_features_size);
}
absl::Span<float_net> ParticleData::player_features() {
  return absl::MakeSpan(&data_ptr()[player_offset()],
                        dims.player_features_size);
}
float& ParticleData::range() {
  SPIEL_DCHECK_EQ(dims.range_size, 1);
  return data_ptr()[range_offset()];
}
float& ParticleData::value() {
  return target_ptr()[0];
}

ParticleData::ParticleData(const ParticleDims& particle_dims,
                           torch::Tensor data, torch::Tensor target)
    : DataPoint(data, target), dims(particle_dims) {
  SPIEL_DCHECK_EQ(data.size(0), dims.particle_size());
  SPIEL_DCHECK_EQ(target.size(0), 1);
}

ParticlesInContext::ParticlesInContext(torch::Tensor data, torch::Tensor target)
    : DataPoint(data, target) {}

float_net& ParticlesInContext::num_particles() {
  return data_ptr()[num_particles_offset()];
}

PositionalData::PositionalData(const PositionalDims& positional_dims,
                               torch::Tensor data, torch::Tensor target)
  : DataPoint(data, target), dims(positional_dims) {
  SPIEL_DCHECK_EQ(data.size(0), dims.point_input_size());
  SPIEL_DCHECK_EQ(target.size(0), dims.point_output_size());
}
absl::Span<float_net> PositionalData::public_features() {
  return absl::MakeSpan(&data_ptr()[public_features_offset()],
                        dims.public_features_size);
}
float_net& PositionalData::range_at(Player pl, int index) {
  return data_ptr()[ranges_offset(pl) + index];
}
float_net& PositionalData::value_at(Player pl, int index) {
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

ParticlesInContext BatchData::point_at(int index) {
  return ParticlesInContext(data[index], target[index]);
}

PositionalData BatchData::point_at(int index,
                                   const PositionalDims& positional_dims) {
  return PositionalData(positional_dims, data[index], target[index]);
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
