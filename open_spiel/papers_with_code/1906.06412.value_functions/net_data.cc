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

ParticlesInContext::ParticlesInContext(const ParticleDims& particle_dims,
                                       torch::Tensor data, torch::Tensor target)
    : DataPoint(data, target), dims(particle_dims) {
  // Make sure the data point is just a view!
  SPIEL_DCHECK_TRUE(is_valid_view());
}

float_net& ParticlesInContext::num_particles() {
  return data_ptr()[num_particles_offset()];
}

absl::Span<float_net> ParticlesInContext::public_features(int particle) {
  SPIEL_DCHECK_LT(particle, dims.max_particles);
  return absl::MakeSpan(
      &particle_data_ptr(particle)[public_features_offset()],
      dims.public_features_size);
}

absl::Span<float_net> ParticlesInContext::hand_features(int particle) {
  SPIEL_DCHECK_LT(particle, dims.max_particles);
  return absl::MakeSpan(
      &particle_data_ptr(particle)[hand_features_offset()],
      dims.hand_features_size);
}

absl::Span<float_net> ParticlesInContext::player_features(int particle) {
  SPIEL_DCHECK_LT(particle, dims.max_particles);
  return absl::MakeSpan(
      &particle_data_ptr(particle)[player_offset()],
      dims.player_features_size);
}

float& ParticlesInContext::range(int particle) {
  SPIEL_DCHECK_LT(particle, dims.max_particles);
  SPIEL_DCHECK_EQ(dims.range_size, 1);
  return particle_data_ptr(particle)[range_offset()];
}

float& ParticlesInContext::value(int particle) {
  SPIEL_DCHECK_LT(particle, dims.max_particles);
  return target_ptr()[particle];
}

BatchData::BatchData(int batch_size, int input_size, int output_size)
    : // Pre-allocate all tensors.
      data(torch::empty({batch_size, input_size}, kDataPointOptions)),
      target(torch::empty({batch_size, output_size}, kDataPointOptions)) {
  SPIEL_CHECK_GT(batch_size, 0);
  SPIEL_CHECK_GT(input_size, 0);
  SPIEL_CHECK_GT(output_size, 0);
}

ParticlesInContext BatchData::point_at(const ParticleDims& dims, int index) {
  return ParticlesInContext(dims, data[index], target[index]);
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
