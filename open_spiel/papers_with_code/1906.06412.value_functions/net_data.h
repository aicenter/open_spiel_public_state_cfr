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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_

#include <iterator>

#include "torch/torch.h"

#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace papers_with_code {

using float_net = float;    // Floats used in the neural network.
using float_tree = double;  // Floats used in the cfr computation.

// Our base class.
// Data points are always a view to the underlying storage,
// placed within a batch of data.
struct DataPoint : torch::data::Example<>{
  using torch::data::Example<>::Example;
  void Reset();  // Zeros-out inputs and outputs.
};

// Dimensions for the current trunk.
struct PositionalDataDims {
  int public_features_size;
  std::array<int, 2> net_ranges_size;

  int point_input_size() const {
    return public_features_size + net_ranges_size[0] +  net_ranges_size[1];
  }
  int point_output_size() const {
    return net_ranges_size[0] +  net_ranges_size[1];
  }
  // Encoding of the input / output -- offsets:
  constexpr int public_features_offset() const { return 0; }
  int ranges_offset(Player pl) const {
    return public_features_size + (pl == 0 ? 0 : net_ranges_size[0]);
  }
  int values_offset(Player pl) const {
    return (pl == 0 ? 0 : net_ranges_size[0]);
  }
};

struct PositionalData final : DataPoint {
  // Views over the data point for easier manipulation.
  absl::Span<float_net> public_features;
  std::array<absl::Span<float_net>, 2> net_ranges;
  std::array<absl::Span<float_net>, 2> net_values;

  PositionalData(torch::Tensor data, torch::Tensor target,
                 const PositionalDataDims& dims);
  // Check if the data point is still a valid view:
  // no tensor pointers or spans are broken.
  bool is_valid_view() const;
};

struct ParticleDataDims {
  int public_features_size;
  int hand_features_size;
  int max_particles;

  int point_input_size() const {
    return 1  // Number of particles in the given point
         + max_particles * particle_size();
  }
  int particle_size() const {
    return public_features_size
         + hand_features_size
         + 2  // Player index features.
         + 1; // Range.
  }
  int point_output_size() const { return max_particles; }

  constexpr int num_particles_offset() const { return 0; }
  constexpr int public_features_offset() const { return 1; }
  int hand_features_offset() const { return public_features_size; }
  int player_offset() const { return public_features_size + hand_features_size; }
  int range_offset() const { return public_features_size + hand_features_size + 2; }
};

struct ParticleData final : DataPoint {
  // Views over the data point for easier manipulation.
  ParticleDataDims dims;
  ParticleData(torch::Tensor data, torch::Tensor target,
               const ParticleDataDims& dims);

  float_net& num_particles();
  absl::Span<float_net> particle_public_features(int particle);
  absl::Span<float_net> particle_hand_features(int particle);
  absl::Span<float_net> particle_player_features(int particle);
  float& particle_range(int particle);
  float& particle_value(int particle);

  float_net* data_ptr() { return data.data_ptr<float_net>(); }
  float_net* particle_data_ptr(int particle) {
    return 1 + &data.data_ptr<float_net>()[particle * dims.particle_size()];
  }
  float_net* target_ptr() { return target.data_ptr<float_net>(); }

  // Check if the data point is still a valid view:
  // no tensor pointers or spans are broken.
  bool is_valid_view() const;
};

struct BatchData {
  torch::Tensor data;
  torch::Tensor target;

  BatchData(int batch_size, int input_size, int output_size);
  void Reset();
  int size() const;

  // Views for individual data points.
  PositionalData point_at(int index, const PositionalDataDims& dims);
  ParticleData point_at(int index, const ParticleDataDims& dims);
};


}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_
