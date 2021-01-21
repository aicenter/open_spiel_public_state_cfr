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

// Our base class. Data points are always a view
// to the underlying storage, placed within a batch of data.
struct DataPoint : torch::data::Example<>{
  using torch::data::Example<>::Example;
  // Zeros-out inputs and outputs.
  void Reset();
  // Check if the data point is still a valid view:
  // no tensor pointers are broken.
  bool is_valid_view() const;
};

// Dimensions for the current trunk, as they depend
// on the game and depth being currently solved.
struct BasicDims {
  int public_features_size;
  int hand_features_size;
  std::array<int, 2> net_ranges_size;
  const int player_features_size = 2;

  // I/O sizes so we know how to construct batch data.
  virtual int point_input_size() const = 0;
  virtual int point_output_size() const = 0;
};

struct ParticleDims final : public BasicDims {
  int max_particles = 1000;
  const int range_size = 1;

  int point_input_size() const override {
    return 1  // Store the number of particles in the given point.
         + max_particles * particle_size();  // Particle contents.
  }
  int point_output_size() const override { return max_particles; }

  int particle_size() const {
    return public_features_size
         + hand_features_size
         + player_features_size
         + range_size;
  }
};

// Particles for one public state, collected into a single data point.
struct ParticlesInContext final : DataPoint {
  const ParticleDims& dims;
  ParticlesInContext(const ParticleDims& dims,
                     torch::Tensor data, torch::Tensor target);

  // Particle accessors.
  float_net& num_particles();
  absl::Span<float_net> public_features(int particle);
  absl::Span<float_net> hand_features(int particle);
  absl::Span<float_net> player_features(int particle);
  float& range(int particle);
  float& value(int particle);

 private:
  float_net* data_ptr() { return data.data_ptr<float_net>(); }
  float_net* target_ptr() { return target.data_ptr<float_net>(); }
  // Contents of the individual particle.
  float_net* particle_data_ptr(int particle) {
    return &data_ptr()[
      particle_storage_offset() + particle * dims.particle_size()
    ];
  }
  // Offsets for number of particles and the storage.
  int num_particles_offset() const { return 0; }
  int particle_storage_offset() const { return 1; }

  // Offsets within the particle.
  int public_features_offset() const {
    return 0;
  }
  int hand_features_offset() const {
    return dims.public_features_size;
  }
  int player_offset() const {
    return dims.public_features_size
         + dims.hand_features_size;
  }
  int range_offset() const {
    return dims.public_features_size
         + dims.hand_features_size
         + dims.player_features_size;
  }
};

struct BatchData {
  torch::Tensor data;
  torch::Tensor target;

  BatchData(int batch_size, int input_size, int output_size);
  void Reset();
  int size() const;

  // Views for individual data points.
  ParticlesInContext point_at(const ParticleDims& dims, int index);
};


}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_
