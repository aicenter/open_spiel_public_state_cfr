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

enum class NetArchitecture {
  kPositional,
  kParticle
};
NetArchitecture GetArchitecture(const std::string& arch);  // Enum from string.

// Our base class. Data points are always a view
// to the underlying storage, placed within a batch of data.
struct DataPoint : torch::data::Example<> {
  DataPoint(torch::Tensor data, torch::Tensor target)
      : torch::data::Example<>(data, target) {
    // Make sure the data point is just a view!
    SPIEL_DCHECK_TRUE(is_valid_view());
  }

  // Zeros-out inputs and outputs.
  void Reset();
  // Check if the data point is still a valid view:
  // no tensor pointers are broken.
  bool is_valid_view() const;
  float_net* data_ptr() { return data.data_ptr<float_net>(); }
  float_net* target_ptr() { return target.data_ptr<float_net>(); }
};

// Dimensions for the current trunk, as they depend
// on the game and depth being currently solved.
struct BasicDims {
  int public_features_size;
  int hand_features_size;
  const int player_features_size = 2;

  // I/O sizes so we know how to construct batch data.
  virtual int point_input_size() const = 0;
  virtual int point_output_size() const = 0;
  virtual ~BasicDims() = default;
};
// Deduce basic dimensions based on the game and observers of that game.
std::unique_ptr<BasicDims> DeduceBasicDims(
    NetArchitecture arch, const Game& game,
    const std::shared_ptr<Observer>& public_observer,
    const std::shared_ptr<Observer>& hand_observer);

struct ParticleDims final : public BasicDims {
  // Should be the same as max particles to handle a worst-case scenario.
  int max_parviews = 1000;

  int point_input_size() const override {
    return 1  // Store the number of parviews in the given point.
         + public_features_size  // Public features.
         + max_parviews * parview_size();  // Parview contents.
  }
  int point_output_size() const override { return max_parviews; }

  int features_size() const {
    return hand_features_size
         + player_features_size;
  }
  int full_features_size() const {
    return public_features_size
         + features_size();
  }
  int parview_size() const {
    return features_size()
         + 1;  // Range size.
  }
};


// Positional encoding.
//
// The data is arranged as:
// - public_features
// - ranges player 0
// - ranges player 1
//
// The target is arranged as:
// - values player 0
// - values player 1
//
// The ranges and values are positionally encoded according
// to HandTable::hand_index().
struct PositionalDims final : public BasicDims {
  std::array<int, 2> net_ranges_size;

  int point_input_size() const override {
    return public_features_size + net_ranges_size[0] +  net_ranges_size[1];
  }
  int point_output_size() const override {
    return net_ranges_size[0] +  net_ranges_size[1];
  }
};

// A single "particles view". We call these "parview" consistently in the code.
// These are different from particles: particles are histories, but parviews are
// a particle aggregation of player's beliefs over those particles, from the
// perspective of the player (i.e. its Action-PrivateObservation history).
//
// The data is arranged as:
// - public_features
// - hand_features
// - player_features
// - range
//
// The target is a single float: value.
struct ParviewDataPoint final : DataPoint {
  const ParticleDims& dims;
  ParviewDataPoint(const ParticleDims& particle_dims,
                   torch::Tensor data, torch::Tensor target);
  // Individual accessors.
  absl::Span<float_net> hand_features();
  absl::Span<float_net> player_features();
  float_net& range();
  float_net& value();
 private:
  // Offsets within the parview.
  int hand_features_offset() const {
    return 0;
  }
  int player_offset() const {
    return dims.hand_features_size;
  }
  int range_offset() const {
    return dims.hand_features_size
         + dims.player_features_size;
  }
};

// Parviews for one public state, collected into a single data point.
struct ParticleDataPoint final : DataPoint {
  const ParticleDims& dims;
  ParticleDataPoint(const ParticleDims& particle_dims,
                    torch::Tensor data, torch::Tensor target);
  absl::Span<float_net> public_features();
  float_net& num_parviews();
  ParviewDataPoint parview_at(int parview_index);
 private:
  // Offsets for number of parviews and the storage.
  int num_parviews_offset() const { return 0; }
  int public_features_offset() const { return 1; }
  int parviews_storage_offset() const { return dims.public_features_size + 1; }
};

struct PositionalDataPoint final : DataPoint {
  const PositionalDims& dims;
  PositionalDataPoint(const PositionalDims& positional_dims,
                      torch::Tensor data, torch::Tensor target);
  // Individual accessors.
  absl::Span<float_net> public_features();
  float_net& range_at(Player pl, int index);
  float_net& value_at(Player pl, int index);
 private:
  // Encoding of the input / output -- offsets:
  constexpr int public_features_offset() const { return 0; }
  int ranges_offset(Player pl) const {
    return dims.public_features_size + (pl == 0 ? 0 : dims.net_ranges_size[0]);
  }
  int values_offset(Player pl) const {
    return (pl == 0 ? 0 : dims.net_ranges_size[0]);
  }
};

struct BatchData {
  torch::Tensor data;
  torch::Tensor target;

  BatchData(int batch_size, int input_size, int output_size);
  void Reset();
  int size() const;

  // Views for individual data points.
  ParticleDataPoint point_at(int index, const ParticleDims& particle_dims);
  PositionalDataPoint point_at(int index, const PositionalDims& positional_dims);
};


}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_
