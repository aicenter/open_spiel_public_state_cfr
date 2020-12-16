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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_

#include <algorithm>
#include <string>
#include <utility>
#include <memory>
#include <vector>

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"
#include "open_spiel/utils/format_observation.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;

template<class T>
struct BijectiveContainer {
  std::map<T, T> x2y;
  std::map<T, T> y2x;

  void put(std::pair<T, T> xy) {
    const T& x = xy.first;
    const T& y = xy.second;
    SPIEL_CHECK_TRUE(x2y.find(x) == x2y.end());
    SPIEL_CHECK_TRUE(y2x.find(y) == y2x.end());
    x2y[x] = y;
    y2x[y] = x;
  }
  const std::map<T, T>& tree_to_net() const { return x2y; }
  const std::map<T, T>& net_to_tree() const { return y2x; }

  size_t size() const {
    SPIEL_CHECK_EQ(x2y.size(), y2x.size());
    return x2y.size();
  }
};


struct RangeTable {
  // Bijection between ranges coming from infostate tree (x)
  // and the input position (y), called also hand, for each public state.
  // This is used for encoding NN inputs (resp. outputs).
  // Forward:  tree  -> input positions
  // Backward: output positions -> tree
  std::vector<BijectiveContainer<size_t>> bijections;

  // List all possible private observations ("hands") for each player.
  // Their vector indices represent the input position for each public state.
  std::vector<Observation> private_hands;

  RangeTable(int num_public_states) : bijections(num_public_states) {}
  int largest_range() const;
  size_t hand_index(const Observation& obs);
};

std::array<RangeTable, 2> CreateRangeTables(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves);

using float_net = float;   // Floats used in the neural network.
using float_cfr = double;  // Floats used in the cfr computation.

// Copy non-contiguous vectors using a permutation map.
// This also converts float <-> double as needed.
template<typename From, typename To>
void PlacementCopy(absl::Span<const From> from, absl::Span<To> to,
                   std::map<size_t, size_t> from_to) {
  for (const auto& [f, t] : from_to) {
    to[t] = from[f];
  }
}

struct BatchData {
  const size_t batch_size;
  const size_t input_size;
  const size_t output_size;
  const size_t public_features_size;
  const std::array<size_t, 2> ranges_size;

  std::vector<float> data;
  std::vector<float> targets;

  BatchData(const std::vector<dlcfr::LeafPublicState>& states,
            size_t input_size, size_t output_size,
            size_t public_features_size, std::array<size_t, 2> ranges_size)
      : batch_size(states.size()),
        input_size(input_size), output_size(output_size),
        public_features_size(public_features_size), ranges_size(ranges_size),
        // Pre-allocate all vectors.
        data(batch_size * input_size, 0.),
        targets(batch_size * output_size, 0.) {
    std::cout << "# Made BatchData with sizes:\n"
              << "#   batch_size=" << batch_size << "\n"
              << "#   input_size=" << input_size << "\n"
              << "#   output_size=" << output_size << "\n"
              << "#   public_features_size=" << public_features_size << "\n"
              << "#   ranges_size=" << ranges_size << "\n";
    std::cout << "# Public features:\n";
    for (int i = 0; i < states.size(); ++i) {
      CopyFeatures(i, states[i]);
      std::cout << "#   states[" << i << "].public_tensor\n#     "
                << ObservationToString(states[i].public_tensor, "\n#     ") << "\n";
    }
    std::cout << "# BatchData after feature copying:\n";
    std::cout << "#   " << data << "\n";
  }
  // Copy public state features
  void CopyFeatures(int batch_index, const dlcfr::LeafPublicState& state) {
    const auto tensor = state.public_tensor.Tensor();
    std::copy(tensor.begin(), tensor.end(),
              data.begin() + (batch_index * input_size));
  }
  // Zero-out ranges and values, keep the features.
  void Reset() {
    for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
      // Ranges are tricky: skip over the public features.
      size_t begin_offset = batch_index * input_size + public_features_size;
      size_t end_offset = (batch_index + 1) * input_size;
      std::fill(data.begin() + begin_offset,
                data.begin() + end_offset, 0.);
    }
    // Values are easy: put zeros everywhere.
    std::fill(targets.begin(), targets.end(), 0.);
  }

  torch::Tensor data_tensor() {
    return at::from_blob((void*) data.data(),
                         {batch_size, input_size}, at::kFloat);
  }
  torch::Tensor targets_tensor() {
    return at::from_blob((void*) targets.data(),
                         {batch_size, output_size}, at::kFloat);
  }
  torch::Tensor data_tensor_at(int batch_index) {
    return at::from_blob((void*) &data[batch_index * input_size],
                         {input_size}, at::kFloat);
  }
  absl::Span<float> data_at(int batch_index) {
    return absl::MakeSpan(&data[batch_index * input_size], input_size);
  }
  absl::Span<float> targets_at(int batch_index) {
    return absl::MakeSpan(&targets[batch_index * output_size], output_size);
  }
  absl::Span<float> ranges_at(int batch_index, Player pl) {
    const size_t offset = range_offset(pl);
    return absl::MakeSpan(
        &data[batch_index * input_size + public_features_size + offset],
        ranges_size[pl]);
  }
  absl::Span<float> values_at(int batch_index, Player pl) {
    const size_t offset = values_offset(pl);
    return absl::MakeSpan(&targets[batch_index * output_size + offset],
                          ranges_size[pl]);
  }
  size_t range_offset(Player pl) const {
    return pl == 1 ? ranges_size[0] : 0;
  }
  size_t values_offset(Player pl) const {
    return pl == 1 ? ranges_size[0] : 0;
  }
};

void RandomizeTrunkStrategy(std::vector<BanditVector>& bandits,
                            std::mt19937& rnd_gen, double prob_pure_strat);

void GenerateData(const std::array<RangeTable, 2>& tables,
                  dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                  std::mt19937& rnd_gen, bool verbose = false);

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_GENERATE_DATA_
