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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_

#include <algorithm>
#include <string>
#include <utility>
#include <memory>
#include <vector>

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"

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
  const std::map<T, T>& forward() const { return x2y; }
  const std::map<T, T>& backward() const { return y2x; }
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
  int hand_index(const Observation& obs);
};

std::array<RangeTable, 2> CreateRangeTables(
    const Game& game,
    const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves);

using net_float = float;   // Floats used in the neural network.
using cfr_float = double;  // Floats used in the cfr computation.

// Copy non-contiguous vectors using a permutation map.
// This also converts float <-> double as needed.
template<typename From, typename To>
void PlacementCopy(absl::Span<const From> from, absl::Span<To> to,
                   std::map<size_t, size_t> from_to) {
  SPIEL_CHECK_EQ(from.size(), from_to.size());
  for (size_t i = 0; i < from.size(); ++i) {
    const int j = from_to[i];
    to[j] = from[i];
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
    for (int i = 0; i < states.size(); ++i) {
      CopyFeatures(i, states[i]);
    }
  }
  // Copy public state features
  void CopyFeatures(int batch_index, const dlcfr::LeafPublicState& state) {
    std::copy(state.public_tensor.begin(),
              state.public_tensor.end(),
              data.begin() + batch_index * input_size);
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
    const size_t offset = pl == 1 ? ranges_size[0] : 0;
    return absl::MakeSpan(
        &data[batch_index * input_size + public_features_size + offset],
        ranges_size[pl]);
  }
  absl::Span<float> values_at(int batch_index, Player pl) {
    const size_t offset = pl == 1 ? ranges_size[0] : 0;
    return absl::MakeSpan(&data[batch_index * output_size + offset],
                          ranges_size[pl]);
  }
};

void GenerateData(const std::array<RangeTable, 2>& tables,
                  dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                  std::mt19937 rnd_gen);

}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_
