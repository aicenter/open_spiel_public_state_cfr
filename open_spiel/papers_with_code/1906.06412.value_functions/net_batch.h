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

#include "torch/torch.h"

#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace papers_with_code {

struct BatchData {
  using LeafPublicState = algorithms::dlcfr::LeafPublicState;

  const size_t batch_size;
  const size_t input_size;
  const size_t output_size;
  const size_t public_features_size;
  const std::array<size_t, 2> ranges_size;

  std::vector<float> data;
  std::vector<float> targets;

  BatchData(const std::vector<LeafPublicState>& states,
            size_t input_size, size_t output_size,
            size_t public_features_size, std::array<size_t, 2> ranges_size)
      : batch_size(states.size()),
        input_size(input_size), output_size(output_size),
        public_features_size(public_features_size), ranges_size(ranges_size),
      // Pre-allocate all vectors.
        data(batch_size * input_size, 0.),
        targets(batch_size * output_size, 0.) {
    for (size_t i = 0; i < states.size(); ++i) {
      CopyFeatures(i, states[i]);
    }
  }

  // Copy public state features
  void CopyFeatures(int batch_index, const LeafPublicState& state) {
    const auto tensor = state.public_tensor.Tensor();
    std::copy(tensor.begin(), tensor.end(),
              data.begin() + (batch_index * input_size));
  }
  // Zero-out ranges and values, keep the features.
  void Reset() {
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
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

void DebugPrintBatchData(const BatchData& batch);

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_BATCH_
