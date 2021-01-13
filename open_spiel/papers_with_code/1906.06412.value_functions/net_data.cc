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

PositionalData::PositionalData(torch::Tensor data, torch::Tensor target,
                               const PositionalDataDims& dims) :
  DataPoint(data, target) {
  SPIEL_CHECK_EQ(data.size(0), dims.net_input_size());
  SPIEL_CHECK_EQ(target.size(0), dims.net_output_size());

  float_net* data_ptr = data.data_ptr<float_net>();
  float_net* target_ptr = target.data_ptr<float_net>();

  public_features = absl::MakeSpan(&data_ptr[dims.public_features_offset()],
                                   dims.public_features_size);
  for (int pl = 0; pl < 2; ++pl) {
    net_ranges[pl] = absl::MakeSpan(&data_ptr[dims.ranges_offset(pl)],
                                    dims.net_ranges_size[pl]);
    net_values[pl] = absl::MakeSpan(&target_ptr[dims.values_offset(pl)],
                                    dims.net_ranges_size[pl]);
  }
}

void PositionalData::Reset() {
  data.zero_();
  target.zero_();
}

BatchData::BatchData(int batch_size, int input_size, int output_size)
    : // Pre-allocate all tensors.
      data(torch::empty({batch_size, input_size}, kDataPointOptions)),
      target(torch::empty({batch_size, output_size}, kDataPointOptions)) {
  SPIEL_CHECK_GT(batch_size, 0);
  SPIEL_CHECK_GT(input_size, 0);
  SPIEL_CHECK_GT(output_size, 0);
}

PositionalData BatchData::point_at(int index, const PositionalDataDims& dims) {
  return PositionalData(data[index], target[index], dims);
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
