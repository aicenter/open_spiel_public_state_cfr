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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TORCH_UTILS_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TORCH_UTILS_

#include <iterator>
#include <fenv.h>

#include "torch/torch.h"


#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"

namespace open_spiel {
namespace papers_with_code {

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

#define _ -1  // A shape placeholder.
#ifndef NDEBUG
inline void CHECK_SHAPE(const torch::Tensor& tensor,
                 std::initializer_list<int64_t> shape) {
  const std::vector<int64_t> expected_shape(shape);
  SPIEL_DCHECK_EQ(tensor.dim(), expected_shape.size());
  for (int i = 0; i < expected_shape.size(); i++) {
    if (expected_shape[i] == _) continue;
    if (tensor.sizes().at(i) != expected_shape[i]) {
      std::string actual_str = absl::StrJoin(tensor.sizes().vec(), ",");
      std::string expected_str = absl::StrJoin(expected_shape, ",");
      SpielFatalError(absl::StrCat(
          "CHECK_SHAPE: ",
          tensor.sizes().at(i), " != ", expected_shape[i], " at index ", i,
          " -- full shapes: actual ", actual_str, " expected ", expected_str));
    }
  }
}
#else
inline void CHECK_SHAPE(const torch::Tensor& tensor,
                        std::initializer_list<int64_t> shape) {}
#endif


#define WITH_FLOAT_ERRORS_DISABLED(x) \
  fedisableexcept(FE_INVALID);        \
  x                                   \
  feenableexcept(FE_INVALID);

} // namespace open_spiel
} // namespace papers_with_code

#endif // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TORCH_UTILS_