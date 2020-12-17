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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_

#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

class NetEvaluator final : public dlcfr::LeafEvaluator {
  ValueNet* model_;
  torch::Device* device_;
  std::shared_ptr<const Game> game_;
  std::shared_ptr<Observer> infostate_observer_;
  const std::array<RangeTable, 2>& tables_;
  BatchData* batch_;

 public:
  NetEvaluator(ValueNet* model, torch::Device* device,
               std::shared_ptr<const Game> game,
               std::shared_ptr<Observer> infostate_observer,
               const std::array<RangeTable, 2>& tables,
               BatchData* batch)
      : model_(model), device_(device), game_(std::move(game)),
        infostate_observer_(std::move(infostate_observer)),
        tables_(tables), batch_(batch) {}


  void EvaluatePublicState(dlcfr::LeafPublicState* state,
                           dlcfr::PublicStateContext* context) const override {
    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy<float_tree, float_net>(
          /*tree=*/ state->ranges[pl],
          /*net=*/  batch_->ranges_at(state->public_id, pl),
                    tables_[pl].bijections[state->public_id].tree_to_net());
    }

    torch::Tensor data = batch_->data_tensor_at(state->public_id).to(*device_);
    torch::Tensor output = model_->forward(data);

    auto raw_output = (float*) output.data_ptr();
    for (int pl = 0; pl < 2; ++pl) {
      const size_t player_offset = batch_->range_offset(pl);
      absl::Span<const float_net> net_values(&raw_output[player_offset],
                                             batch_->ranges_size[pl]);
      PlacementCopy<float_net, float_tree>(
          /*net= */ net_values,
          /*tree=*/ absl::MakeSpan(state->values[pl]),
                    tables_[pl].bijections[state->public_id].net_to_tree());
    }
  }
};

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
