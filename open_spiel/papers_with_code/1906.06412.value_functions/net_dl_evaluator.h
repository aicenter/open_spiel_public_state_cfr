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

#include "open_spiel/papers_with_code/1906.06412.value_functions/hand_table.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

using PublicStateContext = algorithms::dlcfr::PublicStateContext;
using LeafPublicState = algorithms::dlcfr::LeafPublicState;
using LeafEvaluator = algorithms::dlcfr::LeafEvaluator;

// A bijection within the scope of a public state. This is a mapping between
// LeafPublicState::ranges coming from the tree (x) and the input position
// of the neural network (y) which is assigned according to player's private
// hands across all public states of the trunk.
// This is used for encoding NN inputs (resp. outputs).
struct HandMapping : BijectiveContainer<size_t> {
  const std::map<size_t, size_t>& tree_to_net() const { return x2y; }
  const std::map<size_t, size_t>& net_to_tree() const { return y2x; }
};

struct NetContext : public PublicStateContext {
  HandInfo* hand_info_;
  // Hand mapping for each player within a public state:
  // a bijection between the network and tree positions.
  std::vector<HandMapping> hand_mapping;
  NetContext(HandInfo* hand_info) : hand_info_(hand_info), hand_mapping(2) {}

  int net_index(Player pl, int tree_index) const {
    return hand_mapping[pl].tree_to_net().at(tree_index);
  }
  int tree_index(Player pl, int net_index) const {
    return hand_mapping[pl].net_to_tree().at(net_index);
  }
  const Observation& hand_at(Player pl, int infostate_id) const {
    return hand_info_->tables[pl].private_hands.at(net_index(pl, infostate_id));
  }
};

class NetEvaluator : public LeafEvaluator {
  HandInfo* hand_info_;
 public:
  NetEvaluator(HandInfo* hand_info) : hand_info_(hand_info) {};
  std::unique_ptr<PublicStateContext> CreateContext(
      const LeafPublicState& leaf_state) const override;
};

class ParticleNetEvaluator final : public NetEvaluator {
  ParticleValueNet* model_;
  torch::Device* device_;
  BatchData* batch_;
  ParticleDims* const dims_;
 public:
  ParticleNetEvaluator(HandInfo* hand_info,
                       ParticleValueNet* model, ParticleDims* const dims,
                       BatchData* batch, torch::Device* device)
      : NetEvaluator(hand_info),
        model_(model), device_(device), batch_(batch), dims_(dims) {}

  void EvaluatePublicState(LeafPublicState* state,
                           PublicStateContext* context) const override;
};

class PositionalNetEvaluator final : public NetEvaluator {
  PositionalValueNet* model_;
  torch::Device* device_;
  BatchData* batch_;
  PositionalDims* const dims_;
 public:
  PositionalNetEvaluator(HandInfo* hand_info,
                         PositionalValueNet* model, PositionalDims* const dims,
                         BatchData* batch, torch::Device* device)
      : NetEvaluator(hand_info),
        model_(model), device_(device), batch_(batch), dims_(dims) {}

  void EvaluatePublicState(LeafPublicState* state,
                           PublicStateContext* context) const override;
};

std::shared_ptr<NetEvaluator> MakeNetEvaluator(
    BasicDims* dims, HandInfo* hand_info, ValueNet* model,
    BatchData* eval_batch, torch::Device* device);

}  // namespace papers_with_code
}  // namespace open_spiel


#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_NET_EVALUATOR_
