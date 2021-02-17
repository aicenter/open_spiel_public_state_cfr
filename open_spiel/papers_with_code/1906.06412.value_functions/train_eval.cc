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

#include "open_spiel/papers_with_code/1906.06412.value_functions/train_eval.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/dispatch_policy.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;
using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

double TrainNetwork(ParticleValueNet* model, torch::Device* device,
                           torch::optim::Optimizer* optimizer,
                           BatchData* batch) {
  SPIEL_DCHECK_TRUE(model->is_training());
  torch::Tensor data = batch->data.to(*device);
  torch::Tensor target = model->PrepareTarget(batch).to(*device);
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  torch::Tensor loss = torch::mse_loss(output, target);
  SPIEL_CHECK_FALSE(std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer->step();
  return loss.item().to<double>();
}


class UniformISTreePolicy : public Policy {
  const std::vector<std::shared_ptr<InfostateTree>>& trees_;
 public:
  UniformISTreePolicy(const std::vector<std::shared_ptr<InfostateTree>>& trees)
    : trees_(trees) {}
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    for (int pl = 0; pl < 2; ++pl) {
      const InfostateNode* node =
          trees_[pl]->DecisionNodeFromInfostateString(info_state);
      if (node) {
        const std::vector<Action>& actions = node->legal_actions();
        const double p = 1. / actions.size();
        ActionsAndProbs ap;
        ap.reserve(actions.size());
        for (int i = 0; i < actions.size(); ++i) {
          ap.push_back({actions[i], p});
        }
        return ap;
      }
    }
    return {};
  }
};


std::vector<double> EvaluateNetwork(
    std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks_with_net,
    ortools::SequenceFormLpSpecification* whole_game,
    const std::vector<int>& evaluate_iters) {

  std::vector<double> expls;
  expls.reserve(evaluate_iters.size());

  auto uniform_policy = std::make_shared<UniformISTreePolicy>(
      whole_game->trees());
  DispatchPolicy eval_policy;
  for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
    // Important!! We must reset all the bandits & other memory for proper eval.
    eval_policy.AddDispatch(sparse_trunk->uniform_infostates,
                            uniform_policy);
  }

  expls.push_back(ortools::TrunkExploitability(whole_game, eval_policy));
  return expls;
}

}  // namespace papers_with_code
}  // namespace open_spiel

