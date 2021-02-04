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

class TabularizePolicy : public Policy {
  std::map<std::string, std::shared_ptr<Policy>> dispatch_table_;
 public:
  TabularizePolicy(std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks) {
    for (std::unique_ptr<SparseTrunk>& sparse_trunk : sparse_trunks) {
      for (const std::string & eval_infostate : sparse_trunk->eval_infostates) {
        SPIEL_CHECK_TRUE(dispatch_table_.find(eval_infostate)
                         == dispatch_table_.end());
        dispatch_table_[eval_infostate] = sparse_trunk->dlcfr->AveragePolicy();
      }
    }
  };

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    auto it = dispatch_table_.find(info_state);
    if (it == dispatch_table_.end()) {
      return {};
    } else {
      return it->second->GetStatePolicy(info_state);
    }
  }
};

std::vector<double> EvaluateNetwork(
    std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks_with_net,
    ortools::SequenceFormLpSpecification* whole_game,
    const std::vector<int>& evaluate_iters) {

  auto should_evaluate = [&](int i){
    for (auto j : evaluate_iters) {
      if (i == j) return true;
    }
    return false;
  };

  std::vector<double> expls;
  expls.reserve(evaluate_iters.size());
  int trunk_iters = *std::max_element(evaluate_iters.begin(),
                                      evaluate_iters.end());
  TabularizePolicy eval_policy(sparse_trunks_with_net);

  // Important!! We must reset all the bandits & other memory for proper eval.
  for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
    sparse_trunk->dlcfr->Reset();
  }

  for (int i = 1; i <= trunk_iters; ++i) {
    for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
      dlcfr::DepthLimitedCFR* trunk_with_net = sparse_trunk->dlcfr.get();
      ++trunk_with_net->num_iterations_;
      trunk_with_net->UpdateReachProbs();
      trunk_with_net->EvaluateLeaves();
    }

    if (should_evaluate(i)) {
      expls.push_back(ortools::TrunkExploitability(whole_game, eval_policy));
      std::cout << '.' << std::flush;
    }

    for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
      sparse_trunk->dlcfr->UpdateTrunk();
    }
  }

  return expls;
}

}  // namespace papers_with_code
}  // namespace open_spiel

