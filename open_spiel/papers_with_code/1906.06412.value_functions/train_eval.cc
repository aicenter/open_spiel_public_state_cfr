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

// TODO: generalize this! A quick hack.
class FilterPolicy : public Policy {
  Policy* source_policy_;
  const std::vector<std::string> root_infostates_{
    "[Observer: 0][Private: 0][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 0][Private: 1][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 0][Private: 2][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 0][Private: 3][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 0][Private: 4][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 0][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 1][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 2][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 3][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 4][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "[Observer: 1][Private: 5][Round 1][Player: 1][Pot: 2][Money: 99 99][Ante: 1 1]",
    "P0 hand: 1 2 3 4 \nP0 action sequence: \nPoint card sequence: 4 \nWin sequence: \nPoints: 0 0 \nTerminal?: 0\n",
    "P1 hand: 1 2 3 4 \nP1 action sequence: \nPoint card sequence: 4 \nWin sequence: \nPoints: 0 0 \nTerminal?: 0\n",
    "P0 hand: 1 2 3 4 5 \nP0 action sequence: \nPoint card sequence: 5 \nWin sequence: \nPoints: 0 0 \nTerminal?: 0\n",
    "P1 hand: 1 2 3 4 5 \nP1 action sequence: \nPoint card sequence: 5 \nWin sequence: \nPoints: 0 0 \nTerminal?: 0\n",
  };

 public:
  FilterPolicy(Policy* source_policy) : source_policy_(source_policy) {};

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    SPIEL_CHECK_TRUE(source_policy_);
    for (const std::string& root_infostate : root_infostates_) {
      if(root_infostate == info_state) {
        return source_policy_->GetStatePolicy(info_state);
      }
    }
    return {};
  }
};

std::vector<double> EvaluateNetwork(
    dlcfr::DepthLimitedCFR* trunk_with_net,
    ortools::SequenceFormLpSpecification* whole_game,
    const std::vector<int>& evaluate_iters,
    bool eval_only_root) {

  auto should_evaluate = [&](int i){
    for (auto j : evaluate_iters) {
      if (i == j) return true;
    }
    return false;
  };

  std::vector<double> expls;
  expls.reserve(evaluate_iters.size());

  trunk_with_net->Reset();
  int trunk_iters = *std::max_element(evaluate_iters.begin(),
                                      evaluate_iters.end());

  // Possibly specify a subset of the average policy, which should be fixed.
  std::shared_ptr<Policy> trunk_policy = trunk_with_net->AveragePolicy();
  FilterPolicy filter_policy(trunk_policy.get());
  Policy* eval_policy = trunk_policy.get();
  if (eval_only_root) {
    eval_policy = &filter_policy;
  }

  for (int i = 1; i <= trunk_iters; ++i) {
    ++trunk_with_net->num_iterations_;
    trunk_with_net->UpdateReachProbs();
    trunk_with_net->EvaluateLeaves();

    if (should_evaluate(i)) {
      expls.push_back(ortools::TrunkExploitability(whole_game, *eval_policy));
      std::cout << '.' << std::flush;
    }

    trunk_with_net->UpdateTrunk();
  }

  return expls;
}

}  // namespace papers_with_code
}  // namespace open_spiel

