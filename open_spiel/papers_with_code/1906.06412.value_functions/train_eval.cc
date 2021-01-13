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

torch::Tensor TrainNetwork(ValueNet* model, torch::Device* device,
                           torch::optim::Optimizer* optimizer,
                           BatchData* batch) {
  torch::Tensor data = batch->data.to(*device);
  torch::Tensor target = batch->target.to(*device);
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  torch::Tensor loss = torch::mse_loss(output, target);
  SPIEL_CHECK_FALSE(std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer->step();
  return loss;
}

std::vector<double> EvaluateNetwork(
    dlcfr::DepthLimitedCFR* trunk_with_net,
    ortools::SequenceFormLpSpecification* whole_game,
    std::vector<int> evaluate_iters) {

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

  for (int i = 1; i <= trunk_iters; ++i) {
    ++trunk_with_net->num_iterations_;
    trunk_with_net->UpdateReachProbs();
    trunk_with_net->EvaluateLeaves();

    if (should_evaluate(i)) {
      expls.push_back(ortools::TrunkExploitability(whole_game,
                                          *trunk_with_net->AveragePolicy()));
      std::cout << '.' << std::flush;
    }

    trunk_with_net->UpdateTrunk();
  }

  return expls;
}

}  // namespace papers_with_code
}  // namespace open_spiel

