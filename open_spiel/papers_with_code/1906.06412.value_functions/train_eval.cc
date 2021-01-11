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
  torch::Tensor data = batch->data_tensor().to(*device);
  torch::Tensor targets = batch->targets_tensor().to(*device);
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  torch::Tensor loss = torch::mse_loss(output, targets);
  SPIEL_CHECK_FALSE(std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer->step();
  return loss;
}

double EvaluateNetwork(dlcfr::DepthLimitedCFR* trunk_with_net,
                       int trunk_iterations,
                       ortools::SequenceFormLpSpecification* whole_game) {
  trunk_with_net->Reset();
  for (int i = 1; i <= trunk_iterations - 1; ++i) {
    ++trunk_with_net->num_iterations_;
    trunk_with_net->UpdateReachProbs();
    trunk_with_net->EvaluateLeaves();
    double expl = ortools::TrunkExploitability(
        whole_game, *trunk_with_net->AveragePolicy());

    trunk_with_net->UpdateTrunk();
    std::cout << "# i = " << i << " expl=" << expl << std::endl;
  }

  ++trunk_with_net->num_iterations_;
  trunk_with_net->UpdateReachProbs();
  trunk_with_net->EvaluateLeaves();

  double expl = ortools::TrunkExploitability(
      whole_game, *trunk_with_net->AveragePolicy());
  std::cout << "# i = " << trunk_iterations << " expl=" << expl << std::endl;
  return expl;
}

}  // namespace papers_with_code
}  // namespace open_spiel

