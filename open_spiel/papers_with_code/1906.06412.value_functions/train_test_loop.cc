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


#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");
ABSL_FLAG(int, depth, 3, "Max depth of the trunk.");

#include <random>

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

constexpr size_t kSeed = 0;
constexpr char* kUseBanditsForCfr = "PredictiveRegretMatchingPlus";

torch::Tensor TrainNetwork(ValueNet* model, torch::Device* device,
                           torch::optim::Optimizer* optimizer,
                           BatchData* batch) {
  torch::Tensor data = batch->data_tensor().to(*device);
  torch::Tensor targets = batch->targets_tensor().to(*device);
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  torch::Tensor loss = torch::mse_loss(output, targets);
  AT_ASSERT(!std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer->step();
  return loss;
}

double EvaluateNetwork(dlcfr::DepthLimitedCFR* trunk_with_net, int iterations,
                       ortools::SequenceFormLpSpecification* whole_game) {
  trunk_with_net->Reset();
  std::cout << " (trunk iters) " << std::flush;
  trunk_with_net->RunSimultaneousIterations(iterations);
  std::cout << " (trunk expl) " << std::endl;
  return ortools::TrunkExploitability(
      whole_game, *trunk_with_net->AveragePolicy());
}

void TrainEvalLoop(std::unique_ptr<Trunk> t, int train_batches, int num_loops,
                   int cfr_oracle_iterations, int trunk_eval_iterations,
                   bool verbose_every_loop) {
  PrintRangeTables(t->tables);
  PrintBatchData(*t->batch, t->trunk_with_oracle->public_leaves());

  t->oracle_evaluator->num_cfr_iterations = cfr_oracle_iterations;
  torch::manual_seed(kSeed);
  std::mt19937 rnd_gen(kSeed);

  // 1. Create network and optimizer.
  torch::Device device = FindDevice();
  PositionalValueNet model(t->batch->input_size,
                           t->batch->output_size,
                           /*hidden_size=*/t->batch->input_size * 3);
  model.to(device);
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(/*lr=*/0.01));

  // 2. Create trunk net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, t->game, t->infostate_observer,
      t->tables, t->batch.get());
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      t->game, t->trunk_trees, net_evaluator, t->terminal_evaluator,
      t->public_observer, MakeBanditVectors(t->trunk_trees, kUseBanditsForCfr));

  // 3. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*t->game, "CLP");

  // 4. The train-eval loop.
  std::cout << "loop,avg_loss,exploitability" << std::endl;
  for (int loop = 0; loop < num_loops; ++loop) {
    // Train.
    double cumul_loss = 0.;
    std::cout << "# Training  ";
    for (int i = 0; i < train_batches; ++i) {
      GenerateData(t->tables, t->trunk_with_oracle.get(), t->batch.get(), rnd_gen,
          /*verbose=*/(i == 0 && verbose_every_loop) || (i == 0 && loop == 0));
      torch::Tensor loss = TrainNetwork(&model, &device,
                                        &optimizer, t->batch.get());
      cumul_loss += loss.item().to<double>();
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    // Eval.
    std::cout << "# Evaluating  " << std::flush;
    const double exploitability = EvaluateNetwork(
        trunk_with_net.get(), trunk_eval_iterations, &whole_game);
    const double avg_loss = cumul_loss / train_batches;

    // Print.
    std::cout << loop << ',' << avg_loss << ',' << exploitability << std::endl;
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::papers_with_code;

  absl::ParseCommandLine(argc, argv);
  TrainEvalLoop(
      MakeTrunk(absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_depth)),
      /*train_batches=*/64,
      /*num_loops=*/1000,
      /*cfr_oracle_iterations=*/300,
      /*trunk_eval_iterations=*/300,
      /*verbose_every_loop*/false);
}
