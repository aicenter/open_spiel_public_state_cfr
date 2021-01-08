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


// -- FLAGS --------------------------------------------------------------------

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");
ABSL_FLAG(int, depth, 3, "Depth of the trunk.");
ABSL_FLAG(int, train_batches, 32,
          "Number of training batches before the evalution is run.");
ABSL_FLAG(int, num_loops, 5000, "Number of train-eval loops.");
ABSL_FLAG(int, cfr_oracle_iterations, 100, "Number of oracle iterations.");
ABSL_FLAG(int, trunk_eval_iterations, 100, "Number of trunk iterations.");
ABSL_FLAG(int, num_layers, 3, "Number of hidden layers.");
ABSL_FLAG(int, num_width, 3, "Multiplicative constant of the number "
                             "of neurons per layer.");
ABSL_FLAG(int, seed, 0, "Seed.");
ABSL_FLAG(std::string, use_bandits_for_cfr, "PredictiveRegretMatchingPlus",
          "Which bandit should be used in the trunk.");
ABSL_FLAG(bool, verbose_every_loop, false,
          "Make verbose output at the start of every loop.");

// -----------------------------------------------------------------------------

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/train_eval.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;


void TrainEvalLoop(std::unique_ptr<Trunk> t, int train_batches, int num_loops,
                   int cfr_oracle_iterations, int trunk_eval_iterations,
                   std::string use_bandits_for_cfr, int seed,
                   bool verbose_every_loop) {

  DebugPrintRangeTables(t->tables);
  DebugPrintBatchData(*t->batch);
  DebugPrintPublicFeatures(t->fixable_trunk_with_oracle->public_leaves());

  t->oracle_evaluator->num_cfr_iterations = cfr_oracle_iterations;
  torch::manual_seed(seed);
  std::mt19937 rnd_gen(seed);

  // 1. Create network and optimizer.
  torch::Device device = FindDevice();
  PositionalValueNet model(
      t->batch->input_size, t->batch->output_size,
      t->batch->input_size * absl::GetFlag(FLAGS_num_width),
      absl::GetFlag(FLAGS_num_layers),
      PositionalValueNet::ActivationFunction::kRelu);
  model.to(device);
  torch::optim::Adam optimizer(model.parameters());

  // 2. Create trunk net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, t->game, t->infostate_observer,
      t->tables, t->batch.get());
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      t->game, t->trunk_trees, net_evaluator, t->terminal_evaluator,
      t->public_observer,
      MakeBanditVectors(t->trunk_trees, use_bandits_for_cfr));

  // 3. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*t->game, "CLP");

  std::cout << "# Printing all possible generated data.\n";
  for (int i = 1; i <= trunk_eval_iterations; ++i) {
    GenerateDataWithDLCfr(t.get(), rnd_gen, i);
    double expl = ortools::TrunkExploitability(
        &whole_game, *t->iterable_trunk_with_oracle->AveragePolicy());
    std::cout << "# " << i << ": "
              << "expl = " << expl
              << "\n# " << t->batch->data
              << "\n# " << t->batch->targets << "\n";
  }

  // 4. The train-eval loop.
  std::cout << "loop,avg_loss,exploitability" << std::endl;
  for (int loop = 0; loop < num_loops; ++loop) {
    // Train.
    double cumul_loss = 0.;
    std::cout << "# Training  ";
    for (int i = 0; i < train_batches; ++i) {
      int which_iteration =
          std::uniform_int_distribution<>(1, trunk_eval_iterations)(rnd_gen);
      GenerateDataWithDLCfr(t.get(), rnd_gen, which_iteration);
      torch::Tensor loss = TrainNetwork(&model, &device,
                                        &optimizer, t->batch.get());
      cumul_loss += loss.item().to<double>();
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    // Eval.
    std::cout << "# Evaluating  " << std::endl;
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
  INIT_EXPERIMENT();

  TrainEvalLoop(
      MakeTrunk(absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_depth),
                absl::GetFlag(FLAGS_use_bandits_for_cfr)),
      absl::GetFlag(FLAGS_train_batches),
      absl::GetFlag(FLAGS_num_loops),
      absl::GetFlag(FLAGS_cfr_oracle_iterations),
      absl::GetFlag(FLAGS_trunk_eval_iterations),
      absl::GetFlag(FLAGS_use_bandits_for_cfr),
      absl::GetFlag(FLAGS_seed),
      absl::GetFlag(FLAGS_verbose_every_loop));
}
