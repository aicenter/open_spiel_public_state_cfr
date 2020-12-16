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
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/neural_nets.h"
#include "open_spiel/utils/format_observation.h"
#include "torch/torch.h"


namespace open_spiel {
namespace papers_with_code {

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
      PlacementCopy<float_cfr, float_net>(
          absl::MakeSpan(state->ranges[pl]),
          batch_->ranges_at(state->public_id, pl),
          tables_[pl].bijections[state->public_id].tree_to_net());
    }

    torch::Tensor data = batch_->data_tensor_at(state->public_id).to(*device_);
    torch::Tensor output = model_->forward(data);

    auto raw_output = (float*) output.data_ptr();
    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy<float_net, float_cfr>(
          absl::MakeSpan(&raw_output[batch_->range_offset(pl)],
                         batch_->ranges_size[pl]),
          absl::MakeSpan(state->values[pl]),
          tables_[pl].bijections[state->public_id].net_to_tree());
    }
  }
};

double EvaluateNetwork(dlcfr::DepthLimitedCFR* trunk_with_net, int iterations,
                       ortools::SequenceFormLpSpecification* whole_game) {
  for (BanditVector& bandits : trunk_with_net->bandits()) {
    for (DecisionId id : bandits.range()) {
      bandits[id]->Reset();
    }
  }
  std::cout << " (trunk iters) ";
  trunk_with_net->RunSimultaneousIterations(iterations);
  std::cout << " (trunk expl) ";
  return ortools::TrunkExploitability(
      whole_game, *trunk_with_net->AveragePolicy());
}

torch::Device FindDevice() {
  if (torch::cuda::is_available()) {
    std::cerr << "# CUDA available! Training on GPU." << std::endl;
    return torch::Device(torch::kCUDA);
  } else {
    std::cerr << "# Training on CPU." << std::endl;
    return torch::Device(torch::kCPU);
  }
}

// <editor-fold desc=" Debugging functions ">

void PrintRangeTables(const std::array<RangeTable, 2>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# List of private hands for pl " << pl << "\n";
    const RangeTable& table = tables[pl];
    for (int i = 0; i < table.private_hands.size(); ++i) {
      std::cout << "#   private_hand[" << i << "]:\n#      "
                << ObservationToString(table.private_hands[i], "\n#      ") << "\n";
    }

    std::cout << "# List of bijections (tree -> net) for pl " << pl << "\n";
    for (size_t i = 0; i < table.bijections.size(); ++i) {
      std::cout << "#  Public state " << i << "\n";
      const std::map<size_t, size_t>& tree_to_net =
          table.bijections[i].tree_to_net();
      for (auto&[key, val] : tree_to_net) {
        std::cout << "#   " << key << " -> " << val << "\n";
      }
    }
  }
}

void PrintBatchData(const BatchData& batch,
                    const std::vector<dlcfr::LeafPublicState>& states) {
  std::cout << "# Made BatchData with sizes:\n"
            << "#   batch_size=" << batch.batch_size << "\n"
            << "#   input_size=" << batch.input_size << "\n"
            << "#   output_size=" << batch.output_size << "\n"
            << "#   public_features_size=" << batch.public_features_size << "\n"
            << "#   ranges_size=" << batch.ranges_size << "\n";
  std::cout << "# Public features:\n";
  for (int i = 0; i < states.size(); ++i) {
    std::cout << "#   states[" << i << "].public_tensor\n#     "
              << ObservationToString(states[i].public_tensor, "\n#     ") << "\n";
  }
  std::cout << "# BatchData after feature copying:\n";
  std::cout << "#   " << batch.data << "\n";
}

// </editor-fold>

void TrainEvalLoop(const std::string& game_name,
                   int trunk_depth, int train_batches, int num_loops,
                   int cfr_oracle_iterations, int trunk_eval_iterations,
                   bool verbose_every_loop) {

  // 1. Prepare the game, observers and depth-limited (trunk) trees.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> hand_observer =
      game->MakeObserver(kHandHistoryObsType, {});
  std::vector<std::shared_ptr<InfostateTree>> trunk_trees = {
      MakeInfostateTree(*game, 0, trunk_depth),
      MakeInfostateTree(*game, 1, trunk_depth)
  };

  // 2. Create value oracle for the trunk.
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  auto oracle_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      game, /*full_subgame_depth=*/100, /*no_leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  oracle_evaluator->num_cfr_iterations = cfr_oracle_iterations;
  auto trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator, public_observer,
      MakeBanditVectors(trunk_trees, "FixableStrategy"));

  // 3. Make a Batch of data that encompasses all leaf public states.
  std::array<RangeTable, 2> tables = CreateRangeTables(
      *game, hand_observer, trunk_with_oracle->public_leaves());
  PrintRangeTables(tables);
  const dlcfr::LeafPublicState& some_leaf =
      trunk_with_oracle->public_leaves().at(0);
  const size_t encoding_size = some_leaf.public_tensor.Tensor().size();
  std::array<size_t, 2> ranges_size = {tables[0].largest_range(),
                                       tables[1].largest_range()};
  const size_t range_size_sum = ranges_size[0] + ranges_size[1];
  const size_t input_size = encoding_size + range_size_sum;
  const size_t output_size = range_size_sum;
  BatchData batch(trunk_with_oracle->public_leaves(),
                  input_size, output_size, encoding_size, ranges_size);
  PrintBatchData(batch, trunk_with_oracle->public_leaves());

  // 4. Create network and optimizer.
  torch::manual_seed(kSeed);
  torch::Device device = FindDevice();
  PositionalValueNet model(input_size, output_size, input_size * 3);
  model.to(device);
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  // 5. Create trunk net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, game, infostate_observer, tables, &batch);
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, net_evaluator, terminal_evaluator, public_observer,
      MakeBanditVectors(trunk_trees, kUseBanditsForCfr));

  // 6. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*game, "CLP");

  // 7. The train-eval loop.
  std::cout << "loop,avg_loss,exploitability" << std::endl;
  std::mt19937 rnd_gen(kSeed);
  for (int loop = 0; loop < num_loops; ++loop) {
    // Train.
    double cumul_loss = 0.;
    std::cout << "# Training  ";
    for (int i = 0; i < train_batches; ++i) {
      GenerateData(tables, trunk_with_oracle.get(), &batch, rnd_gen,
          /*verbose=*/(i == 0 && verbose_every_loop) || (i == 0 && loop == 0));
      torch::Tensor loss = TrainNetwork(&model, &device, &optimizer, &batch);
      cumul_loss += loss.item().to<double>();
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    // Eval.
    std::cout << "# Evaluating  ";
    const double exploitability = EvaluateNetwork(
        trunk_with_net.get(), trunk_eval_iterations, &whole_game);
    const double avg_loss = cumul_loss / train_batches;
    std::cout << std::endl;

    // Print.
    std::cout << loop << ',' << avg_loss << ',' << exploitability << std::endl;
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  open_spiel::papers_with_code::TrainEvalLoop(
      absl::GetFlag(FLAGS_game_name),
      /*trunk_depth=*/absl::GetFlag(FLAGS_depth),
      /*train_batches=*/8,
      /*num_loops=*/100,
      /*cfr_oracle_iterations=*/100,
      /*trunk_eval_iterations=*/100,
      /*verbose_every_loop*/false);
}
