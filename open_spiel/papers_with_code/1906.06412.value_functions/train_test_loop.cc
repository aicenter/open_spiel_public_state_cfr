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


#include <random>

#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "absl/random/random.h"
#include "torch/torch.h"


namespace open_spiel {
namespace papers_with_code {

constexpr size_t kSeed = 0;

// A simple 3-layer MLP neural network.
struct Net : torch::nn::Module {
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
  Net(size_t inputs_size, size_t outputs_size, size_t hidden_size) :
      fc1(inputs_size, hidden_size),
      fc2(hidden_size, hidden_size),
      fc3(hidden_size, outputs_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return torch::relu(fc3->forward(x));
  }
};

torch::Tensor TrainNetwork(Net* model, torch::Device* device,
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
  Net* model_;
  torch::Device* device_;
  std::shared_ptr<const Game> game_;
  std::shared_ptr<Observer> infostate_observer_;
  const std::array<RangeTable, 2>& tables_;
  BatchData* batch_;

 public:
  NetEvaluator(Net* model, torch::Device* device,
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
      PlacementCopy(
          absl::MakeSpan(state->ranges[pl]),
          batch_->ranges_at(state->public_id, pl),
          tables_[pl].bijections[state->public_id].forward());
    }

    torch::Tensor data = batch_->data_tensor_at(state->public_id).to(*device_);
    torch::Tensor output = model_->forward(data);

//    std::cout << state->public_id << " " << data << " " << output << std::endl;

    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy(
          absl::MakeSpan((float*) output.data_ptr(), batch_->ranges_size[pl]),
          absl::MakeSpan(state->values[pl]),
          tables_[pl].bijections[state->public_id].backward());
    }
  }
};

double EvaluateNetwork(
    dlcfr::DepthLimitedCFR* trunk_with_net,
    ortools::OracleEvaluator* oracle_evaluator,
    int iterations) {
  trunk_with_net->RunSimultaneousIterations(iterations);
  return ortools::TrunkExploitability(trunk_with_net, oracle_evaluator);
}

torch::Device FindDevice() {
  if (torch::cuda::is_available()) {
    std::cerr << "CUDA available! Training on GPU." << std::endl;
    return torch::Device(torch::kCUDA);
  } else {
    std::cerr << "Training on CPU." << std::endl;
    return torch::Device(torch::kCPU);
  }
}

void TrainEvalLoop(
    const std::string& game_name,
    int trunk_depth, int train_batches, int num_loops, int trunk_iterations) {

  // 1. Prepare the game, observers and depth-limited (trunk) trees.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<Observer> hand_observer =
      game->MakeObserver(kHandObsType, {});
  std::array<std::shared_ptr<InfostateTree>, 2> trunk_trees = {
      MakeInfostateTree(*game, 0, trunk_depth),
      MakeInfostateTree(*game, 1, trunk_depth)
  };

  // 2. Create value oracle for the trunk.
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  auto oracle_evaluator = std::make_shared<ortools::OracleEvaluator>(
      game, infostate_observer);
  auto trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator, public_observer);

  // 3. Make a Batch of data that encompasses all leaf public states.
  std::array<RangeTable, 2> tables = CreateRangeTables(
      *game, hand_observer, trunk_with_oracle->public_leaves());
  const dlcfr::LeafPublicState& some_leaf =
      trunk_with_oracle->public_leaves().at(0);
  const size_t encoding_size = some_leaf.public_tensor.size();
  std::array<size_t, 2> ranges_size = {tables[0].largest_range(),
                                       tables[1].largest_range()};
  const size_t range_size_sum = ranges_size[0] + ranges_size[1];
  const size_t input_size = encoding_size + range_size_sum;
  const size_t output_size = range_size_sum;
  BatchData batch(trunk_with_oracle->public_leaves(),
                  input_size, output_size, encoding_size, ranges_size);

  // 4. Create network and optimizer.
  torch::manual_seed(kSeed);
  torch::Device device = FindDevice();
  Net model(input_size, output_size, input_size*3);
  model.to(device);
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));

  // 5. Create trunk net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, game, infostate_observer, tables, &batch);
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, net_evaluator, terminal_evaluator, public_observer);

  // 6. The train-eval loop.
  std::mt19937 rnd_gen(kSeed);
  for (int loop = 0; loop < num_loops; ++loop) {
    // Train.
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      GenerateData(tables, trunk_with_oracle.get(), &batch, rnd_gen);
      torch::Tensor loss = TrainNetwork(&model, &device, &optimizer, &batch);
      cumul_loss += loss.item().to<double>();
    }
    const double avg_loss = cumul_loss / train_batches;
    // Eval.
    const double exploitability = EvaluateNetwork(
        trunk_with_net.get(), oracle_evaluator.get(), trunk_iterations);
    // Print.
    std::cout << loop << ',' << avg_loss << ',' << exploitability << std::endl;
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TrainEvalLoop("kuhn_poker",
      /*trunk_depth=*/3,
      /*train_batches=*/8,
      /*num_loops=*/10,
      /*trunk_iterations=*/2);
}
