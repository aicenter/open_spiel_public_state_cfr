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


#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "absl/random/random.h"

#include "torch/torch.h"


namespace open_spiel {
namespace papers_with_code {

constexpr size_t kSeed = 0;

struct Net : torch::nn::Module {
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

  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
};

torch::Tensor TrainNetwork(Net& model, torch::Device device,
                           torch::optim::Optimizer& optimizer,
                           BatchData& batch) {
  torch::Tensor data = batch.data_tensor().to(device);
  torch::Tensor targets = batch.targets_tensor().to(device);
  optimizer.zero_grad();
  torch::Tensor output = model.forward(data);
  torch::Tensor loss = torch::mse_loss(output, targets);
  AT_ASSERT(!std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer.step();
  return loss;
}

class NetEvaluator : public dlcfr::LeafEvaluator {
  Net* model_;
  std::shared_ptr<const Game> game_;
  std::shared_ptr<Observer> infostate_observer_;
 public:
  NetEvaluator(Net* model, std::shared_ptr<const Game> game,
               std::shared_ptr<Observer> infostate_observer)
      : model_(model), game_(std::move(game)),
        infostate_observer_(std::move(infostate_observer)) {}
  void EvaluatePublicState(dlcfr::LeafPublicState* s,
                           dlcfr::PublicStateContext* context) const override {
    // TODO.
  }
};

std::unique_ptr<dlcfr::DepthLimitedCFR> MakeTrunkWithNetEvaluator(
    Net* model, std::shared_ptr<const Game> game, int trunk_depth) {

  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  auto leaf_evaluator = std::make_shared<NetEvaluator>(model, game, infostate_observer);

  return std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_depth, leaf_evaluator, terminal_evaluator);
}

double EvaluateNetwork(
    dlcfr::DepthLimitedCFR* trunk_with_net,
    ortools::OracleEvaluator* oracle_evaluator,
    int iterations) {
  trunk_with_net->RunSimultaneousIterations(iterations);
  return ortools::TrunkExploitability(trunk_with_net, oracle_evaluator);
}

torch::Device FindDevice() {
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    return torch::Device(torch::kCUDA);
  } else {
    std::cout << "Training on CPU." << std::endl;
    return torch::Device(torch::kCPU);
  }
}

void TrainEvalLoop(
    const std::string& game_name,
    int trunk_depth, int train_batches, int num_loops, int trunk_iterations) {

  // 1. Data generation.
  std::shared_ptr<const Game> game = LoadGame(game_name);
  std::shared_ptr<const dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  auto oracle_evaluator = std::make_shared<ortools::OracleEvaluator>(
      game, infostate_observer);
  std::unique_ptr<dlcfr::DepthLimitedCFR> trunk_with_oracle =
      std::make_unique<dlcfr::DepthLimitedCFR>(
          game, trunk_depth, oracle_evaluator, terminal_evaluator);

  std::shared_ptr<Observer> private_observer =
      game->MakeObserver(kPrivateObsType, {});
  std::array<RangeTable, 2> tables = CreateRangeTables(
      *game, private_observer, trunk_with_oracle->GetPublicLeaves());

  const dlcfr::LeafPublicState& some_leaf =
      trunk_with_oracle->GetPublicLeaves().at(0);
  const size_t encoding_size = some_leaf.public_tensor.size();
  const size_t range_size_sum =
      tables[0].largest_range() + tables[1].largest_range();
  const size_t input_size = encoding_size + range_size_sum;
  const size_t output_size = range_size_sum;
  BatchData batch(
      trunk_with_oracle->GetPublicLeaves(), input_size, output_size, encoding_size,
      {tables[0].largest_range(), tables[1].largest_range()});

  // 2. Network preparation.
  torch::manual_seed(kSeed);
  torch::Device device = FindDevice();
  Net model(input_size, output_size, input_size*3);
  model.to(device);
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(0.01).momentum(0.5));
  std::unique_ptr<dlcfr::DepthLimitedCFR> trunk_with_net =
      MakeTrunkWithNetEvaluator(&model, game, trunk_depth);

//  absl::BitGen bitgen(kSeed);
  absl::BitGen bitgen;
  for (int loop = 0; loop < num_loops; ++loop) {
    double avg_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      GenerateData(tables, trunk_with_oracle.get(), &batch, &bitgen);
      torch::Tensor loss = TrainNetwork(model, device, optimizer, batch);
      avg_loss += loss.item().to<double>();
    }
    const double exploitability = EvaluateNetwork(
        trunk_with_net.get(), oracle_evaluator.get(), trunk_iterations);
    std::cout << loop << ','
              << avg_loss / train_batches << ','
              << exploitability << std::endl;
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TrainEvalLoop("leduc_poker",
      /*trunk_depth=*/3,
      /*train_batches=*/8,
      /*num_loops=*/10,
      /*trunk_iterations=*/200);
}
