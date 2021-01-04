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

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/train_eval.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

constexpr size_t kSeed = 0;


void PrepareTestData(const std::vector<dlcfr::RangeTable>& tables,
                     dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                     std::mt19937& rnd_gen) {
  RandomizeTrunkStrategy(trunk->bandits(), rnd_gen, /*prob_pure_strat=*/0.9);
  trunk->RunSimultaneousIterations(1);

  auto& leaves = trunk->public_leaves();
  for (size_t i = 0; i < leaves.size(); ++i) {
    for (int pl = 0; pl < 2; ++pl) {
      // Copy randomized ranges.
      PlacementCopy<float_tree, float_net>(
          /*tree=*/ leaves[i].ranges[pl],
          /*net=*/  batch->ranges_at(i, pl),
                    tables[pl].bijections[i].tree_to_net());
      // Set fixed values.
      for (size_t j = 0; j < batch->values_at(i, pl).size(); ++j) {
        batch->values_at(i, pl)[j] = 1.;
      }
    }
  }
}

void LearnFixedValuesTest(std::unique_ptr<Trunk> t,
                          int train_batches, int num_loops) {
  std::mt19937 rnd_gen(kSeed);

  // 1. Create network and optimizer.
  torch::Device device = FindDevice();
  PositionalValueNet model(t->batch->input_size, t->batch->output_size,
                           t->batch->input_size * 3);
  model.to(device);
  torch::optim::SGD optimizer(model.parameters(),
                              torch::optim::SGDOptions(/*lr=*/0.4));

  // 2. Make a single target.
  PrepareTestData(t->tables, t->trunk_with_oracle.get(),
                  t->batch.get(), rnd_gen);

  // 3. Train the network.
  double avg_loss;
  for (int loop = 0; loop < num_loops; ++loop) {
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      torch::Tensor loss = TrainNetwork(&model, &device, &optimizer,
                                        t->batch.get());
      cumul_loss += loss.item().to<double>();
    }
    avg_loss = cumul_loss / train_batches;
    std::cout << loop << "," << avg_loss << std::endl;
  }

  // 4. Create the net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, t->game, t->infostate_observer, t->tables, t->batch.get());

  // 5. Run the network, check outputs.
  dlcfr::LeafPublicState& some_leaf = t->trunk_with_oracle->public_leaves()[0];
  net_evaluator->EvaluatePublicState(&some_leaf, nullptr);
  std::cout << "Values outputs: " << some_leaf.values << "\n";

  SPIEL_CHECK_LT(avg_loss, 1e-3);
  for (int pl = 0; pl < 2; ++pl) {
    for (size_t i = 0; i < some_leaf.values[pl].size(); ++i) {
      SPIEL_CHECK_TRUE(Near<double>(some_leaf.values[pl][i], 1., 0.03));
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::papers_with_code;

  LearnFixedValuesTest(MakeTrunk("kuhn_poker", /*trunk_depth=*/3),
                       /*train_batches=*/32, /*num_loops=*/20);
}
