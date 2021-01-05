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
  RandomizeStrategy(trunk->bandits(), rnd_gen, /*prob_pure_strat=*/0.9);
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

double LearnFixedValues(BatchData* batch, ValueNet* model,
                        torch::Device* device,
                        int train_batches = 32, int num_loops = 20) {
  // 1. Create the optimizer.
  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(/*lr=*/0.4));

  // 2. Train the network.
  double avg_loss;
  for (int loop = 0; loop < num_loops; ++loop) {
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      torch::Tensor loss = TrainNetwork(model, device, &optimizer, batch);
      cumul_loss += loss.item().to<double>();
    }
    avg_loss = cumul_loss / train_batches;
  }
  return avg_loss;
}

void FillRanges(std::vector<dlcfr::LeafPublicState>& states, float_tree value) {
  for (auto& state: states) {
    if (state.IsTerminal()) continue;
    for (int pl = 0; pl < 2; ++pl) {
      std::fill(state.ranges[pl].begin(), state.ranges[pl].end(), value);
    }
  }
}

void FillValues(std::vector<dlcfr::LeafPublicState>& states, float_tree value) {
  for (auto& state: states) {
    if (state.IsTerminal()) continue;
    for (int pl = 0; pl < 2; ++pl) {
      std::fill(state.values[pl].begin(), state.values[pl].end(), value);
    }
  }
}

void CheckRanges(std::vector<dlcfr::LeafPublicState>& states, float_tree value,
                 double eps = 0.03) {
  for (auto& state: states) {
    if (state.IsTerminal()) continue;
    for (int pl = 0; pl < 2; ++pl) {
      for (size_t i = 0; i < state.ranges[pl].size(); ++i) {
        SPIEL_CHECK_FLOAT_NEAR(state.ranges[pl][i], value, eps);
      }
    }
  }
}

void CheckValues(std::vector<dlcfr::LeafPublicState>& states, float_tree value,
                 double eps = 0.03) {
  for (auto& state: states) {
    if (state.IsTerminal()) continue;
    for (int pl = 0; pl < 2; ++pl) {
      for (size_t i = 0; i < state.values[pl].size(); ++i) {
        SPIEL_CHECK_FLOAT_NEAR(state.values[pl][i], value, eps);
      }
    }
  }
}

void LearnFixedValuesTest(std::unique_ptr<Trunk> t) {
  torch::manual_seed(kSeed);
  const float_net fixed_range_input = 0.5;
  const float_net fixed_value_ouput = 1.0;

  torch::Device device = FindDevice();
  PositionalValueNet model(t->batch->input_size, t->batch->output_size,
                           t->batch->input_size * 3);
  model.to(device);

  // Make a single target -- all values are 1.0, all ranges are 0.5
  std::vector<dlcfr::LeafPublicState>& public_leaves =
      t->trunk_with_oracle->public_leaves();
  FillRanges(public_leaves, fixed_range_input);
  CheckRanges(public_leaves, fixed_range_input);  // Test the test.
  FillValues(public_leaves, fixed_value_ouput);
  CheckValues(public_leaves, fixed_value_ouput);  // Test the test.

  CopyRangesAndValues(t.get());
  double avg_loss = LearnFixedValues(t->batch.get(), &model, &device);
  SPIEL_CHECK_LT(avg_loss, 1e-5);

  // Test that training did not touch batch data.
  CheckRanges(public_leaves, fixed_range_input);
  CheckValues(public_leaves, fixed_value_ouput);

  // Check if the value network indeed learned anything.
  FillValues(public_leaves, 0.0);
  CheckValues(public_leaves, 0.0);  // Test the test.

  NetEvaluator net_evaluator(&model, &device,
                             t->game, t->infostate_observer,
                             t->tables, t->batch.get());

  for (int i = 0; i < t->trunk_with_oracle->public_leaves().size(); ++i) {
    dlcfr::LeafPublicState* state = &t->trunk_with_oracle->public_leaves()[i];
    if (!state->IsTerminal()) net_evaluator.EvaluatePublicState(state, nullptr);
  }

  // Evaluation should not change ranges.
  CheckRanges(public_leaves, fixed_range_input);
  // Finally, check the learned values.
  CheckValues(public_leaves, fixed_value_ouput);
}

enum KuhnLearnCase {
  kLearnNashEqDeterministicTargets,
  kLearnNashEqRandomizedTargets,
  kLearnExploitableStrategyRandomizedTargets,
  kLearnExploitableStrategyDeterministicTargets,
};

void KuhnLearningTest(KuhnLearnCase learning_case) {
  // 1. Setup.
  torch::manual_seed(kSeed);
  std::mt19937 rnd_gen(kSeed);

  std::unique_ptr<Trunk> t = MakeTrunk("kuhn_poker", 3);
  torch::Device device = FindDevice();
  PositionalValueNet model(t->batch->input_size, t->batch->output_size,
                           t->batch->input_size * 3);
  model.to(device);

  std::vector<dlcfr::LeafPublicState>& public_leaves =
      t->trunk_with_oracle->public_leaves();
  dlcfr::LeafPublicState& pass_state = public_leaves[0];
  dlcfr::LeafPublicState& bet_state = public_leaves[1];

  // 2. Create trunk net evaluator.
  auto net_evaluator = std::make_shared<NetEvaluator>(
      &model, &device, t->game, t->infostate_observer,
      t->tables, t->batch.get());
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      t->game, t->trunk_trees, net_evaluator, t->terminal_evaluator,
      t->public_observer,
      MakeBanditVectors(t->trunk_trees, "RegretMatchingPlus"));

  // 3. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*t->game, "CLP");

  double expl_before_training = EvaluateNetwork(trunk_with_net.get(), 100,
                                                &whole_game);
  // Should get a high expl before training.
  SPIEL_CHECK_GT(expl_before_training, 0.1);

  // 4. Learn the fixed values.
  for (int j = 0; j <= 64; ++j) {
    // Generate random ranges.
    RandomizeStrategy(t->trunk_with_oracle->bandits(), rnd_gen);
    t->trunk_with_oracle->UpdateReachProbs();

    if (learning_case == kLearnNashEqDeterministicTargets) {
      // The player will learn to get high values for passing, and low values
      // for betting. This corresponds to Nash equilibrium with alpha = 0
      std::fill(pass_state.values[0].begin(), pass_state.values[0].end(), 1);
      std::fill(bet_state.values[0].begin(), bet_state.values[0].end(), -1);
    } else if (learning_case == kLearnNashEqRandomizedTargets) {
      // Same, but randomized.
      for (double& v: pass_state.values[0])
        v = std::uniform_real_distribution<>(0., 1.)(rnd_gen);
      for (double& v: bet_state.values[0])
        v = std::uniform_real_distribution<>(-1., 0.)(rnd_gen);
    } else if (learning_case == kLearnExploitableStrategyDeterministicTargets) {
      // The player will learn to get high values for betting, and low values
      // for passing. Only betting is not a Nash eq and is highly exploitable.
      std::fill(pass_state.values[0].begin(), pass_state.values[0].end(), -1);
      std::fill(bet_state.values[0].begin(), bet_state.values[0].end(), 1);
    } else if (learning_case == kLearnExploitableStrategyRandomizedTargets) {
      // Same, but randomized.
      for (double& v: pass_state.values[0])
        v = std::uniform_real_distribution<>(-1., 0.)(rnd_gen);
      for (double& v: bet_state.values[0])
        v = std::uniform_real_distribution<>(0., 1.)(rnd_gen);
    }

    CopyRangesAndValues(t.get());
    LearnFixedValues(t->batch.get(), &model, &device, 1, 1);
  }

  double expl_after_training = EvaluateNetwork(trunk_with_net.get(), 100,
                                               &whole_game);
  SPIEL_CHECK_NE(expl_before_training, expl_after_training);

  switch (learning_case) {
    case kLearnNashEqDeterministicTargets:
    case kLearnNashEqRandomizedTargets:
      SPIEL_CHECK_LT(expl_after_training, 1e-4);
      break;
    case kLearnExploitableStrategyDeterministicTargets:
    case kLearnExploitableStrategyRandomizedTargets:
      SPIEL_CHECK_GT(expl_after_training, 0.1);
      break;
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::papers_with_code;

  LearnFixedValuesTest(MakeTrunk("matrix_mp", /*trunk_depth=*/0));
  LearnFixedValuesTest(MakeTrunk("kuhn_poker", /*trunk_depth=*/3));
  LearnFixedValuesTest(MakeTrunk("leduc_poker", /*trunk_depth=*/4));
  LearnFixedValuesTest(MakeTrunk(
      "goofspiel(players=2,num_cards=3,imp_info=True)", /*trunk_depth=*/1));

  KuhnLearningTest(kLearnNashEqDeterministicTargets);
  KuhnLearningTest(kLearnNashEqRandomizedTargets);
  KuhnLearningTest(kLearnExploitableStrategyDeterministicTargets);
  KuhnLearningTest(kLearnExploitableStrategyRandomizedTargets);
}
