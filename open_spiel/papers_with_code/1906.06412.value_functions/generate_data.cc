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

#include <algorithm>
#include <string>
#include <utility>
#include <memory>

#include "absl/random/random.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;


void RandomizeStrategy(
    std::vector<BanditVector>& bandits, std::mt19937& rnd_gen,
    double prob_pure_strat, double prob_fully_mixed) {
  const bool fully_mixed_strategy =
      std::bernoulli_distribution(prob_fully_mixed)(rnd_gen);
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : bandits[pl].range()) {
      // Randomize current policy
      bandits::Bandit* bandit = bandits[pl][id].get();
      const size_t num_actions = bandit->num_actions();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      absl::Span<double> policy = fixable_bandit->mutable_strategy();

      if (fully_mixed_strategy) {
        for (int i = 0; i < num_actions; ++i) {
          policy[i] = 1. / num_actions;
        }
        continue;
      }

      const bool single_pure_strategy =
          std::bernoulli_distribution(prob_pure_strat)(rnd_gen);
      if (single_pure_strategy) {
        const int which_action =
            std::uniform_int_distribution<>(0, num_actions - 1)(rnd_gen);
        std::fill(policy.begin(), policy.end(), 0.);
        policy[which_action] = 1.;
      } else {
        for (int i = 0; i < num_actions; ++i) {
          if (std::bernoulli_distribution(prob_pure_strat)(rnd_gen)) {
            policy[i] = std::uniform_real_distribution<>(0., 1.)(rnd_gen);
          } else {
            policy[i] = 0.;
          }
        }
        Normalize(absl::MakeSpan(policy));
      }
    }
  }
}


void GenerateData(const std::vector<dlcfr::RangeTable>& tables,
                  dlcfr::DepthLimitedCFR* trunk_with_oracle, BatchData* batch,
                  std::mt19937& rnd_gen, bool verbose) {
  // Randomize strategy in the trunk.
  RandomizeStrategy(trunk_with_oracle->bandits(), rnd_gen);
  // Compute the reach probs from the trunk.
  trunk_with_oracle->UpdateReachProbs();
  // Do not call bottom-up, just evaluate leaves.
  trunk_with_oracle->EvaluateLeaves();
  // Copy the leaves values to the batch.
  CopyRangesAndValues(trunk_with_oracle, tables, batch, verbose);

  if (verbose) {
    for (int i = 0; i < batch->batch_size; ++i) {
      std::cout << "# Public state " << i << std::endl;
      std::cout << "#   Inputs:  " << batch->data_at(i) << std::endl;
      std::cout << "#   Outputs: " << batch->targets_at(i) << std::endl;
    }
    std::cout << "\n# ";
  }
}

void GenerateDataWithDLCfr(Trunk* trunk, std::mt19937& rnd_gen,
                           int pick_max_iter) {
  // The network should imitate DL-CFR when we use this generation method.
  // Running DL-CFR with this leaf evaluator yields following
  // trunk exploitabilities:

  //  0 0.0694444
  //  1 0.0347222
  //  2 0.0138889
  //  3 0.00351338
  //  4 0.00314063
  //  5 0.00261719
  //  6 0.00224331
  //  7 0.00822699
  //  8 0.00823886
  //  9 0.00545127
  //  10 0.00288314
  //  11 0.00123926
  //  12 0.00172119
  //  13 0.0020411
  //  14 0.00237835
  //  15 0.00267345
  //  16 0.00293383
  //  17 0.00191372
  //  18 0.00120782
  //  19 0.00184289

  // CFR leaf evaluator.
  auto leaf_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      trunk->game, /*depth_limit=*/100, /*leaf_evaluator=*/nullptr,
      trunk->terminal_evaluator, trunk->public_observer,
      trunk->infostate_observer);
  leaf_evaluator->reset_subgames_on_evaluation = false;
  leaf_evaluator->bandit_name = "RegretMatchingPlus";
  leaf_evaluator->leaf_evaluator = leaf_evaluator;
  leaf_evaluator->num_cfr_iterations = 2;

  dlcfr::DepthLimitedCFR dl_cfr(trunk->game, trunk->trunk_depth,
                                leaf_evaluator, trunk->terminal_evaluator);
  auto average_policy = dl_cfr.AveragePolicy();

  int which_iteration =
      std::uniform_int_distribution<>(1, pick_max_iter)(rnd_gen);
  dl_cfr.RunSimultaneousIterations(which_iteration - 1);

  dl_cfr.UpdateReachProbs();
  dl_cfr.EvaluateLeaves();
  CopyRangesAndValues(&dl_cfr, trunk->tables, trunk->batch.get(),
      /*verbose=*/false);
}

}  // papers_with_code
}  // open_spiel
