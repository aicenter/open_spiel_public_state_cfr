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
#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;

void RandomizeTrunkStrategy(std::vector<BanditVector>& bandits,
                            std::mt19937& rnd_gen, double prob_pure_strat) {
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : bandits[pl].range()) {
      // Randomize current policy
      bandits::Bandit* bandit = bandits[pl][id].get();
      const size_t num_actions = bandit->num_actions();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      std::vector<double>& policy = fixable_bandit->mutable_strategy();

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
      SPIEL_DCHECK_TRUE(IsValidProbDistribution(policy));
    }
  }
}

// Copy generated train data into a network batch.
void CopyRangesAndValues(dlcfr::DepthLimitedCFR* trunk,
                         const std::vector<dlcfr::RangeTable>& tables,
                         BatchData* batch, bool verbose = false) {
  const std::vector<dlcfr::LeafPublicState>& leaves = trunk->public_leaves();
  SPIEL_DCHECK_EQ(batch->batch_size, leaves.size());
  for (size_t i = 0; i < leaves.size(); ++i) {
    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy<float_tree, float_net>(
          /*tree=*/ leaves[i].ranges[pl],
          /*net=*/  batch->ranges_at(i, pl),
          tables[pl].bijections[i].tree_to_net());
      PlacementCopy<float_tree, float_net>(
          /*tree=*/ leaves[i].values[pl],
          /*net=*/  batch->values_at(i, pl),
          tables[pl].bijections[i].tree_to_net());
    }
  }

  if (verbose) {
    std::cout << "\n# BatchData copying ranges and values:\n";
    for (size_t i = 0; i < leaves.size(); ++i) {
      for (int pl = 0; pl < 2; ++pl) {
        std::cout << "#   leaves[" << i << "].ranges[" << pl << "]    = "
                  << leaves[i].ranges[pl] << "\n";
        std::cout << "#   batch->ranges_at(" << i << ", " << pl << ") = "
                  << batch->ranges_at(i, pl) << "\n";
        std::cout << "#   leaves[" << i << "].values[" << pl << "]    = "
                  << leaves[i].values[pl] << "\n";
        std::cout << "#   batch->values_at(" << i << ", " << pl << ") = "
                  << batch->values_at(i, pl) << "\n";
      }
    }
    std::cout << "#\n";
    std::cout << "#   batch->data    = " << batch->data << "\n";
    std::cout << "#   batch->targets = " << batch->targets << "\n";
  }
}

void GenerateData(const std::vector<dlcfr::RangeTable>& tables,
                  dlcfr::DepthLimitedCFR* trunk_with_oracle, BatchData* batch,
                  std::mt19937& rnd_gen, bool verbose) {
  RandomizeTrunkStrategy(trunk_with_oracle->bandits(), rnd_gen,
                         /*prob_pure_strat=*/0.3);
  // This call invokes public state evaluation under the hood.
  trunk_with_oracle->RunSimultaneousIterations(1);
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

}  // papers_with_code
}  // open_spiel
