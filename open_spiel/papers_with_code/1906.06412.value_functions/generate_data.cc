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

void RandomizeStrategy(std::vector<BanditVector>& bandits,
                       std::mt19937& rnd_gen, double prob_pure_strat) {
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : bandits[pl].range()) {
      // Randomize current policy
      bandits::Bandit* bandit = bandits[pl][id].get();
      const size_t num_actions = bandit->num_actions();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      absl::Span<double> policy = fixable_bandit->mutable_strategy();

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
  RandomizeStrategy(trunk_with_oracle->bandits(), rnd_gen,
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
