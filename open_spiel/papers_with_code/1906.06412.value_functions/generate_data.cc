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
                         const std::array<RangeTable, 2>& tables,
                         BatchData* batch, bool verbose = false) {
  const std::vector<dlcfr::LeafPublicState>& leaves = trunk->public_leaves();
  SPIEL_DCHECK_EQ(batch->batch_size, leaves.size());
  for (int i = 0; i < leaves.size(); ++i) {
    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy<float_cfr, float_net>(
          absl::MakeSpan(leaves[i].ranges[pl]),
          batch->ranges_at(i, pl),
          tables[pl].bijections[i].tree_to_net());
      PlacementCopy<float_cfr, float_net>(
          leaves[i].values[pl], batch->values_at(i, pl),
          tables[pl].bijections[i].tree_to_net());
    }
  }

  if (verbose) {
    std::cout << "\n# BatchData copying ranges and values:\n";
    for (int i = 0; i < leaves.size(); ++i) {
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

int RangeTable::largest_range() const { return private_hands.size(); }

size_t RangeTable::hand_index(const Observation& obs) {
  auto it = std::find(private_hands.begin(), private_hands.end(), obs);
  if (it == private_hands.end()) {
    private_hands.push_back(obs);
    return private_hands.size() - 1;
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

std::array<RangeTable, 2> CreateRangeTables(
    const Game& game, const std::shared_ptr<Observer>& hand_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves) {
  std::array<RangeTable, 2> tables{public_leaves.size(), public_leaves.size()};
  Observation hand(game, hand_observer);
  for (int state_idx = 0; state_idx < public_leaves.size(); ++state_idx) {
    const dlcfr::LeafPublicState& state = public_leaves[state_idx];
    for (int pl = 0; pl < 2; ++pl) {
      for (int i = 0; i < state.leaf_nodes[pl].size(); ++i) {
        const InfostateNode* node = state.leaf_nodes[pl][i];
        // All states within an infostate should have the same hands.
        const State& some_state = *node->corresponding_states().at(0);
        hand.SetFrom(some_state, pl);
        size_t j = tables[pl].hand_index(hand);
        tables[pl].bijections[state_idx].put({i, j});
      }
    }
  }
  return tables;
}
void GenerateData(const std::array<RangeTable, 2>& tables,
                  dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                  std::mt19937& rnd_gen, bool verbose) {
  RandomizeTrunkStrategy(trunk->bandits(), rnd_gen, /*prob_pure_strat=*/0.9);
  trunk->RunSimultaneousIterations(1);
  CopyRangesAndValues(trunk, tables, batch, verbose);

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
