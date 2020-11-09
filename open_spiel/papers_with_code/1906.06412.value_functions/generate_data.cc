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

namespace {

void RandomizeTrunkStrategy(
    std::array<DecisionVector<CFRInfoStateValues>, 2> node_values,
    absl::BitGen* bitgen, double prob_pure_strat) {
  for (int pl = 0; pl < 2; ++pl) {
    for (CFRInfoStateValues& values : node_values[pl]) {
      // Randomize current policy
      std::vector<double>& policy = values.current_policy;

      const bool single_pure_strategy =
          absl::Bernoulli(*bitgen, prob_pure_strat);
      if (single_pure_strategy) {
        const int
            which_action = absl::Uniform(*bitgen, 0, values.num_actions());
        std::fill(policy.begin(), policy.end(), 0.);
        policy[which_action] = 1.;
      } else {
        for (int i = 0; i < values.num_actions(); ++i) {
          if (absl::Bernoulli(*bitgen, prob_pure_strat)) {
            policy[i] = absl::Uniform(*bitgen, 0., 1.);
          } else {
            policy[i] = 0.;
          }
        }
      }
    }
  }
}
void PlacementCopy(const std::vector<float>& from, absl::Span<float> to,
                   std::map<int, int> from_to) {
  SPIEL_CHECK_EQ(from.size(), from_to.size());
  for (int i = 0; i < from.size(); ++i) {
    const int j = from_to[i];
    to[j] = from[i];
  }
}

} // namespace


void CopyRangesAndValues(dlcfr::DepthLimitedCFR* trunk,
                         const std::array<RangeTable, 2>& tables,
                         BatchData* batch) {
  const std::vector<dlcfr::LeafPublicState>& leaves = trunk->GetPublicLeaves();
  for (int i = 0; i < leaves.size(); ++i) {
    for (int pl = 0; pl < 2; ++pl) {
      PlacementCopy(leaves[i].ranges[pl], batch->ranges_at(i, pl),
                    tables[pl].bijections[i].association(0));
      PlacementCopy(leaves[i].values[pl], batch->values_at(i, pl),
                    tables[pl].bijections[i].association(0));
    }
  }
}

int RangeTable::largest_range() const { return private_hands.size(); }

int RangeTable::hand_index(const Observation& obs) {
  auto it = std::find(private_hands.begin(), private_hands.end(), obs);
  if (it == private_hands.end()) {
    private_hands.push_back(obs);
    return private_hands.size() - 1;
  } else {
    return std::distance(private_hands.begin(), it);
  }
}

std::array<RangeTable, 2> CreateRangeTables(
    const Game& game,
    const std::shared_ptr<Observer>& private_observer,
    const std::vector<dlcfr::LeafPublicState>& public_leaves) {
  std::array<RangeTable, 2> tables{public_leaves.size(), public_leaves.size()};
  Observation hand(game, private_observer);
  for (int i = 0; i < public_leaves.size(); ++i) {
    const dlcfr::LeafPublicState& state = public_leaves[i];
    for (int pl = 0; pl < 2; ++pl) {
      for (int j = 0; j < state.leaf_nodes[pl].size(); ++j) {
        const InfostateNode* node = state.leaf_nodes[pl][j];
        const State& some_state = *node->corresponding_states().at(0);
        hand.SetFrom(some_state, pl);
        tables[pl].bijections[i].put({tables[pl].hand_index(hand), j});
      }
    }
  }
  return tables;
}
void GenerateData(const std::array<RangeTable, 2>& tables,
                  dlcfr::DepthLimitedCFR* trunk, BatchData* batch,
                  absl::BitGen* bitgen) {
  RandomizeTrunkStrategy(trunk->node_values(), bitgen, /*prob_pure_strat=*/0.9);
  trunk->RunSimultaneousIterations(1);
  CopyRangesAndValues(trunk, tables, batch);
//  for (int i = 0; i < batch->batch_size; ++i) {
//    std::cout << "Inputs: " << batch->data_at(i) << std::endl;
//    std::cout << "Ouputs: " << batch->targets_at(i) << std::endl;
//  }
}

}  // papers_with_code
}  // open_spiel
