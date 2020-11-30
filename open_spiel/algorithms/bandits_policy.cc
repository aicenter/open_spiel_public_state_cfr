// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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


#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/utils/functional.h"


namespace open_spiel {
namespace algorithms {

std::vector<BanditVector> MakeBanditVectors(
    const std::vector<std::shared_ptr<InfostateTree>>& trees,
    const std::string& bandit_name, GameParameters bandit_params) {
  std::vector<BanditVector> out;
  out.reserve(trees.size());
  for (const std::shared_ptr<InfostateTree>& tree : trees) {
    BanditVector bandits(tree.get());
    for (DecisionId id : bandits.range()) {
      bandits[id] = MakeBandit(
          bandit_name, tree->decision_infostate(id)->num_children(),
          bandit_params);
    }
    out.push_back(std::move(bandits));
  }
  return out;
}

ActionsAndProbs BanditsPolicy::GetInfoStatePolicy(
    const std::string& info_state,
    BanditsPolicy::PolicySelection selection) const {
  const InfostateNode* node = nullptr;
  int pl = 0;
  while (pl < trees_.size() && !node) {
    node = trees_[pl]->DecisionNodeFromInfostateString(info_state);
    pl++;
  }
  SPIEL_CHECK_TRUE(node);  // Infostate not found!
  const int found_pl = pl - 1;
  const bandits::Bandit* bandit =
      bandits_[found_pl][node->decision_id()].get();
  const std::vector<Action>& actions = node->legal_actions();
  std::vector<double> probs;
  if (selection == PolicySelection::kCurrentStrategy) {
    probs = bandit->current_strategy();
  } else if (selection == PolicySelection::kAverageStrategy) {
    probs = bandit->AverageStrategy();
  } else {
    SpielFatalError("Exhausted pattern match!");
  }
  std::vector<std::pair<Action, double>> out;
  Zip(actions.begin(), actions.end(), probs.begin(), out);
  return out;
}


}  // namespace algorithms
}  // namespace open_spiel
