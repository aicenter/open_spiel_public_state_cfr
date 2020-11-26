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

#ifndef OPEN_SPIEL_ALGORITHMS_BANDITS_POLICY_H_
#define OPEN_SPIEL_ALGORITHMS_BANDITS_POLICY_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/utils/functional.h"


namespace open_spiel {
namespace algorithms {

using BanditVector = DecisionVector<std::unique_ptr<bandits::Bandit>>;

class BanditsCurrentPolicy : public Policy {
  const InfostateTree* tree_;
  const BanditVector* bandits_;
 public:
  BanditsCurrentPolicy(const InfostateTree* tree, const BanditVector* bandits)
      : tree_(tree), bandits_(bandits) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    const InfostateNode* node =
        tree_->DecisionNodeFromInfostateString(info_state);
    SPIEL_CHECK_TRUE(node);
    const bandits::Bandit* bandit = (*bandits_)[node->decision_id()].get();
    const std::vector<Action>& actions = node->legal_actions();
    const std::vector<double>& probs = bandit->current_strategy();
    std::vector<std::pair<Action, double>> out;
    Zip(actions.begin(), actions.end(), probs.begin(), out);
    return out;
  }
};

class BanditsAveragePolicy : public Policy {
  const InfostateTree* tree_;
  const BanditVector* bandits_;
 public:
  BanditsAveragePolicy(const InfostateTree* tree, const BanditVector* bandits)
      : tree_(tree), bandits_(bandits) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    const InfostateNode* node =
        tree_->DecisionNodeFromInfostateString(info_state);
    SPIEL_CHECK_TRUE(node);
    const bandits::Bandit* bandit = (*bandits_)[node->decision_id()].get();
    const std::vector<Action>& actions = node->legal_actions();
    const std::vector<double> probs = bandit->AverageStrategy();
    std::vector<std::pair<Action, double>> out;
    Zip(actions.begin(), actions.end(), probs.begin(), out);
    return out;
  }
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BANDITS_POLICY_H_
