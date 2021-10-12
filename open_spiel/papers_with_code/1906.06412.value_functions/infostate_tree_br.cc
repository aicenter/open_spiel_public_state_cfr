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


#include "open_spiel/papers_with_code/1906.06412.value_functions/infostate_tree_br.h"

namespace open_spiel {
namespace papers_with_code {

class ResponseBandit : public algorithms::bandits::Bandit {
  int time_;
 public:
  ResponseBandit(std::vector<double> init_strategy)
      : Bandit(std::move(init_strategy)) {
    SPIEL_DCHECK_TRUE(IsValidProbDistribution(current_strategy_));
  }
  void ComputeStrategy(size_t current_time, double reach_prob = 1.) override {
    time_ = current_time;
  }
  void ObserveRewards(absl::Span<const double> rewards) override {
    SPIEL_DCHECK_EQ(rewards.size(), current_strategy_.size());
    if (time_ > 1) return;  // Do not modify on second pass when computing value.

    double max_reward = -std::numeric_limits<double>::infinity();
    int num_max = 0;
    for (const double& reward : rewards) {
      if (reward > max_reward) {
        max_reward = reward;
        num_max = 1;
      } else if (reward == max_reward) {
        ++num_max;
      }
    }
    SPIEL_CHECK_GE(num_max, 1);

    for (int i = 0; i < num_actions(); ++i) {
      if (rewards[i] == max_reward) {
        current_strategy_[i] = 1. / num_max;
      } else {
        current_strategy_[i] = 0.;
      }
    }
    SPIEL_DCHECK_TRUE(IsValidProbDistribution(current_strategy_));
  };
};

// Fix sometimes problematic LP results.
void NumericalNormalization(std::vector<double>& ps) {
  for(double&p : ps) {
    if (p < 1e-10) p = 0;
    if (p > 1. && p < 1.+1e-8) p = 1;
  }
  SPIEL_CHECK_TRUE(IsValidProbDistribution(ps));
}

std::vector<algorithms::BanditVector> MakeResponseBandits(
    const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees,
    const Policy& optimal_brs) {
  std::vector<algorithms::BanditVector> out;
  out.reserve(2);
  for (const std::shared_ptr<algorithms::InfostateTree>& tree : trees) {
    algorithms::BanditVector bandits(tree.get());
    for (auto* node: tree->AllDecisionInfostates()) {
      auto optimal_local_policy = optimal_brs.GetStatePolicy(node->infostate_string());
      if (optimal_local_policy.empty()) {
        int num_actions = node->num_children();
        bandits[node->decision_id()] = std::make_unique<ResponseBandit>(
            std::vector<double>(num_actions, 1. / num_actions));
      } else {
        std::vector<double> ps = GetProbs(optimal_local_policy);
        NumericalNormalization(ps);
        bandits[node->decision_id()] =
            std::make_unique<algorithms::bandits::FixedStrategy>(ps);
      }
    }
    out.push_back(std::move(bandits));
  }
  return out;
}

// BR on infostate trees.
std::vector<double> BestResponse(
    std::vector<std::shared_ptr<algorithms::InfostateTree>> trees,
    const Policy& fixed_policy) {

  algorithms::InfostateCFR cfr(trees, MakeResponseBandits(trees, fixed_policy));

  cfr.RunSimultaneousIterations(1);
  cfr.ResetCumulValues();
  cfr.RunSimultaneousIterations(1);

  return cfr.RootValues();
}


}  // namespace papers_with_code
}  // namespace open_spiel




