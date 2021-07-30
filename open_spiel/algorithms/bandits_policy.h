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

std::vector<BanditVector> MakeBanditVectors(
    const std::vector<std::shared_ptr<InfostateTree>>& trees,
//    const std::string& bandit_name = "RegretMatchingPlus",
    const std::string& bandit_name = "RegretMatching",
    GameParameters bandit_params = {});

enum class PolicySelection {
  kCurrentPolicy,
  kAveragePolicy
};

class BanditsPolicy : public Policy {
  const std::vector<std::shared_ptr<InfostateTree>>& trees_;
  const std::vector<BanditVector>& bandits_;
 public:
  BanditsPolicy(
      const std::vector<std::shared_ptr<InfostateTree>>& trees,
      const std::vector<BanditVector>& bandits)
      : trees_(trees), bandits_(bandits) {}

  ActionsAndProbs GetInfoStatePolicy(
      const std::string& info_state, PolicySelection selection) const;
};

class BanditsAveragePolicy : public BanditsPolicy {
 public:
  BanditsAveragePolicy(const std::vector<std::shared_ptr<InfostateTree>>& trees,
                       const std::vector<BanditVector>& bandits)
      : BanditsPolicy(trees, bandits) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    return GetInfoStatePolicy(info_state, PolicySelection::kAveragePolicy);
  }
};

class BanditsCurrentPolicy : public BanditsPolicy {
 public:
  BanditsCurrentPolicy(const std::vector<std::shared_ptr<InfostateTree>>& trees,
                       const std::vector<BanditVector>& bandits)
      : BanditsPolicy(trees, bandits) {}

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    return GetInfoStatePolicy(info_state, PolicySelection::kCurrentPolicy);
  }
};

// The bandits must be derived from FixableStrategy class.
void RandomizeStrategy(std::vector<algorithms::BanditVector>& bandits,
                       double prob_pure_strat, double prob_fully_mixed,
                       double prob_benford_dist, std::mt19937& rnd_gen);

// Randomize single decision point.
void RandomizeDecisionPoint(absl::Span<double> policy,
                            double prob_pure_strat,
                            double prob_fully_mixed,
                            double prob_benford_dist,
                            std::mt19937& rnd_gen);
void RandomizeDecisionPoint(ActionsAndProbs& policy,
                            double prob_pure_strat,
                            double prob_fully_mixed,
                            double prob_benford_dist,
                            std::mt19937& rnd_gen);

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BANDITS_POLICY_H_
