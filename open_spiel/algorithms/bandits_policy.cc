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
  if (!node) {  // Infostate not found!
    return {};
  }

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

void AssignUniformRandom(absl::Span<double> policy) {
  for (int i = 0; i < policy.size(); ++i) {
    policy[i] = 1. / policy.size();
  }
}

void AssignSingleAction(absl::Span<double> policy, std::mt19937& rnd_gen) {
  const int which_action =
      std::uniform_int_distribution<>(0, policy.size() - 1)(rnd_gen);
  std::fill(policy.begin(), policy.end(), 0.);
  policy[which_action] = 1.;
}

void AssignMixedStrategy(absl::Span<double> policy, std::mt19937& rnd_gen,
                         double prob_pure_strat) {
  // Mixed strategy.
  for (int i = 0; i < policy.size(); ++i) {
    const bool in_support =
        std::bernoulli_distribution(1. - prob_pure_strat)(rnd_gen);
    policy[i] = in_support
                ? std::uniform_real_distribution<>(0., 1.)(rnd_gen)
                : 0.;
  }
  // We need to normalize the result!
  const double normalizer = absl::c_accumulate(policy, 0.);
  const double uniform_prob = 1.0 / policy.size();
  for (int i = 0; i < policy.size(); ++i) {
    policy[i] = normalizer < 1e-3
                ? uniform_prob
                : policy[i] / normalizer;
  }
}

void RandomizeStrategy(std::vector<BanditVector>& bandits,
                       double prob_pure_strat, double prob_fully_mixed,
                       std::mt19937& rnd_gen) {
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : bandits[pl].range()) {
      // Randomize current policy
      bandits::Bandit* bandit = bandits[pl][id].get();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      absl::Span<double> policy = fixable_bandit->mutable_strategy();
      SPIEL_DCHECK_EQ(policy.size(), bandit->num_actions());

      if (std::bernoulli_distribution(prob_fully_mixed)(rnd_gen)) {
        // Special case since uniform random is a starting point of CFR.
        AssignUniformRandom(policy);
      } else if (std::bernoulli_distribution(prob_pure_strat)(rnd_gen)) {
        AssignSingleAction(policy, rnd_gen);
      } else {
        AssignMixedStrategy(policy, rnd_gen, prob_pure_strat);
      }
    }
  }
}

}  // namespace algorithms
}  // namespace open_spiel
