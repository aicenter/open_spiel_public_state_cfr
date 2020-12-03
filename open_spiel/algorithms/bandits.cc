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

#include "open_spiel/algorithms/bandits.h"

namespace open_spiel {
namespace algorithms {
namespace bandits {

// -- RegretMatching -----------------------------------------------------------

RegretMatching::RegretMatching(size_t num_actions)
    : Bandit(num_actions),
      cumulative_regrets_(num_actions, 0.),
      cumulative_strategy_(num_actions, 0.)  {}

void RegretMatching::ComputeStrategy(size_t current_time, double weight) {
  double positive_regrets_sum = 0.;
  for (double regret : cumulative_regrets_) {
    positive_regrets_sum += regret > 0. ? regret : 0.;
  }

  if (positive_regrets_sum) {
    for (int i = 0; i < num_actions(); ++i) {
      const double regret = cumulative_regrets_[i];
      current_strategy_[i] =
          (regret > 0. ? regret : 0.) / positive_regrets_sum;
    }
  } else {
    for (int i = 0; i < num_actions(); ++i) {
      current_strategy_[i] = 1. / num_actions();
    }
  }

  for (int i = 0; i < num_actions(); ++i) {
    cumulative_strategy_[i] += weight * current_strategy_[i];
  }
}

void RegretMatching::ObserveRewards(absl::Span<const double> rewards) {
  SPIEL_DCHECK_EQ(rewards.size(), num_actions());
  double expected_reward = 0.;
  for (int i = 0; i < num_actions(); ++i) {
    expected_reward += rewards[i] * current_strategy_[i];
  }
  for (int i = 0; i < num_actions(); ++i) {
    cumulative_regrets_[i] += rewards[i] - expected_reward;
  }
}

std::vector<double> RegretMatching::AverageStrategy() const {
  std::vector<double> strategy;
  strategy.reserve(num_actions());
  double normalization = 0.;
  for (double action : cumulative_strategy_) normalization += action;

  if (normalization) {
    for (int i = 0; i < num_actions(); ++i) {
      strategy.push_back(cumulative_strategy_[i] / normalization);
    }
  } else {
    for (int i = 0; i < num_actions(); ++i) {
      strategy.push_back(1. / num_actions());
    }
  }
  return strategy;
}

void RegretMatching::Reset() {
  Bandit::Reset();
  std::fill(cumulative_regrets_.begin(), cumulative_regrets_.end(), 0.);
  std::fill(cumulative_strategy_.begin(), cumulative_strategy_.end(), 0.);
}

// -- RegretMatchingPlus -------------------------------------------------------

RegretMatchingPlus::RegretMatchingPlus(size_t num_actions)
    : Bandit(num_actions),
      cumulative_regrets_(num_actions, 0.),
      cumulative_strategy_(num_actions, 0.) {}

void RegretMatchingPlus::ComputeStrategy(size_t current_time, double weight) {
  double positive_regrets_sum = 0.;
  for (double regret : cumulative_regrets_) {
    positive_regrets_sum += regret > 0. ? regret : 0.;
  }

  if (positive_regrets_sum) {
    for (int i = 0; i < num_actions(); ++i) {
      const double regret = cumulative_regrets_[i];
      current_strategy_[i] =
          (regret > 0. ? regret : 0.) / positive_regrets_sum;
    }
  } else {
    for (int i = 0; i < num_actions(); ++i) {
      current_strategy_[i] = 1. / num_actions();
    }
  }

  for (int i = 0; i < num_actions(); ++i) {
    cumulative_strategy_[i] += current_time * weight * current_strategy_[i];
  }
  return current_strategy_;
}

void RegretMatchingPlus::ObserveRewards(absl::Span<const double> rewards) {
  SPIEL_DCHECK_EQ(rewards.size(), num_actions());
  double expected_reward = 0.;
  for (int i = 0; i < num_actions(); ++i) {
    expected_reward += rewards[i] * current_strategy_[i];
  }
  for (int i = 0; i < num_actions(); ++i) {
    cumulative_regrets_[i] =
        std::fmax(0, cumulative_regrets_[i] + rewards[i] - expected_reward);
  }
}

std::vector<double> RegretMatchingPlus::AverageStrategy() const {
  std::vector<double> strategy;
  strategy.reserve(num_actions());
  double normalization = 0.;
  for (double action : cumulative_strategy_) normalization += action;

  if (normalization) {
    for (int i = 0; i < num_actions(); ++i) {
      strategy.push_back(cumulative_strategy_[i] / normalization);
    }
  } else {
    for (int i = 0; i < num_actions(); ++i) {
      strategy.push_back(1. / num_actions());
    }
  }
  return strategy;
}

void RegretMatchingPlus::Reset() {
  Bandit::Reset();
  std::fill(cumulative_regrets_.begin(), cumulative_regrets_.end(), 0.);
  std::fill(cumulative_strategy_.begin(), cumulative_strategy_.end(), 0.);
}

// TODO:
// -- PredictiveRegretMatching -------------------------------------------------
// -- PredictiveRegretMatchingPlus ---------------------------------------------
// -- FollowTheLeader ----------------------------------------------------------
// -- FollowTheRegularizedLeader -----------------------------------------------
// -- PredictiveFollowTheRegularizedLeader -------------------------------------
// -- OptimisticMirrorDescent --------------------------------------------------
// -- PredictiveOptimisticMirrorDescent ----------------------------------------
// -- Exp3 ---------------------------------------------------------------------
// -- Exp4 ---------------------------------------------------------------------
// -- DiscountedRegretMatching -------------------------------------------------
// -- Hedge --------------------------------------------------------------------
// -- OptimisticHedge ----------------------------------------------------------
// -- UpperConfidenceBounds ----------------------------------------------------
// -- EpsGreedy ----------------------------------------------------------------


}  // namespace bandits

std::unique_ptr<bandits::Bandit> MakeBandit(
    const std::string& bandit_name, size_t num_actions,
    GameParameters bandit_params) {
  if (bandit_name == "RegretMatching") {
    return std::make_unique<bandits::RegretMatching>(num_actions);
  }
  if (bandit_name == "UniformStrategy") {
    return std::make_unique<bandits::UniformStrategy>(num_actions);
  }
  // TODO: finish
  SpielFatalError("Exhausted pattern match!");
}

}  // namespace algorithms
}  // namespace open_spiel

