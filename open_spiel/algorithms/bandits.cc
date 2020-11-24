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
      cumulative_strategy_(num_actions, 0.),
      current_strategy_(num_actions, 1. / num_actions) {}

const std::vector<double>& RegretMatching::NextStrategy(double weight) {
  double positive_regrets_sum = 0.;
  for (double regret : cumulative_regrets_) {
    positive_regrets_sum += regret > 0. ? regret : 0.;
  }

  if (positive_regrets_sum) {
    for (int i = 0; i < num_actions_; ++i) {
      const double regret = cumulative_regrets_[i];
      current_strategy_[i] =
          (regret > 0. ? regret : 0.) / positive_regrets_sum;
    }
  } else {
    for (int i = 0; i < num_actions_; ++i) {
      current_strategy_[i] = 1. / num_actions_;
    }
  }

  for (int i = 0; i < num_actions_; ++i) {
    cumulative_strategy_[i] += weight * current_strategy_[i];
  }
  return current_strategy_;
}

void RegretMatching::ObserveLoss(absl::Span<const double> loss) {
  SPIEL_DCHECK_EQ(loss.size(), num_actions_);
  double v = 0.;
  for (int i = 0; i < num_actions_; ++i) {
    v += loss[i] * current_strategy_[i];
  }
  for (int i = 0; i < num_actions_; ++i) {
    cumulative_regrets_[i] += v - loss[i];
  }
}

std::vector<double> RegretMatching::AverageStrategy() {
  std::vector<double> strategy;
  strategy.reserve(num_actions_);
  double normalization = 0.;
  for (double action : cumulative_strategy_) normalization += action;

  if (normalization) {
    for (int i = 0; i < num_actions_; ++i) {
      strategy.push_back(cumulative_strategy_[i] / normalization);
    }
  } else {
    for (int i = 0; i < num_actions_; ++i) {
      strategy.push_back(1. / num_actions_);
    }
  }
  return strategy;
}

void RegretMatching::Reset() {
  std::fill(cumulative_regrets_.begin(), cumulative_regrets_.end(), 0.);
  std::fill(cumulative_strategy_.begin(), cumulative_strategy_.end(), 0.);
  std::fill(current_strategy_.begin(), current_strategy_.end(),
            1. / num_actions_);
}

// -- RegretMatchingPlus -------------------------------------------------------

RegretMatchingPlus::RegretMatchingPlus(size_t num_actions)
    : Bandit(num_actions),
      cumulative_regrets_(num_actions, 0.),
      cumulative_strategy_(num_actions, 0.),
      current_strategy_(num_actions, 1. / num_actions),
      time_(1) {}

const std::vector<double>& RegretMatchingPlus::NextStrategy(double weight) {
  double positive_regrets_sum = 0.;
  for (double regret : cumulative_regrets_) {
    positive_regrets_sum += regret > 0. ? regret : 0.;
  }

  if (positive_regrets_sum) {
    for (int i = 0; i < num_actions_; ++i) {
      const double regret = cumulative_regrets_[i];
      current_strategy_[i] =
          (regret > 0. ? regret : 0.) / positive_regrets_sum;
    }
  } else {
    for (int i = 0; i < num_actions_; ++i) {
      current_strategy_[i] = 1. / num_actions_;
    }
  }

  for (int i = 0; i < num_actions_; ++i) {
    cumulative_strategy_[i] += time_ * weight * current_strategy_[i];
  }
  ++time_;
  return current_strategy_;
}

void RegretMatchingPlus::ObserveLoss(absl::Span<const double> loss) {
  SPIEL_DCHECK_EQ(loss.size(), num_actions_);
  double v = 0.;
  for (int i = 0; i < num_actions_; ++i) {
    v += loss[i] * current_strategy_[i];
  }
  for (int i = 0; i < num_actions_; ++i) {
    cumulative_regrets_[i] = std::fmax(0, cumulative_regrets_[i] + v - loss[i]);
  }
}

std::vector<double> RegretMatchingPlus::AverageStrategy() {
  std::vector<double> strategy;
  strategy.reserve(num_actions_);
  double normalization = 0.;
  for (double action : cumulative_strategy_) normalization += action;

  if (normalization) {
    for (int i = 0; i < num_actions_; ++i) {
      strategy.push_back(cumulative_strategy_[i] / normalization);
    }
  } else {
    for (int i = 0; i < num_actions_; ++i) {
      strategy.push_back(1. / num_actions_);
    }
  }
  return strategy;
}

void RegretMatchingPlus::Reset() {
  std::fill(cumulative_regrets_.begin(), cumulative_regrets_.end(), 0.);
  std::fill(cumulative_strategy_.begin(), cumulative_strategy_.end(), 0.);
  std::fill(current_strategy_.begin(), current_strategy_.end(),
            1. / num_actions_);
  time_ = 1;
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
}  // namespace algorithms
}  // namespace open_spiel

