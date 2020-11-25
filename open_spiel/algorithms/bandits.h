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

#ifndef OPEN_SPIEL_ALGORITHMS_BANDITS_H_
#define OPEN_SPIEL_ALGORITHMS_BANDITS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This file contains implementations of (multi-armed) bandit algorithms.
//
// At each time `t`, a bandit with `n` arms for the online linear optimization
// (OLO) problem supports the following two operations:
//
// 1. ComputeStrategy() computes a strategy `x_t ∈ S^n` (probability simplex
//    `S^n ⊆ R^n`).
// 2. ObserveLoss() receives a loss vector `l_t` that is meant to evaluate the
//    strategy `x_t` that was last computed.
//
// Specifically, the bandit incurs a loss equal to inner product of `l_t` and
// `x_t`. The loss vector `l_t` can depend on all past strategies that were
// output by the bandit. The bandit operates online in the sense that each
// strategy `x_t` can depend only on the decision `x_1 , ... , x_(t−1)` output
// in the past, as well as the loss vectors `l_1 , ... , l_(t−1)` that were
// observed in the past. No information about the future losses `l_t, l_(t+1),
// ...` is available to the bandit at time `t`.
//
// An example of a computation of a loss function: if bandit algorithms are used
// on a zero-sum normal-form game described by a payoff matrix `A` for the first
// player, then the loss vector `l_t` for the first player is `-Ay` and `x^T A`
// for the second player, where `x` and `y` are the strategies for the players
// one and two respectively.

namespace open_spiel {
namespace algorithms {
namespace bandits {

class Bandit {
 protected:
  const size_t num_actions_;
  std::vector<double> current_strategy_;
 public:
  Bandit(size_t num_actions) :
      num_actions_(num_actions),
      current_strategy_(num_actions, 1. / num_actions) {
    SPIEL_CHECK_GT(num_actions_, 0);
  }
  virtual ~Bandit() = default;
  // Return the positive number of actions (arms) available to the bandit.
  size_t num_actions() const { return num_actions_; }

  // Reset the bandit to the same state as when it was constructed.
  virtual void Reset() {
    std::fill(current_strategy_.begin(), current_strategy_.end(),
              1. / num_actions_);
  }

  // Compute the strategy `x_t` and save it into current_strategy().
  //
  // Optionally, the algorithm receives a weight it should put on the strategy.
  // This is intended for the use case within counter-factual regret
  // minimization framework, and the weight is the reach probability of the
  // current strategy.
  virtual void ComputeStrategy(size_t current_time, double weight = 1.) = 0;

  // Return the strategy `x_t`
  virtual const std::vector<double>& current_strategy() const {
    return current_strategy_;
  }

  // Observe the loss `l_t` incurred after the strategy `x_t` was used.
  virtual void ObserveLoss(absl::Span<const double> loss) = 0;

  // Does this bandit compute also an average strategy?
  virtual bool uses_average_strategy() { return false; }
  virtual std::vector<double> AverageStrategy() {
    SpielFatalError("AverageStrategy() is not implemented.");
  }

  // Does this bandit use (externally supplied) predictions?
  // If it does, the function `ObservePrediction()` is called before each call
  // of `ComputeStrategy()`.
  virtual bool uses_predictions() { return false; }
  virtual void ObservePrediction(absl::Span<const double> prediction) {
    SpielFatalError("ObservePrediction() is not implemented.");
  }

  // Does this bandit use a context for computation of its strategy?
  // If it does, the function `ObserveContext()` is called before each call
  // of `ComputeStrategy()` and `ObservePrediction()`.
  virtual bool uses_context() { return false; }
  virtual void ObserveContext(absl::Span<const double> context) {
    SpielFatalError("ObserveContext() is not implemented.");
  }
};

// [1] A Simple Adaptive Procedure Leading to Correlated Equilibrium
//     Sergiu Hart, Andreu Mas‐Colell
//     http://wwwf.imperial.ac.uk/~dturaev/Hart0.pdf
class RegretMatching final : public Bandit {
  std::vector<double> cumulative_regrets_;
  std::vector<double> cumulative_strategy_;
 public:
  RegretMatching(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// [2] Solving Large Imperfect Information Games Using CFR+
//     Oskari Tammelin
//     https://arxiv.org/abs/1407.5042
class RegretMatchingPlus final : public Bandit {
  std::vector<double> cumulative_regrets_;
  std::vector<double> cumulative_strategy_;
 public:
  RegretMatchingPlus(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Faster Game Solving via Predictive Blackwell Approachability:
// Connecting Regret Matching and Mirror Descent
// Gabriele Farina, Christian Kroer, Tuomas Sandholm
// https://arxiv.org/abs/2007.14358
class PredictiveRegretMatching final : public Bandit {
 public:
  PredictiveRegretMatching(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Faster Game Solving via Predictive Blackwell Approachability:
// Connecting Regret Matching and Mirror Descent
// Gabriele Farina, Christian Kroer, Tuomas Sandholm
// https://arxiv.org/abs/2007.14358
class PredictiveRegretMatchingPlus final : public Bandit {
 public:
  PredictiveRegretMatchingPlus(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Follow-the-leader
// https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture2/scribeNote.pdf
// Isn't that just greedy??
class FollowTheLeader final : public Bandit {
 public:
  FollowTheLeader(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Follow-the-regularized-leader
// http://www-stat.wharton.upenn.edu/~steele/Resources/Projects/SequenceProject/Hannan.pdf
// https://ttic.uchicago.edu/~shai/papers/ShalevSi07_mlj.pdf
// https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture3/lecture3.pdf
class FollowTheRegularizedLeader final : public Bandit {
 public:
  FollowTheRegularizedLeader(size_t num_actions,
                             std::function<double(
                                 std::vector<double>/*weight*/)> regularizer);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Faster Game Solving via Predictive Blackwell Approachability:
// Connecting Regret Matching and Mirror Descent
// Gabriele Farina, Christian Kroer, Tuomas Sandholm
// https://arxiv.org/abs/2007.14358
class PredictiveFollowTheRegularizedLeader final : public Bandit {
 public:
  PredictiveFollowTheRegularizedLeader(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class OptimisticMirrorDescent final : public Bandit {
 public:
  OptimisticMirrorDescent(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class PredictiveOptimisticMirrorDescent final : public Bandit {
 public:
  PredictiveOptimisticMirrorDescent(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Exp3 final : public Bandit {
 public:
  Exp3(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Exp4 final : public Bandit {
 public:
  Exp4(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Solving Imperfect-Information Gamesvia Discounted Regret Minimization
// Noam Brown, Tuomas Sandholm
// https://arxiv.org/pdf/1809.04040v3.pdf
class DiscountedRegretMatching final : public Bandit {
 public:
  DiscountedRegretMatching(size_t num_actions, double alpha, double beta,
                           double gamma);
  bool uses_average_strategy() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Hedge final : public Bandit {
 public:
  Hedge(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// https://arxiv.org/pdf/1507.00407.pdf
class OptimisticHedge final : public Bandit {
 public:
  OptimisticHedge(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class UpperConfidenceBounds final : public Bandit {
 public:
  UpperConfidenceBounds(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class EpsGreedy final : public Bandit {
 public:
  EpsGreedy(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  void ComputeStrategy(size_t current_time, double weight = 1.) override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

}  // namespace bandits
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BANDITS_H_
