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

// This file contains implementation of a number of bandit algorithms.
// TODO problem description. (online learning)

namespace open_spiel {
namespace algorithms {
namespace bandits {

// TODO: docs
class Bandit {
 protected:
  const size_t num_actions_;
 public:
  Bandit(size_t num_actions) : num_actions_(num_actions) {
    SPIEL_CHECK_GT(num_actions_, 0);
  }
  virtual ~Bandit() = default;
  size_t num_actions() const { return num_actions_; }

  virtual const std::vector<double>& NextStrategy() = 0;
  virtual void ObserveLoss(absl::Span<const double> loss) = 0;
  virtual void Reset() = 0;

  virtual bool uses_average_strategy() { return false; }
  virtual std::vector<double> AverageStrategy() {
    SpielFatalError("AverageStrategy() is not implemented.");
  }

  virtual bool uses_predictions() { return false; }
  virtual void ObservePrediction(absl::Span<const double> prediction) {
    SpielFatalError("ObservePrediction() is not implemented.");
  }

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
  std::vector<double> current_strategy_;
 public:
  RegretMatching(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  const std::vector<double>& NextStrategy() override;
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
  std::vector<double> current_strategy_;
  size_t time_;
 public:
  RegretMatchingPlus(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  const std::vector<double>& NextStrategy() override;
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

  const std::vector<double>& NextStrategy() override;
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

  const std::vector<double>& NextStrategy() override;
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

  const std::vector<double>& NextStrategy() override;
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
      std::function<double(std::vector<double>/*weight*/)> regularizer);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
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

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class OptimisticMirrorDescent final : public Bandit {
 public:
  OptimisticMirrorDescent(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class PredictiveOptimisticMirrorDescent final : public Bandit {
 public:
  PredictiveOptimisticMirrorDescent(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Exp3  final : public Bandit {
 public:
  Exp3(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Exp4  final : public Bandit {
 public:
  Exp4(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// Solving Imperfect-Information Gamesvia Discounted Regret Minimization
// Noam Brown, Tuomas Sandholm
// https://arxiv.org/pdf/1809.04040v3.pdf
class DiscountedRegretMatching final : public Bandit {
 public:
  DiscountedRegretMatching(size_t num_actions, double alpha, double beta, double gamma);
  bool uses_average_strategy() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class Hedge final : public Bandit {
 public:
  Hedge(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

// https://arxiv.org/pdf/1507.00407.pdf
class OptimisticHedge final : public Bandit {
 public:
  OptimisticHedge(size_t num_actions);
  bool uses_average_strategy() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class UpperConfidenceBounds  final : public Bandit {
 public:
  UpperConfidenceBounds(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};

class EpsGreedy  final : public Bandit {
 public:
  EpsGreedy(size_t num_actions);
  bool uses_average_strategy() override { return true; }
  bool uses_predictions() override { return true; }

  const std::vector<double>& NextStrategy() override;
  void ObserveLoss(absl::Span<const double> loss) override;
  std::vector<double> AverageStrategy() override;
  void Reset() override;
};


}  // namespace bandits
}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_BANDITS_H_
