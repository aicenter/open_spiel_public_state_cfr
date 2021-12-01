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

#ifndef OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
#define OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// This file provides a vectorized implementation of CFR on top of infostate
// trees. This is intended for advanced usage and the code may not be as
// readable as a more basic algorithm. See cfr.h for a basic implementation.
//
// This code uses a preconstructed infostate trees of each player. It updates
// all infostates at a tree depth at once. While the current implementation is
// okay in efficiency (~10x faster than CFRSolver), it could be further
// improved:
//
// - Use contiguous memory blocks for storing regrets/current/cumul
//   for each action.
// - Skip subtrees that have zero reach probabilities.
// - Use stack allocation: most trees are small. If they exceed the allocated
//   limit (1024? nodes), use something like a linked list with these large
//   memory blocks.
//
// If you decide to make contributions to this code, please open up an issue
// on github first. Thank you!

namespace open_spiel {
namespace algorithms {

// Make a top-down pass on an infostate tree, using the provided local policy
// function at each decision infostate.
//
// This writes to the provided span of reach_probs to store the cumulative
// product of reach probabilities for leaf nodes. The starting values at depth 1
// must be provided externally.
void TopDown(const InfostateTree& tree, absl::Span<double> reach_probs,
             std::function<std::vector<double>(
                 DecisionId, /*current_reach=*/double)> policy_fn);

inline void TopDownCurrent(const InfostateTree& tree, BanditVector& bandits,
                           absl::Span<double> reach_probs, size_t current_time) {
  TopDown(tree, reach_probs, [&](DecisionId id, double reach_prob) {
      bandits::Bandit* bandit = bandits[id].get();
      bandit->ComputeStrategy(current_time, reach_prob);
      return bandit->current_strategy();
  });
}

inline void TopDownAverage(const InfostateTree& tree, BanditVector& bandits,
                        absl::Span<double> reach_probs) {
    TopDown(tree, reach_probs, [&](DecisionId id, double reach_prob) {
        bandits::Bandit* bandit = bandits[id].get();
        return bandit->AverageStrategy();
    });
}

// Make a bottom-up pass, starting with the current cf_values stored
// in the buffer. This loops over all depths from the bottom.
// The leaf values must be provided externally by writing to leaves_cf_values().
void BottomUp(
    const InfostateTree& tree, absl::Span<double> cf_values,
    std::function</*infostate_policy=*/std::vector<double>(
        DecisionId, /*rewards=*/absl::Span<const double>)> observe_rewards_fn);

using RewardPredictor = std::function<
    /*predictions=*/absl::Span<const double>(
        /*decision_infostate=*/DecisionId,
        /*rewards=*/absl::Span<const double>)>;

// Reward predictor which predicts identical rewards.
constexpr absl::Span<const double> IdentityPrediction(
    DecisionId id, absl::Span<const double> rewards) { return rewards; }

inline void BottomUp(
    const InfostateTree& tree, BanditVector& bandits, absl::Span<double> cf_values,
    RewardPredictor predictor = IdentityPrediction) {
  BottomUp(tree, cf_values,
           [&](DecisionId id, absl::Span<const double> rewards) {
               bandits::Bandit* bandit = bandits[id].get();
               bandit->ObserveRewards(rewards);
               if (bandit->uses_predictions()) {
                 absl::Span<const double> predictions = predictor(id, rewards);
                 bandit->ObservePrediction(predictions);
               }
               return bandit->current_strategy();
           });
}

inline void BottomUpCfBestResponse(const InfostateTree& tree,
                                   absl::Span<double> cf_values) {
  BottomUp(
      tree, cf_values,
      [](DecisionId id, absl::Span<const double> rewards) {
          size_t num_actions = rewards.size();
          auto iter_max = std::max_element(rewards.begin(), rewards.end());
          size_t response_index = std::distance(rewards.begin(), iter_max);
          auto policy = std::vector<double>(num_actions);
          policy[response_index] = 1.;
          return policy;
      }
  );
}

// Calculates the root cf value as weighted sum of the cf_values()
// (the weights are the respective ranges). If the supplied range is empty,
// it returns their sum (i.e. all weights have a value of 1).
double RootCfValue(int root_branching_factor,
                   absl::Span<const double> cf_values,
                   absl::Span<const double> range = {});

// Run vectorized CFR on the whole game or on specified trees.
class InfostateCFR {
 public:
  // Basic constructor for the whole game.
  explicit InfostateCFR(const Game& game);
  // Run CFR on the specified trees.
  explicit InfostateCFR(
      std::vector<std::shared_ptr<InfostateTree>> trees,
      std::vector<BanditVector> bandits);

  void RunSimultaneousIterations(int iterations);
  void RunAlternatingIterations(int iterations);

  // Computes the root value. If the algorithm is running on the full trees
  // of the game, this converges to the game value.
  double RootValue() const;
  std::vector<double> RootValues() const;

  void ResetCumulValues();

  std::shared_ptr<Policy> AveragePolicy();
  std::shared_ptr<Policy> CurrentPolicy();

  std::vector<BanditVector>& bandits_modifiable() { return bandits_; }
  const std::vector<BanditVector>& bandits() const { return bandits_; }
  const std::vector<std::shared_ptr<InfostateTree>>& trees() const {
    return trees_;
  }

 private:
  void PrepareTerminals();
  void PrepareRootReachProbs();
  void PrepareRootReachProbs(Player pl);
  void EvaluateLeaves();
  void EvaluateLeaves(Player pl);
  double TerminalReachProbSum();

  std::vector<std::shared_ptr<InfostateTree>> trees_;
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> terminal_permutation_;
  // Chance reach probs.
  std::vector<double> terminal_ch_reaches_;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> terminal_values_;

  // Mutable values to keep track of for each tree.
  // These have the size of largest depth of the tree (i.e. leaf nodes).
  std::vector<std::vector<double>> reach_probs_;
  std::vector<std::vector<double>> cf_values_;

  std::vector<BanditVector> bandits_;

  size_t num_iterations_ = 0;
};

}  // namespace algorithms
}  // namespace open_spiel

#endif  // OPEN_SPIEL_ALGORITHMS_INFOSTATE_CFR_H_
