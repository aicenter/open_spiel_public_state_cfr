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

#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"

#include <algorithm>
#include <string>
#include <utility>
#include <memory>

#include "absl/random/random.h"

namespace open_spiel {
namespace papers_with_code {

using namespace open_spiel::algorithms;

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


void GenerateDataRandomRanges(
    Trunk* trunk, ExperienceReplay* replay,
    double prob_pure_strat, double prob_fully_mixed,
    std::mt19937& rnd_gen, bool shuffle_input, bool shuffle_output) {
  trunk->fixable_trunk_with_oracle->Reset();

  // Randomize strategy in the trunk.
  RandomizeStrategy(trunk->fixable_trunk_with_oracle->bandits(),
                    prob_pure_strat, prob_fully_mixed, rnd_gen);
  // Compute the reach probs from the trunk.
  trunk->fixable_trunk_with_oracle->UpdateReachProbs();
  // Do not call bottom-up, just evaluate leaves.
  trunk->fixable_trunk_with_oracle->EvaluateLeaves();
  // Copy the leaves values to the experience replay.
  AddExperiencesFromTrunk(
      trunk->fixable_trunk_with_oracle->public_leaves(),
      trunk->tables, *trunk->dims, replay,
      rnd_gen, shuffle_input, shuffle_output);

//  if (verbose) {
//    for (int i = 0; i < batch->batch_size; ++i) {
//      std::cout << "# Public state " << i << std::endl;
//      std::cout << "#   Inputs:  " << batch->data_at(i) << std::endl;
//      std::cout << "#   Outputs: " << batch->targets_at(i) << std::endl;
//    }
//    std::cout << "\n# ";
//  }
}

// The network should imitate DL-CFR at each iteration
// when we use this generation method.
void GenerateDataDLCfrIterations(
    Trunk* trunk, ExperienceReplay* replay, int trunk_iters,
    std::function<void(/*trunk_iter=*/int)> monitor_fn,
    std::mt19937& rnd_gen, bool shuffle_input, bool shuffle_output) {
  trunk->iterable_trunk_with_oracle->Reset();
  for (int iter = 1; iter <= trunk_iters; ++iter) {
    ++trunk->iterable_trunk_with_oracle->num_iterations_;
    trunk->iterable_trunk_with_oracle->UpdateReachProbs();
    trunk->iterable_trunk_with_oracle->EvaluateLeaves();

    AddExperiencesFromTrunk(trunk->iterable_trunk_with_oracle->public_leaves(),
                            trunk->tables, *trunk->dims, replay,
                            rnd_gen, shuffle_input, shuffle_output);
    monitor_fn(iter);

    trunk->iterable_trunk_with_oracle->UpdateTrunk();
  }
}

}  // papers_with_code
}  // open_spiel
