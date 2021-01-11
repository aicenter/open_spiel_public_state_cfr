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


void RandomizeStrategy(
    std::vector<BanditVector>& bandits, std::mt19937& rnd_gen,
    double prob_pure_strat, double prob_fully_mixed) {
  const bool fully_mixed_strategy =
      std::bernoulli_distribution(prob_fully_mixed)(rnd_gen);
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : bandits[pl].range()) {
      // Randomize current policy
      bandits::Bandit* bandit = bandits[pl][id].get();
      const size_t num_actions = bandit->num_actions();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      absl::Span<double> policy = fixable_bandit->mutable_strategy();

      if (fully_mixed_strategy) {
        for (int i = 0; i < num_actions; ++i) {
          policy[i] = 1. / num_actions;
        }
        continue;
      }

      const bool single_pure_strategy =
          std::bernoulli_distribution(prob_pure_strat)(rnd_gen);
      if (single_pure_strategy) {
        const int which_action =
            std::uniform_int_distribution<>(0, num_actions - 1)(rnd_gen);
        std::fill(policy.begin(), policy.end(), 0.);
        policy[which_action] = 1.;
      } else {
        for (int i = 0; i < num_actions; ++i) {
          policy[i] = std::uniform_real_distribution<>(0., 1.)(rnd_gen);
        }
        Normalize(absl::MakeSpan(policy));
      }
    }
  }
}


void GenerateDataRandomRanges(const std::vector<dlcfr::RangeTable>& tables,
                              dlcfr::DepthLimitedCFR* fixable_trunk_with_oracle, BatchData* batch,
                              std::mt19937& rnd_gen, bool verbose) {
  fixable_trunk_with_oracle->Reset();

  // Randomize strategy in the trunk.
  RandomizeStrategy(fixable_trunk_with_oracle->bandits(), rnd_gen);
  // Compute the reach probs from the trunk.
  fixable_trunk_with_oracle->UpdateReachProbs();
  // Do not call bottom-up, just evaluate leaves.
  fixable_trunk_with_oracle->EvaluateLeaves();
  // Copy the leaves values to the batch.
  CopyRangesAndValues(fixable_trunk_with_oracle, tables, batch, verbose);

  if (verbose) {
    for (int i = 0; i < batch->batch_size; ++i) {
      std::cout << "# Public state " << i << std::endl;
      std::cout << "#   Inputs:  " << batch->data_at(i) << std::endl;
      std::cout << "#   Outputs: " << batch->targets_at(i) << std::endl;
    }
    std::cout << "\n# ";
  }
}

// The network should imitate DL-CFR at each iteration
// when we use this generation method.
void GenerateDataWithDLCfr(Trunk* trunk, std::mt19937& rnd_gen,
                           int which_iteration) {
  SPIEL_CHECK_GE(which_iteration, 1);
  trunk->iterable_trunk_with_oracle->Reset();
  trunk->iterable_trunk_with_oracle->RunSimultaneousIterations(which_iteration - 1);

  trunk->iterable_trunk_with_oracle->num_iterations_++;
  trunk->iterable_trunk_with_oracle->UpdateReachProbs();
  trunk->iterable_trunk_with_oracle->EvaluateLeaves();
  CopyRangesAndValues(trunk->iterable_trunk_with_oracle.get(), trunk->tables,
                      trunk->batch.get(), /*verbose=*/false);
}

void PrecomputeExperienceReplayForDLCfr(
    Trunk* trunk, ExperienceReplay* replay,
    ortools::SequenceFormLpSpecification* whole_game,
    const std::vector<int>& eval_iters, std::ostream& os) {
  int trunk_eval_iterations = *std::max_element(eval_iters.begin(),
                                                eval_iters.end());

  auto should_evaluate = [&](int i){
      for (auto j : eval_iters) {
        if (i == j) return true;
      }
      return false;
  };

  os << "# Computing reference exploitabilities for given trunk iterations.\n";
  trunk->iterable_trunk_with_oracle->Reset();
  double expl;
  for (int i = 1; i <= trunk_eval_iterations; ++i) {
    ++trunk->iterable_trunk_with_oracle->num_iterations_;
    trunk->iterable_trunk_with_oracle->UpdateReachProbs();
    trunk->iterable_trunk_with_oracle->EvaluateLeaves();

    if (should_evaluate(i)) {
      expl = ortools::TrunkExploitability(
          whole_game, *trunk->iterable_trunk_with_oracle->AveragePolicy());
      os << "# " << i << ": " << "expl = " << expl << std::endl;
    }
    CopyRangesAndValues(trunk->iterable_trunk_with_oracle.get(), trunk->tables,
                        trunk->batch.get(), /*verbose=*/false);
    replay->buffer.push_back(*trunk->batch);

    trunk->iterable_trunk_with_oracle->UpdateTrunk();
  }
}

}  // papers_with_code
}  // open_spiel
