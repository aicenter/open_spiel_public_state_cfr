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


#include "open_spiel/papers_with_code/1906.06412.value_functions/ismcts_playthroughs.h"

namespace open_spiel {
namespace papers_with_code {


void IsmctsPlaythroughs::GenerateNodes(const Game& game, std::mt19937& rnd) {
  // 1. Capture visited infostates along with visit statistics.
  std::cout << "# Making IS-MCTS play." << "\n# ";
  for (int i = 0; i < num_matches; ++i) {
    if (i % 1 == 0) std::cout << '.' << std::flush;
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      Action chosen_action = kInvalidAction;
      if (state->IsChanceNode()) {
        chosen_action = SampleAction(state->ChanceOutcomes(), rnd).first;
      } else {
        chosen_action = bot->Step(*state);

        // Save all stats before next step resets them.
        for (const std::unique_ptr<algorithms::ISMCTSNode>& node : bot->node_pool()) {
          NodeStats& stats = infostate_stats[node->infostate_observation];
          stats.visits += node->total_visits;
          // Fix move numbers coming from turn_based game transform.
          stats.move_number = node->move_number % 2;
          stats.player = node->player;
        }
      }
      state->ApplyAction(chosen_action);
    }

  }
  std::cout << "\n";
  std::cout << "# Visited " << infostate_stats.size() << " infostates\n";

  // 2. Prepare a CDF for easy sampling.
  double normalizer = 0.;
  for (const auto&[obs, stats] : infostate_stats) {
    normalizer += stats.visits;
  }

  double cumul = 0.;
  for (auto it = infostate_stats.begin(); it != infostate_stats.end(); it++) {
    double p = it->second.visits / normalizer;
    SPIEL_CHECK_GT(p, 0.);
    SPIEL_CHECK_LE(p, 1.);
    cumul += p;
    cdf[cumul] = it;
  }
}

void IsmctsPlaythroughs::MakeBot(std::mt19937& rnd_gen) {
  auto evaluator = std::make_shared<algorithms::RandomRolloutEvaluator>(
      /*n_rollouts=*/1, /*seed=*/rnd_gen());
  bot = std::make_unique<algorithms::ISMCTSBot>(
      /*seed=*/rnd_gen(), evaluator, uct_c,
               max_simulations, algorithms::kUnlimitedNumWorldSamples, policy_type,
      /*use_observation_string=*/false,
      /*allow_inconsistent_action_sets=*/false);
}

Observation& IsmctsPlaythroughs::SampleInfostate(
    std::mt19937& rnd_gen) {
  std::uniform_real_distribution<double> unif(0., 1.);  // Interval [0,1)
  double p = unif(rnd_gen);
  InfostateStats::iterator it = cdf.upper_bound(p)->second;
  return const_cast<Observation&>(it->first);
}

}  // namespace papers_with_code
}  // namespace open_spiel




