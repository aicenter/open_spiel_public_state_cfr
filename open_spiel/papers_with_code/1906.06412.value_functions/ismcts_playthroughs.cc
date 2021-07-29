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


#include <game_transforms/turn_based_simultaneous_game.h>
#include "open_spiel/papers_with_code/1906.06412.value_functions/ismcts_playthroughs.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

// Hacked together for a use in IS-MCTS.
std::unique_ptr<State> GoofspielInfostateStateResampler(
    const State& state, int player_id, std::function<double()> rng) {


  // The game is after the turn-based transformation.
  auto turn_state =
      open_spiel::down_cast<const TurnBasedSimultaneousState*>(&state);
  auto turn_game =
      open_spiel::down_cast<const TurnBasedSimultaneousGame*>(state.GetGame().get());
  auto goof_game = turn_game->wrapped_game();
  auto goof_state = turn_state->SimultaneousGameState();

  auto observer = goof_game->MakeObserver(kInfoStateObsType, {});
  Observation infostate(*goof_game, observer);
  infostate.SetFrom(*goof_state, player_id);
  std::mt19937 rnd_gen(rng());

  std::unique_ptr<papers_with_code::ParticleSet> set =
      papers_with_code::GenerateParticles(infostate, player_id,
                                          /*max_particles=*/1,
                                          /*max_rejection_cnt=*/100,
                                          /*infostate_particles=*/1,
                                          rnd_gen);

  if (set->particles.empty()) {
    return state.Clone();
  } else {
    auto& h = set->particles[0].history;
    std::unique_ptr<State> sampled_state = turn_game->NewInitialState();
    for (int j = 0; j < h.size(); ++j) {
      sampled_state->ApplyAction(h[j]);
    }
    // Apply some last action if player 0 made its turn.
    if (turn_state->CurrentPlayer() == 1) {
      auto actions = sampled_state->LegalActions(Player{0});
      std::uniform_int_distribution<int> dist(0, actions.size()-1);
      sampled_state->ApplyAction(actions[dist(rnd_gen)]);
    }
    SPIEL_DCHECK_EQ(sampled_state->CurrentPlayer(), state.CurrentPlayer());
    SPIEL_DCHECK_EQ(sampled_state->FullHistory().size(),
                    state.FullHistory().size());
    SPIEL_DCHECK_EQ(sampled_state->InformationStateString(),
                    state.InformationStateString());
    return sampled_state;
  }
}

} // namespace


void IsmctsPlaythroughs::GenerateNodes(const Game& game, std::mt19937& rnd) {
  // 1. Capture visited infostates along with visit statistics.
  std::cout << "# Making IS-MCTS play ... (this takes a while)\n# ";

  int max_moves = 0;
  for (int i = 0; i < num_matches; ++i) {
    if (i % 10 == 0) std::cout << '.' << std::flush;
    std::unique_ptr<State> state = game.NewInitialState();
    while (!state->IsTerminal()) {
      Action chosen_action;
      if (state->IsChanceNode()) {
        chosen_action = SampleAction(state->ChanceOutcomes(), rnd).first;
      } else {
        chosen_action = bot->Step(*state);

        // Save all stats before next step resets them.
        for (const std::unique_ptr<algorithms::ISMCTSNode>& node : bot->node_pool()) {
          NodeStats& stats = infostate_stats[node->infostate_observation];
          stats.visits += node->total_visits;
          // Fix move numbers coming from turn_based game transform.
          stats.move_number = node->move_number / 2;
          max_moves = std::max(stats.move_number, max_moves);
          stats.player = node->player;
        }
      }
      state->ApplyAction(chosen_action);
    }

  }
  std::cout << "\n";
  std::cout << "# Visited " << infostate_stats.size()
            << " infostates with max moves " << max_moves << " \n";

  // 2. Prepare a CDFs per move number for easy sampling.
  std::cout << "# Preparing CDFs.\n";
  for (int i = 0; i <= max_moves; ++i) {
    cdfs.push_back({});

    double normalizer = 0.;
    for (const auto&[obs, stats] : infostate_stats) {
      if (stats.move_number == i && stats.visits > 0) normalizer += stats.visits;
    }

    double cumul = 0.;
    // Maintain some statistics about the pdfs
    double p_min = 1., p_max = 0., p_mean = 0., p_std = 0.;
    for (auto it = infostate_stats.begin(); it != infostate_stats.end(); it++) {
      if (it->second.move_number == i && it->second.visits > 0) {
        double p = it->second.visits / normalizer;
        SPIEL_CHECK_GT(p, 0.);
        SPIEL_CHECK_LE(p, 1.);
        cumul += p;
        cdfs[i][cumul] = it;

        p_min = std::fmin(p, p_min);
        p_max = std::fmax(p, p_max);
        // Based on Welford's online algorithm
        double delta = p - p_mean;
        p_mean += delta / cdfs[i].size();
        double delta2 = p - p_mean;
        p_std += delta * delta2;
      }
    }
    std::cout << "# Move number " << i
              << " has " << cdfs[i].size() << " entries"
              << "\tp_min="  << std::setprecision(3) << p_min
              << "\tp_max="  << std::setprecision(3) << p_max
              << "\tp_mean=" << std::setprecision(3) << p_mean
              << "\tp_std="  << std::setprecision(3) << p_std / cdfs[i].size()
              << "\n";
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
  bot->SetResampler(GoofspielInfostateStateResampler);
}

InfostateStats::iterator IsmctsPlaythroughs::SampleInfostate(
    int move_number, std::mt19937& rnd_gen) {
  std::uniform_real_distribution<double> unif(0., 1.);  // Interval [0,1)
  double p = unif(rnd_gen);
  InfostateStats::iterator it = cdfs.at(move_number).upper_bound(p)->second;
  return it;
}

}  // namespace papers_with_code
}  // namespace open_spiel




