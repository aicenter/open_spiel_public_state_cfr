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

#include <ortools/sat/integer_search.h>
#include <ortools/sat/cp_model_loader.h>
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"

namespace open_spiel {
namespace papers_with_code {

std::unique_ptr<ParticleSet> GenerateParticles(
    Observation& infostate,
    Player player_hand,
    int max_particles,
    int max_rejection_cnt,
    int infostate_particles,
    std::mt19937& rnd_gen
) {
  const auto point_cards = infostate.Tensor("point_card_sequence");
  int num_bets = std::accumulate(point_cards.begin(), point_cards.end(), -1);

  auto set = std::make_unique<ParticleSet>();
  if (num_bets == 0) {  // Initial state -- no actions made yet.
    set->particles.push_back(std::vector<Action>{});
    return set;
  }

  int num_cards = infostate.Tensor("player_hand").size();
  DimensionedSpan wins = infostate.GetSpan("win_sequence");
  int num_turns = wins.shape[0];
  DimensionedSpan ties = infostate.GetSpan("tie_sequence");
  DimensionedSpan player_action_sequence = infostate.GetSpan("player_action_sequence");

  SPIEL_CHECK_LE(infostate_particles, max_particles);
  SPIEL_CHECK_GE(num_bets, 1);
  SPIEL_CHECK_LE(num_bets, num_cards);
  SPIEL_CHECK_EQ(wins.shape[0], num_turns);
  SPIEL_CHECK_EQ(ties.shape[0], num_turns);

  opr::sat::CpModelBuilder cp_model;
  // Init variables.
  const opr::Domain cards(0, num_cards-1);
  std::array<std::vector<opr::sat::IntVar>, 2> played;
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < num_bets; ++i) {
      played[pl].push_back(cp_model.NewIntVar(cards));
    }
  }
  // Players can play each card only once.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < num_bets; ++i) {
      for (int j = i + 1; j < num_bets; ++j) {
        cp_model.AddNotEqual(played[pl][i], played[pl][j]);
      }
    }
  }
  // Encode the outcome constraints.
  for (int i = 0; i < num_bets; ++i) {
    opr::sat::IntVar& a = played[0][i];
    opr::sat::IntVar& b = played[1][i];
    if (ties.at(i)) cp_model.AddEquality(a, b);  // draw: a == b
    else if (wins.at(i, 0)) cp_model.AddGreaterThan(a, b);  // win : a  > b
    else if (wins.at(i, 1)) cp_model.AddLessThan(a, b);     // lose: a  < b
    else SpielFatalError("Unknown outcome");
  }

  // Store solution.
  auto card_dist = std::uniform_int_distribution<int>(1, num_cards);
  auto player_dist = std::uniform_int_distribution<int>(0, 1);
  auto dir_dist = std::uniform_int_distribution<int>(0, num_bets);

  int num_rejected = 0;
  while (set->particles.size() < max_particles
      && num_rejected < max_rejection_cnt) {
    opr::sat::CpModelBuilder rnd_model = cp_model;

    for (int i = 0; i < num_bets; ++i) {
      int diverse_player;
      if (set->particles.size() < infostate_particles) {
        // Set player's actions.
        int card = -1;
        for (int c  = 0; c < num_cards; ++c) {
          if (player_action_sequence.at(i, c)) card = c;
        }
        SPIEL_CHECK_GE(card, 0);
        SPIEL_CHECK_LT(card, num_cards);
        rnd_model.AddEquality(played[player_hand][i], card);

        // Randomize only through the opponent.
        diverse_player = 1 - player_hand;
      } else {
        diverse_player = player_dist(rnd_gen);
      }

      // Add random constraints to generate diverse solutions.
      // Without this we would get exactly the same solution each call.
      int card = card_dist(rnd_gen);
      int dir = dir_dist(rnd_gen);
      opr::sat::IntVar& a = played[diverse_player][i];
      // TODO: make more tuning of the constraints so that we have
      //       better diversity. Now from 1000 particles ~500 begin with
      //       first action 1
      if      (dir == 0) rnd_model.AddLessOrEqual(a, card);
      else if (dir == 1) rnd_model.AddEquality(a, card);
      else               rnd_model.AddGreaterOrEqual(a, card);
    }

    // Solve.
    opr::sat::CpSolverResponse response = Solve(rnd_model.Build());
    if (response.status() == opr::sat::OPTIMAL) {
      std::vector<Action> history;
      history.reserve(num_bets * 2);
      for (int j = 0; j < num_bets; ++j) {
        // -1 due to 0-based indexing in the goofspiel game implementation.
        history.push_back(SolutionIntegerValue(response, played[0][j]));
        history.push_back(SolutionIntegerValue(response, played[1][j]));
      }
      // (Possibly) increases particles size.
      if (set->has(history)) {
        ++num_rejected;
      } else {
        set->add(history);
      }
    } else {
      ++num_rejected;
    }
  }

  SPIEL_CHECK_TRUE(set->particles.size() == max_particles
                  || num_rejected == max_rejection_cnt);
  return set;
}

}  // papers_with_code
}  // open_spiel
