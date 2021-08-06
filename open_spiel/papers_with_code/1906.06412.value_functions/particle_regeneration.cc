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

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"

namespace open_spiel {
namespace papers_with_code {

void ParticleGenerator::SetPublicOutcomes(const std::vector<int>& outcomes) {
  ResetModel(/*num_bets=*/outcomes.size());

  for (int i = 0; i < num_bets_; ++i) {
    opr::sat::IntVar& a = played_[0][i];
    opr::sat::IntVar& b = played_[1][i];
    switch(outcomes[i]) {
      case  0: cp_model_->AddEquality(a, b);    break;  // draw: a == b
      case  1: cp_model_->AddGreaterThan(a, b); break;  // win : a  > b
      case -1: cp_model_->AddLessThan(a, b);    break;  // lose: a  < b
      default: SpielFatalError("Unknown outcome");
    }
  }
}

void ParticleGenerator::SetPublicState(const Observation& public_state) {
  // Build up public outcomes.
  const auto point_cards = public_state.Tensor("point_card_sequence");
  const int num_bets = std::accumulate(point_cards.begin(),
                                       point_cards.end(), -1);
  ConstDimensionedSpan wins = public_state.GetConstSpan("win_sequence");
  ConstDimensionedSpan ties = public_state.GetConstSpan("tie_sequence");
  std::vector<int> outcomes(num_bets, 0);
  for (int i = 0; i < num_bets; ++i) {
    if(ties.at(i)) outcomes[i] = 0;
    else if (wins.at(i, 0)) outcomes[i] = 1;
    else if (wins.at(i, 1)) outcomes[i] = -1;
    else SpielFatalError("Unknown outcome");
  }
  SetPublicOutcomes(outcomes);
}

void ParticleGenerator::SetInfoState(const Observation& infostate,
                                     Player player_hand) {
  SetPublicState(infostate);
  current_player_ = player_hand;
  ConstDimensionedSpan bets = infostate.GetConstSpan("player_action_sequence");
  for (int i = 0; i < num_bets_; ++i) {
    // Set player's actions.
    int card = -1;
    for (int c = 0; c < game_->NumCards(); ++c) {
      if (bets.at(i, c)) card = c;
    }
    SPIEL_CHECK_NE(card, -1);
    cp_model_->AddEquality(played_[player_hand][i], card);
  }
}

std::unique_ptr<ParticleSet> ParticleGenerator::GenerateParticles(
    int max_particles, int max_rejection_cnt) {

  auto set = std::make_unique<ParticleSet>();
  if (num_bets_ == 0) {  // Initial state -- no actions made yet.
    set->particles.push_back(std::vector<Action>{});
    return set;
  }

  auto card_dist = std::uniform_int_distribution<int>(1, game_->NumCards());
  auto player_dist = std::uniform_int_distribution<int>(0, 1);
  auto dir_dist = std::uniform_int_distribution<int>(0, num_bets_);
  int num_rejected = 0;

  while (set->particles.size() < max_particles
         && num_rejected < max_rejection_cnt) {
    opr::sat::CpModelBuilder rnd_model = *cp_model_;

    for (int i = 0; i < num_bets_; ++i) {
      int diverse_player;
      if (current_player_ == kInvalidPlayer) {
        diverse_player = player_dist(rnd_gen_);
      } else {
        diverse_player = 1 - current_player_;
      }

      // Add random constraints to generate diverse solutions.
      // Without this we would get exactly the same solution each call.
      int card = card_dist(rnd_gen_);
      int dir = dir_dist(rnd_gen_);
      opr::sat::IntVar& a = played_[diverse_player][i];
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
      history.reserve(num_bets_ * 2);
      for (int j = 0; j < num_bets_; ++j) {
        // -1 due to 0-based indexing in the goofspiel game implementation.
        history.push_back(SolutionIntegerValue(response, played_[0][j]));
        history.push_back(SolutionIntegerValue(response, played_[1][j]));
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

void ParticleGenerator::ResetModel(int num_bets) {
  SPIEL_CHECK_GE(num_bets, 0);
  SPIEL_CHECK_LE(num_bets, game_->NumCards());
  cp_model_ = std::make_unique<opr::sat::CpModelBuilder>();
  num_bets_ = num_bets;
  current_player_ = kInvalidPlayer;

  // Init variables.
  for (int pl = 0; pl < 2; ++pl) {
    played_[pl].clear();
    for (int i = 0; i < num_bets; ++i) {
      played_[pl].push_back(cp_model_->NewIntVar(cards_));
    }
  }
  // Players can play each card only once.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < num_bets; ++i) {
      for (int j = i + 1; j < num_bets; ++j) {
        cp_model_->AddNotEqual(played_[pl][i], played_[pl][j]);
      }
    }
  }
}

}  // papers_with_code
}  // open_spiel
