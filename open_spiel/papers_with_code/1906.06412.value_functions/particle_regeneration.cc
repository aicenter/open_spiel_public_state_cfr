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

std::unique_ptr<ParticleSet> GenerateParticles(Observation& public_state,
                                               int max_particles) {
  int num_cards = public_state.Tensor("tie_sequence").size();
  const auto point_cards = public_state.Tensor("point_card_sequence");
  int bet_rounds = std::accumulate(point_cards.begin(), point_cards.end(), -1);
  DimensionedSpan wins = public_state.GetSpan("win_sequence");
  DimensionedSpan ties = public_state.GetSpan("tie_sequence");
  SPIEL_CHECK_GE(bet_rounds, 1);
  SPIEL_CHECK_LE(bet_rounds, num_cards);
  SPIEL_CHECK_EQ(wins.shape[0], num_cards);
  SPIEL_CHECK_EQ(ties.shape[0], num_cards);

  opr::sat::CpModelBuilder cp_model;
  // Init variables.
  const opr::Domain cards(1, num_cards);
  std::array<std::vector<opr::sat::IntVar>, 2> played;
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < bet_rounds; ++i) {
      played[pl].push_back(cp_model.NewIntVar(cards));
    }
  }
  // Players can play each card only once.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < bet_rounds; ++i) {
      for (int j = i + 1; j < bet_rounds; ++j) {
        cp_model.AddNotEqual(played[pl][i], played[pl][j]);
      }
    }
  }
  // Encode the outcome constraints.
  for (int i = 0; i < bet_rounds; ++i) {
    opr::sat::IntVar& a = played[0][i];
    opr::sat::IntVar& b = played[1][i];
    if (ties.at(i)) cp_model.AddEquality(a, b);  // draw: a == b
    else if (wins.at(i, 0)) cp_model.AddGreaterThan(a, b);  // win : a  > b
    else if (wins.at(i, 1)) cp_model.AddLessThan(a, b);     // lose: a  < b
    else SpielFatalError("Unknown outcome");
  }

  // Solving part.
  opr::sat::Model model;
  opr::sat::SatParameters parameters;
  parameters.set_enumerate_all_solutions(true);
  parameters.set_cp_model_presolve(false);
  model.Add(NewSatParameters(parameters));
  
  // Create an atomic Boolean that will be periodically checked by the limit.
  std::atomic<bool> stopped(false);
  model.GetOrCreate<opr::TimeLimit>()->RegisterExternalBooleanAsLimit(&stopped);

  // Store solution.
  auto set = std::make_unique<ParticleSet>();
  model.Add(opr::sat::NewFeasibleSolutionObserver(
      [&](const opr::sat::CpSolverResponse& response){
        std::vector<Action> history;
        history.reserve(bet_rounds * 2);

        for (int i = 0; i < bet_rounds; ++i) {
          history.push_back(SolutionIntegerValue(response, played[0][i]));
          history.push_back(SolutionIntegerValue(response, played[1][i]));
        }
        set->particles.emplace_back(history);

        if (set->particles.size() >= max_particles) stopped = true;
      }
  ));

  // Solve.
  SolveCpModel(cp_model.Build(), &model);
  SPIEL_CHECK_LE(set->particles.size(), max_particles);
  return set;
}

}  // papers_with_code
}  // open_spiel
