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


#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"

#include <utility>
#include <vector>
#include <unordered_map>

#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

OracleEvaluator::OracleEvaluator(std::shared_ptr<const Game> game,
                                 std::shared_ptr<Observer> infostate_observer)
    : game(std::move(game)),
      infostate_observer(std::move(infostate_observer)) {}

void OracleEvaluator::EvaluatePublicState(
    dlcfr::LeafPublicState* s, dlcfr::PublicStateContext* context) const {
  SPIEL_CHECK_FALSE(context);  // Oracle doesn't use any context.

  std::array<std::unique_ptr<ortools::ZeroSumSequentialGameSolution>, 2>
      solutions;
  std::unordered_map<std::string, std::array<float, 2>> history_ranges;
  for (int pl = 0; pl < 2; ++pl) {
    std::vector<const State*> start_states;
    std::vector<float> chance_range;

    SPIEL_CHECK_EQ(s->leaf_nodes[1 - pl].size(), s->ranges[1 - pl].size());
    for (int i = 0; i < s->leaf_nodes[1 - pl].size(); ++i) {
      const InfostateNode* cfr_node = s->leaf_nodes[1 - pl][i];
      const double opponent_prob = s->ranges[1 - pl][i];
      SPIEL_DCHECK_TRUE(cfr_node->is_leaf_node());
      SPIEL_CHECK_EQ(cfr_node->corresponding_states().size(),
                     cfr_node->corresponding_chance_reach_probs().size());
      for (int j = 0; j < cfr_node->corresponding_states().size(); ++j) {
        const State* state = cfr_node->corresponding_states()[j].get();
        const double chn_prob = cfr_node->corresponding_chance_reach_probs()[j];
        history_ranges[state->HistoryString()][1 - pl] = opponent_prob;

        start_states.push_back(state);
        chance_range.push_back(chn_prob * opponent_prob);
      }
    }

    solutions[pl] = ortools::SolveZeroSumSequentialGame(
        infostate_observer, start_states, chance_range,
        /*solve_only_player=*/pl, /*collect_tabular_policy=*/true);
  }
  const std::vector<const Policy*>& policy_profile = {
      &solutions[0]->policy, &solutions[1]->policy
  };

  // Compute the root cfvs.
  double public_state_utility = 0.;
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(s->leaf_nodes[pl].size(), s->ranges[pl].size());
    SPIEL_CHECK_EQ(s->values[pl].size(), s->ranges[pl].size());

    for (int i = 0; i < s->leaf_nodes[pl].size(); ++i) {
      const InfostateNode* cfr_node = s->leaf_nodes[pl][i];
      double infostate_value = 0.;
      for (int j = 0; j < cfr_node->corresponding_states().size(); ++j) {
        const State* state = cfr_node->corresponding_states()[j].get();
        const double chn_prob = cfr_node->corresponding_chance_reach_probs()[j];
        const double
            opponent_prob = history_ranges[state->HistoryString()][1 - pl];
        const double state_utility = ExpectedReturns(
            *state, policy_profile, /*depth_limit=*/-1)[pl];

        public_state_utility += state_utility;
        infostate_value += chn_prob * opponent_prob * state_utility;
      }
      s->values[pl][i] = infostate_value;
    }
  }
  SPIEL_DCHECK_FLOAT_NEAR(public_state_utility, 0., 1e-6);
}

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
