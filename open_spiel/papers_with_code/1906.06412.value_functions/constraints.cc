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

#include "open_spiel/algorithms/ortools/trunk_exploitability.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/constraints.h"

namespace open_spiel {
namespace papers_with_code {

SafeResolvingConstraints GetSafeResolvingConstraints(const std::string& cf) {
  if (cf == "average") return SafeResolvingConstraints::kAverageOfCurrentValues;
  else if (cf == "oracle") return SafeResolvingConstraints::kOracleConstraints;
  else
    SpielFatalError("Exhausted pattern match! "
                    "SafeResolvingConstraints not recognized.");
}

std::unordered_map<std::string, double> ComputeOracleConstraints(
    const PublicState& state,
    Player opponent,
    const Policy& player_past_policy) {
  Player player = 1 - opponent;

  algorithms::ortools::SequenceFormLpSpecification spec(*state.game());
  spec.SpecifyLinearProgram(player);
  algorithms::ortools::RecursivelyRefineSpecFixStrategyWithPolicy(
      spec.trees()[player]->mutable_root(), player_past_policy, &spec);
  spec.Solve();

  std::unordered_map<std::string, double> CFVs;
  for (int j = 0; j < state.nodes[opponent].size(); j++) {
    // Can't use the InfostateNode directly: they belong to distinct trees!
    std::string infostate_string =
        state.nodes[opponent][j]->infostate_string();
    const algorithms::InfostateNode* node =
        spec.trees()[opponent]->NodeFromInfostateString(infostate_string);
    SPIEL_CHECK_TRUE(node);

    double cfv = spec.node_spec().at(node).var_cf_value->solution_value();
    CFVs.emplace(infostate_string, cfv);
  }

  return CFVs;
}

}  // namespace papers_with_code
}  // namespace open_spiel

