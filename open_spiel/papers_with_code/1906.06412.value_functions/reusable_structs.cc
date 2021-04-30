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

#include "open_spiel/papers_with_code/1906.06412.value_functions/reusable_structs.h"

namespace open_spiel {
namespace papers_with_code {

std::shared_ptr<Subgame> ReusableStructures::GetTrunk() {
  if (!trunk) {
    trunk = subgame_factory->MakeTrunk();
  }
  return trunk;
}

SubgameSolver* ReusableStructures::GetFixableTrunkWithOracle() {
  if (!fixable_trunk_with_oracle) {
    fixable_trunk_with_oracle = solver_factory->MakeSolver(
        GetTrunk(), pbs_oracle, "FixableStrategy");
  }
  return fixable_trunk_with_oracle.get();
}

SubgameSolver* ReusableStructures::GetIterableTrunkWithOracle() {
  if (!iterable_trunk_with_oracle) {
    iterable_trunk_with_oracle = solver_factory->MakeSolver(
        GetTrunk(), pbs_oracle, solver_factory->use_bandits_for_cfr);
  }
  return iterable_trunk_with_oracle.get();
}

SubgameSolver* ReusableStructures::GetTrunkWithNet() {
  if (!trunk_with_net) {
    trunk_with_net = solver_factory->MakeSolver(
        GetTrunk(), solver_factory->leaf_evaluator,
        solver_factory->use_bandits_for_cfr);
  }
  return trunk_with_net.get();
}

SequenceFormLpSpecification* ReusableStructures::GetSfLp() {
  if (!sf_lp) {
    sf_lp = std::make_unique<SequenceFormLpSpecification>(
        *subgame_factory->game, "CLP", /*return_nan_if_non_optimal=*/true);
  }
  return sf_lp.get();
}

PublicStatesInGame* ReusableStructures::GetAllPublicStates() {
  if (!all_states) {
    all_states = MakeAllPublicStates(*subgame_factory->game);
  }
  return all_states.get();
}

std::vector<algorithms::BanditVector>&
    ReusableStructures::GetFixableBanditsForAllPublicStates() {
  if (fixable_bandits_for_all_public_states.empty()) {
    fixable_bandits_for_all_public_states =
        MakeBanditVectors(GetAllPublicStates()->infostate_trees,
                          "FixableStrategy");
  }
  return fixable_bandits_for_all_public_states;
}

}  // papers_with_code
}  // open_spiel
