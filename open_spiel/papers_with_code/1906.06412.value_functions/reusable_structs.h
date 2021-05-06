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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_REUSABLE_STRUCTS_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_REUSABLE_STRUCTS_

#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/ortools/trunk_exploitability.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/ismcts_playthroughs.h"

namespace open_spiel {
namespace papers_with_code {

struct ReusableStructures {
  SubgameFactory* subgame_factory;
  SolverFactory* solver_factory;
  const std::shared_ptr<const PublicStateEvaluator> pbs_oracle;

  explicit ReusableStructures(
      SubgameFactory* subgame_factory,
      SolverFactory* solver_factory,
      const std::shared_ptr <const PublicStateEvaluator>& pbs_oracle)
      : subgame_factory(subgame_factory), solver_factory(solver_factory),
        pbs_oracle(pbs_oracle) {}

  // Each of these is essentially a cache of the object.
  // You can set the cache by assigning to it directly, or you can
  // call a getter and initalize the values (if they are empty).

  std::shared_ptr<Subgame> trunk;
  std::shared_ptr<Subgame> GetTrunk();

  std::unique_ptr<SubgameSolver> fixable_trunk_with_oracle;
  SubgameSolver* GetFixableTrunkWithOracle();

  std::unique_ptr<SubgameSolver> iterable_trunk_with_oracle;
  SubgameSolver* GetIterableTrunkWithOracle();

  std::unique_ptr<SubgameSolver> trunk_with_net;
  SubgameSolver* GetTrunkWithNet();

  std::unique_ptr<SequenceFormLpSpecification> sf_lp;
  SequenceFormLpSpecification* GetSfLp();

  std::unique_ptr<PublicStatesInGame> all_states;
  PublicStatesInGame* GetAllPublicStates();

  std::vector<algorithms::BanditVector> fixable_bandits_for_all_public_states;
  std::vector<algorithms::BanditVector>& GetFixableBanditsForAllPublicStates();

  std::unique_ptr<IsmctsPlaythroughs> playthroughs;
  IsmctsPlaythroughs* GetIsmctsPlaythroughs();
};


}  // papers_with_code
}  // open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_REUSABLE_STRUCTS_
