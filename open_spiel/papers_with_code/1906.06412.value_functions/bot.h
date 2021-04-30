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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_BOT_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_BOT_

#include "open_spiel/spiel_bots.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"

namespace open_spiel {
namespace papers_with_code {

using BotParameters = GameParameters;
using BotParameter = GameParameter;

std::unique_ptr<Bot> MakeSherlockBot(std::unique_ptr<SubgameFactory> subgame_factory,
                                     std::unique_ptr<SolverFactory> solver_factory,
                                     Player player_id, int seed);

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_BOT_
