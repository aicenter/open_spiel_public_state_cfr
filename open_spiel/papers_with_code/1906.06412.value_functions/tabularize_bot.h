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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_BOT_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_BOT_

#include "policy.h"
#include "bot.h"

namespace open_spiel {
    namespace papers_with_code {
        namespace tabularize_bot{
            std::shared_ptr<TabularPolicy>
            FullBotPolicy(std::unique_ptr<SherlockBot> bot, Player player, const std::shared_ptr<const Game> &game);

            void SavePolicyFromState(std::unique_ptr<SherlockBot> bot, Player player,
                                     std::unique_ptr<State> state, const std::shared_ptr<TabularPolicy>& policy);
        }
    } // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_BOT_

