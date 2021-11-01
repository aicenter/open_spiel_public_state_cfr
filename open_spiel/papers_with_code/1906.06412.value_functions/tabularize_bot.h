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

#include "open_spiel/policy.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/bot.h"

namespace open_spiel {
namespace papers_with_code {

// Create offline tabular policy from an online bot by traversing infostate
// tree of given player and saving bot's policy for that player at each decision
// point (infostate).
//
// This method assumes that the bot does not "cheat" in imperfect-information games,
// i.e. it outputs the same policy regardless of the supplied
// perfect-information state.
//
// The bot must implement the Clone() method. This is required to ensure that
// the bot's internalstate does not get corrupted while traversing the infostate
// tree.
std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, Player player, const Game& game,
    absl::optional<int> max_actions = absl::nullopt,
    absl::optional<int> max_depth = absl::nullopt);
std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, std::shared_ptr<algorithms::InfostateTree> tree,
    absl::optional<int> max_actions = absl::nullopt,
    absl::optional<int> max_depth = absl::nullopt);

} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_BOT_

