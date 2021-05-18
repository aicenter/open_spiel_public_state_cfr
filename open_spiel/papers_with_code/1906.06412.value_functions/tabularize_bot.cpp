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

#include "tabularize_bot.h"
#include <memory>
#include <utility>
#include "action_view.h"

namespace open_spiel {
    namespace papers_with_code {
        namespace tabularize_bot{
            std::shared_ptr<TabularPolicy> FullBotPolicy(std::unique_ptr<SherlockBot> bot, Player player,
                    const std::shared_ptr<const Game>& game) {
                std::shared_ptr<TabularPolicy> policy = std::make_shared<TabularPolicy>();
                SavePolicyFromState(std::move(bot), player, game->NewInitialState(), policy);
                return policy;
            }

            void SavePolicyFromState(std::unique_ptr<SherlockBot> bot, Player player,
                                     std::unique_ptr<State> state, const std::shared_ptr<TabularPolicy>& policy) {
                std::pair<ActionsAndProbs, Action> step = bot->StepWithPolicy(*state);
                if(state->IsPlayerActing(player)) {
                    policy->SetStatePolicy(state->InformationStateString(player), step.first);
                } else {
                    SPIEL_CHECK_TRUE(step.first.empty());
                }
                if (state->IsPlayerNode() || state->IsSimultaneousNode()) {
                    const ActionView action_view(*state);
                    for(Action action : action_view.flat_joint_actions()) {
                        std::unique_ptr<SherlockBot> new_bot = std::make_unique<SherlockBot>(*bot);
                        std::unique_ptr<State> child = state->Child(action);
                        SavePolicyFromState(std::move(new_bot), player, std::move(child), policy);
                    }
                } else if(state->IsChanceNode()){
                    for(Action action : state->LegalChanceOutcomes()) {
                        std::unique_ptr<SherlockBot> new_bot = std::make_unique<SherlockBot>(*bot);
                        std::unique_ptr<State> child = state->Child(action);
                        SavePolicyFromState(std::move(new_bot), player, std::move(child), policy);
                    }
                }
            }
        }
    }
}