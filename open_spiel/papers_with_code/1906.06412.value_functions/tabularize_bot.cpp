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

#include "open_spiel/papers_with_code/1906.06412.value_functions/tabularize_bot.h"

#include <memory>
#include <utility>

#include "open_spiel/action_view.h"
#include "open_spiel/algorithms/infostate_tree.h"

namespace open_spiel {
namespace papers_with_code {

using algorithms::InfostateNode;

void RecursivelySavePolicyForInfostate(Bot* bot,
                                       InfostateNode* node,
                                       TabularPolicy* policy) {
  if (node->type() == algorithms::kTerminalInfostateNode) return;

  // Skip filler nodes (used for balancing the tree).
  if (!node->corresponding_states().empty()) {
    // Fetch a corresponding state for the node
    State* a_state = node->corresponding_states().at(0).get();
    SPIEL_CHECK_TRUE(a_state);

    std::pair<ActionsAndProbs, Action> step = bot->StepWithPolicy(*a_state);
    if (node->type() == algorithms::kDecisionInfostateNode) {
      policy->SetStatePolicy(node->infostate_string(), step.first);
    } else {
      SPIEL_CHECK_TRUE(step.first.empty());
    }
  }

  for (InfostateNode* child : node->children()) {
    std::unique_ptr<Bot> new_bot = bot->Clone();
    RecursivelySavePolicyForInfostate(new_bot.get(), child, policy);
  }
}

std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, Player player, const Game& game) {
  auto tree = algorithms::MakeInfostateTree(game, player,
                                            algorithms::kNoMoveAheadLimit,
                                            algorithms::kStoreAllStatesPolicy);
  return TabularizeOnlinePolicy(bot, tree);
}

std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, std::shared_ptr<algorithms::InfostateTree> tree) {
  SPIEL_CHECK_EQ(tree->storage_policy(), algorithms::kStoreAllStatesPolicy);

  auto policy = std::make_unique<TabularPolicy>();
  std::unique_ptr<Bot> tab_bot = bot->Clone();
  RecursivelySavePolicyForInfostate(
      tab_bot.get(), tree->mutable_root(), policy.get());

  return policy;
}

}  // papers_with_code
}  // open_spiel
