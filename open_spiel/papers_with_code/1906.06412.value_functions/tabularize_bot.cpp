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

namespace {

void LimitPolicy(ActionsAndProbs* policy, int max_actions) {
  std::sort(policy->begin(), policy->end(),
            [](std::pair<Action, double>& a, std::pair<Action, double>& b) {
                if (a.second != b.second) return a.second > b.second;
                // Special-case for equal probs (typicall uniform strategies).
                // Prefer higher actions (goofspiel cards).
                return a.first > a.second;
            });
  int n = std::min(max_actions, (int) policy->size());
  double normalize = 0.;
  for (int i = 0; i < n; ++i) {
    normalize += (*policy)[i].second;
  }
  for (int i = 0; i < policy->size(); ++i) {
    if (i < n) (*policy)[i].second /= normalize;
    else (*policy)[i].second = 0.;
  }
  // Legal actions are sorted -- the mapping to the infostate tree policy
  // relies on this invariant.
  std::sort(policy->begin(), policy->end(),
            [](std::pair<Action, double>& a, std::pair<Action, double>& b) {
    return a.first < b.first;
  });
}

}  // namespace

void RecursivelySavePolicyForInfostate(Bot* bot,
                                       InfostateNode* node,
                                       TabularPolicy* policy,
                                       absl::optional<int> max_actions) {
  if (node->type() == algorithms::kTerminalInfostateNode) return;

  std::pair<ActionsAndProbs, Action> step_policy;

  // Skip filler nodes (used for balancing the tree).
  if (!node->corresponding_states().empty()) {
    // Fetch a corresponding state for the node
    State* a_state = node->corresponding_states().at(0).get();
    SPIEL_CHECK_TRUE(a_state);

    step_policy = bot->StepWithPolicy(*a_state);
    if (node->type() == algorithms::kDecisionInfostateNode) {
      if (max_actions) LimitPolicy(&step_policy.first, *max_actions);
      policy->SetStatePolicy(node->infostate_string(), step_policy.first);
    } else {
      SPIEL_CHECK_TRUE(step_policy.first.empty());
    }
  }

  for (InfostateNode* child : node->children()) {
    if (!step_policy.first.empty()) {
      int idx = child->incoming_index();
      double prob = step_policy.first[idx].second;
      if (prob == 0) continue;
    }

    std::unique_ptr<Bot> new_bot = bot->Clone();
    RecursivelySavePolicyForInfostate(new_bot.get(), child, policy,
                                      max_actions);
  }
}

std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, Player player, const Game& game,
    absl::optional<int> max_actions) {
  auto tree = algorithms::MakeInfostateTree(game, player,
                                            algorithms::kNoMoveAheadLimit,
                                            algorithms::kStoreAllStatesPolicy);
  return TabularizeOnlinePolicy(bot, tree, max_actions);
}

std::unique_ptr<TabularPolicy> TabularizeOnlinePolicy(
    Bot* bot, std::shared_ptr<algorithms::InfostateTree> tree,
    absl::optional<int> max_actions) {
  SPIEL_CHECK_EQ(tree->storage_policy(), algorithms::kStoreAllStatesPolicy);

  auto policy = std::make_unique<TabularPolicyWithUniformDefault>();
  std::unique_ptr<Bot> tab_bot = bot->Clone();
  RecursivelySavePolicyForInfostate(
      tab_bot.get(), tree->mutable_root(), policy.get(), max_actions);

  return policy;
}

}  // papers_with_code
}  // open_spiel
