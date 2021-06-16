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

#include "spiel.h"
#include "policy.h"
#include "tabularize_bot.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void ComparePolicySize(const std::shared_ptr<TabularPolicy>& joint_policy,
                       const std::shared_ptr<const Game>& game) {
  TabularPolicy full_policy = GetUniformPolicy(*game);

  SPIEL_CHECK_EQ(full_policy.PolicyTable().size(),
                 joint_policy->PolicyTable().size());

  for (const auto& info_state_policy : full_policy.PolicyTable()) {
    SPIEL_CHECK_EQ(info_state_policy.second.size(),
                   joint_policy->GetStatePolicy(
                       info_state_policy.first).size());
  }
}

void ComparePolicyToOnlinePlay(
    const std::shared_ptr<TabularPolicy>& joint_policy, BotParameters params,
    const std::shared_ptr<const Game>& game, int num_games) {
  std::mt19937 rnd_gen_(0);
  for (int i = 0; i < num_games; i++) {
    std::vector<std::unique_ptr<Bot>> bots;
    for (Player p = 0; p < 2; ++p) {
      bots.push_back(LoadBot("sherlock", game, p, params));
    }
    for (int player = 0; player < 2; player++) {
      std::unique_ptr<State> state = game->NewInitialState();
      while (!state->IsTerminal()) {
        std::pair<ActionsAndProbs, Action>
            step = bots[player]->StepWithPolicy(*state);

        if (state->IsPlayerActing(player)) {
          std::vector<double> online_strategy = GetProbs(step.first);
          std::vector<double> offline_strategy = GetProbs(
              joint_policy->GetStatePolicy(
                  state->InformationStateString(player)));
          for (int action_index = 0; action_index < online_strategy.size();
               action_index++) {
            SPIEL_CHECK_FLOAT_EQ(online_strategy[action_index],
                                 offline_strategy[action_index]);
          }
        }

        if (state->IsSimultaneousNode()) {
          std::vector<Action> actions = {Action(0), Action(0)};
          actions[player] = step.second;
          std::vector<Action> legal_actions = state->LegalActions(1 - player);
          int index = rnd_gen_() % legal_actions.size();
          actions[1 - player] = legal_actions[index];
          state->ApplyActions(actions);
        } else if (state->IsPlayerActing(player)) {
          state->ApplyAction(step.second);
        } else {
          auto state_policy = UniformStatePolicy(*state);
          Action action = SampleAction(state_policy, rnd_gen_).first;
          state->ApplyAction(action);
        }
      }
    }
  }
}

std::shared_ptr<TabularPolicy> CreatePolicyFromSetup(
    const BotParameters& params,
    const std::shared_ptr<const Game>& game) {
  SherlockBotFactory bot_factory = SherlockBotFactory();

  std::unique_ptr<Bot> bot_player_one = bot_factory.Create(game, Player(0),
                                                           params);
  std::shared_ptr<TabularPolicy> bot_policy_player_one =
      tabularize_bot::FullBotPolicy(std::move(bot_player_one), Player(0), *game);

  std::unique_ptr<Bot> bot_player_two = bot_factory.Create(game, Player(1),
                                                           params);
  std::shared_ptr<TabularPolicy> joint_policy =
      tabularize_bot::FullBotPolicy(std::move(bot_player_two), Player(1), *game);

  joint_policy->ImportPolicy(*bot_policy_player_one);

  return joint_policy;
}

void TestTabularPolicyGoofspielNetBot() {
  std::string current_dir = __FILE__;
  current_dir.resize(current_dir.rfind("/"));

  std::shared_ptr<const Game> game = LoadGame("goofspiel("
                                                "players=2,"
                                                "num_cards=3,"
                                                "imp_info=True,"
                                                "points_order=descending"
                                              ")");

  BotParameters params{
      {"seed", BotParameter(0)},
      {"num_layers", BotParameter(5)},
      {"num_width", BotParameter(5)},
      {"num_inputs_regression", BotParameter(8)},
      {"cfr_iterations", BotParameter(100)},
      {"max_move_ahead_limit", BotParameter(1)},
      {"max_particles", BotParameter(1000)},
      {"device", BotParameter("cpu")},
      {"use_bandits_for_cfr", BotParameter("RegretMatchingPlus")},
      {"save_values_policy", BotParameter("average")},
      {"zero_sum_regression", BotParameter(false)},
      {"load_from",
       BotParameter(
           absl::StrCat(current_dir, "/snapshots/iigs3/random.model"))},
  };
  params["seed"] = BotParameter(0);  // Different seeds for different outcomes.

  std::shared_ptr<TabularPolicy> joint_policy =
      CreatePolicyFromSetup(params, game);

  ComparePolicySize(joint_policy, game);
  ComparePolicyToOnlinePlay(joint_policy, params, game, 10);
}

void TestTabularPolicyKuhnCfrBot() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");

  BotParameters params{
      {"seed", BotParameter(0)},
      {"cfr_iterations", BotParameter(100)},
      {"max_move_ahead_limit", BotParameter(1)},
      {"max_particles", BotParameter(1000)},
      {"use_bandits_for_cfr", BotParameter("RegretMatchingPlus")},
      {"save_values_policy", BotParameter("average")},
      {"non_terminal_evaluator", BotParameter("cfr")},
      {"subgame_cfr_iterations", BotParameter(10)},
  };

  std::shared_ptr<TabularPolicy> joint_policy =
      CreatePolicyFromSetup(params, game);

  ComparePolicySize(joint_policy, game);
  ComparePolicyToOnlinePlay(joint_policy, params, game, 10);
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestTabularPolicyGoofspielNetBot();
  open_spiel::papers_with_code::TestTabularPolicyKuhnCfrBot();
}
