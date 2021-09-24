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
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"

namespace open_spiel {
namespace papers_with_code {

using BotParameters = GameParameters;
using BotParameter = GameParameter;

class SherlockBot : public Bot {
  const std::shared_ptr<SubgameFactory> subgame_factory_;
  const std::shared_ptr<SolverFactory> solver_factory_;
  const Player player_id_;

  std::shared_ptr<Subgame> subgame_;
  TabularPolicy past_policy_;
  bool first_step_ = true;

 public:
  SherlockBot(std::shared_ptr<SubgameFactory> subgame_factory,
              std::shared_ptr<SolverFactory> solver_factory,
              Player player_id);

  SherlockBot(const SherlockBot& bot);
  std::unique_ptr<Bot> Clone() const override;
  Action Step(const State& state) override;
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override;
  void SetSeed(int seed);
  void InformAction(const State& state,
                    Player player_id,
                    Action action) override {
    Bot::InformAction(state, player_id, action);
  }
  void InformActions(const State& state,
                     const std::vector<Action>& actions) override {
    Bot::InformActions(state, actions);
  }
  void Restart() override;
  void RestartAt(const State& state) override { Bot::RestartAt(state); }
  bool ProvidesForceAction() override { return Bot::ProvidesForceAction(); }
  void ForceAction(const State& state, Action action) override {
    Bot::ForceAction(state, action);
  }
  bool ProvidesPolicy() override {  return Bot::ProvidesPolicy(); }
  ActionsAndProbs GetPolicy(const State& state) override {
    return Bot::GetPolicy(state);
  }
 private:
  std::mt19937& rnd_gen() { return *solver_factory_->rnd_gen; }

  void StorePastPolicy(const std::shared_ptr<algorithms::InfostateTree> tree,
                       const Policy& policy);
  std::unique_ptr<ParticleSet> PickParticles(
      const PublicState& for_state,
      const Observation& infostate_observation,
      const Observation& public_observation) const;

  std::unordered_map<std::string, double> GetOpponentCfvs(
      const PublicState& state, const TabularPolicy& past_policy) const;
};

std::unique_ptr<Bot> MakeSherlockBot(
    std::shared_ptr<SubgameFactory> subgame_factory,
    std::shared_ptr<SolverFactory> solver_factory,
    Player player_id = kDefaultPlayerId);

namespace {

template <typename T>
T GetParameterValue(const GameParameters& table,
                    const std::string& key,
                    const T& default_value);

template <>
int GetParameterValue<int>(const GameParameters& table,
                           const std::string& key,
                           const int& default_value) {
  auto it = table.find(key);
  if (it == table.end()) return default_value;
  if (!it->second.has_int_value()) {
    SpielFatalError(absl::StrCat(
        "Parameter '", key, "' should have int value"));
  }
  return it->second.int_value();
}

template <>
bool GetParameterValue<bool>(const GameParameters& table,
                             const std::string& key,
                             const bool& default_value) {
  auto it = table.find(key);
  if (it == table.end()) return default_value;
  if (!it->second.has_bool_value()) {
    SpielFatalError(absl::StrCat(
        "Parameter '", key, "' should have bool value"));
  }
  return it->second.bool_value();
}

template <>
std::string GetParameterValue<std::string>(const GameParameters& table,
                                           const std::string& key,
                                           const std::string& default_value) {
  auto it = table.find(key);
  if (it == table.end()) return default_value;
  if (!it->second.has_string_value()) {
    SpielFatalError(absl::StrCat(
        "Parameter '", key, "' should have string value"));
  }
  return it->second.string_value();
}

class SherlockBotFactory : public BotFactory {
 public:
  ~SherlockBotFactory() = default;

  bool CanPlayGame(const Game& game, Player player_id) const override {
    const GameType& type = game.GetType();
    return game.NumPlayers() == 2
        && type.utility == GameType::Utility::kZeroSum
        && type.reward_model == GameType::RewardModel::kTerminal
        && type.provides_information_state_string
        && type.provides_information_state_tensor
        && type.provides_observation_string
        && type.provides_observation_tensor;
  }

  std::unique_ptr<Bot> Create(std::shared_ptr<const Game> game,
                              Player player_id,
                              const GameParameters& bot_params) const override {
    // Extract all param values.
    int seed = GetParameterValue(bot_params, "seed", 0);
    int num_layers = GetParameterValue(bot_params, "num_layers", 3);
    int num_width = GetParameterValue(bot_params, "num_width", 3);
    int num_inputs_regression =
        GetParameterValue(bot_params, "num_inputs_regression", 128);
    bool zero_sum_regression =
        GetParameterValue(bot_params, "zero_sum_regression", false);
    bool normalize_beliefs =
        GetParameterValue(bot_params, "normalize_beliefs", false);
    SetPoolingOp set_pooling_op = GetPoolingOp(
        GetParameterValue<std::string>(bot_params, "set_pooling_op", "sum"));
    int cfr_iterations =
        GetParameterValue(bot_params, "cfr_iterations", kDefaultCfrIterations);
    int max_move_ahead_limit =
        GetParameterValue(bot_params, "max_move_ahead_limit", 1);
    int max_particles = GetParameterValue(bot_params, "max_particles", 1000);
    int subgame_cfr_iterations =
        GetParameterValue(bot_params, "subgame_cfr_iterations", 100);
    std::string device_spec =
        GetParameterValue<std::string>(bot_params, "device", "auto");
    std::string use_bandits_for_cfr =
        GetParameterValue<std::string>(bot_params, "use_bandits_for_cfr",
                                       kDefaultDlCfrBandit);
    std::string save_values_policy =
        GetParameterValue<std::string>(bot_params, "save_values_policy",
                                       "average");
    std::string non_terminal_evaluator =
        GetParameterValue<std::string>(bot_params, "non_terminal_evaluator",
                                       "net");
    std::string game_model = absl::StrCat(game->GetType().short_name, ".model");
    std::string load_from =
        GetParameterValue<std::string>(bot_params, "load_snapshot", game_model);

    // -- Create all necessary structures --------------------------------------
    auto rnd_gen = std::make_shared<std::mt19937>(seed);
    auto subgame_factory = std::make_shared<SubgameFactory>();
    subgame_factory->game = game;
    subgame_factory->infostate_observer =
        game->MakeObserver(kInfoStateObsType, {});
    subgame_factory->public_observer =
        game->MakeObserver(kPublicStateObsType, {});
    subgame_factory->hand_observer =
        game->MakeObserver(kHandHistoryObsType, {});
    subgame_factory->max_move_ahead_limit = max_move_ahead_limit;
    subgame_factory->max_particles = max_particles;
    if (game->GetType().short_name == "goofspiel") {
      auto goof_game =
          std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game);
      subgame_factory->particle_generator =
          std::make_shared<ParticleGenerator>(goof_game, rnd_gen);
    }
    //
    auto solver_factory = std::make_shared<SolverFactory>();
    solver_factory->rnd_gen = rnd_gen;
    if (non_terminal_evaluator == "net") {
      torch::Device device(device_spec);

      std::shared_ptr<BasicDims> dims = DeduceBasicDims(
          NetArchitecture::kParticle, *game,
          subgame_factory->public_observer,
          subgame_factory->hand_observer);
      auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
      particle_dims->max_parviews = subgame_factory->max_particles * 2;
      //
      auto eval_batch = std::make_shared<BatchData>(1,
                                                    dims->point_input_size(),
                                                    dims->point_output_size());
      //
      std::shared_ptr<ValueNet> model = MakeModel(
          NetArchitecture::kParticle,
          dims,
          num_layers,
          num_width,
          num_inputs_regression,
          zero_sum_regression,
          normalize_beliefs,
          set_pooling_op);
      LoadNetSnapshot(model, load_from);
      model->eval();  // Set only eval mode.
      //
      solver_factory->leaf_evaluator = MakeNetEvaluator(
          dims, model, eval_batch, device,
          rnd_gen, nullptr, subgame_factory->hand_observer);
    } else {
      std::shared_ptr<const PublicStateEvaluator> terminal_evaluator =
          MakeTerminalEvaluator();
      std::shared_ptr<Observer> public_observer =
          game->MakeObserver(kPublicStateObsType, {});
      std::shared_ptr<Observer> infostate_observer =
          game->MakeObserver(kInfoStateObsType, {});

      solver_factory->leaf_evaluator = std::make_shared<CFREvaluator>(
          game, algorithms::kNoMoveAheadLimit,
          /*leaf_evaluator=*/nullptr, terminal_evaluator,
          public_observer, infostate_observer, subgame_cfr_iterations);
    }
    solver_factory->cfr_iterations = cfr_iterations;
    solver_factory->use_bandits_for_cfr = use_bandits_for_cfr;
    solver_factory->save_values_policy = GetSaveValuesPolicy(save_values_policy);
    solver_factory->terminal_evaluator = std::make_shared<TerminalEvaluator>();
    solver_factory->safe_resolving = true;

    return MakeSherlockBot(subgame_factory, solver_factory, player_id);
  }
};
REGISTER_SPIEL_BOT("sherlock", SherlockBotFactory);
}  // namespace

std::unique_ptr<Bot> MakeBot();

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_BOT_
