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


#include "open_spiel/spiel_bots.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle_regeneration.h"

namespace open_spiel {
namespace papers_with_code {


class SherlockBot : public Bot {
  std::unique_ptr<SubgameFactory> subgame_factory_;
  std::unique_ptr<SolverFactory> solver_factory_;
  Player player_id_;
  std::mt19937 rnd_gen_;
  std::shared_ptr<Subgame> subgame_;

 public:
  SherlockBot(std::unique_ptr<SubgameFactory> subgame_factory,
              std::unique_ptr<SolverFactory> solver_factory,
              Player player_id, int seed)
    : subgame_factory_(std::move(subgame_factory)),
      solver_factory_(std::move(solver_factory)),
      player_id_(player_id),
      rnd_gen_(seed) {}

  Action Step(const State& state) override {
    return StepWithPolicy(state).second;
  }

  std::unordered_map<std::string, double> GetCFVs(PublicState *publicState) {
      std::unordered_map<std::string, double> CFVs;
      for (auto& infostate : publicState->nodes) {

      }
  }

  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override {
    SPIEL_CHECK_TRUE(state.IsPlayerActing(player_id_));

    if (state.MoveNumber() == 0) {
        subgame_ = subgame_factory_->MakeTrunk(1);
    }

    // These should be provided by the referee at some point,
    // not accessed from the perfect-information State.
    std::cout << "# Make observations\n";
    // infostate observations
    Observation infostate_observation(*subgame_factory_->game,subgame_factory_->infostate_observer);
    infostate_observation.SetFrom(state, player_id_);
    const std::string infostate =
        subgame_factory_->infostate_observer->StringFrom(state, player_id_);

    // public state observations
    Observation public_observation(*subgame_factory_->game,subgame_factory_->public_observer);
    public_observation.SetFrom(state, 0);
    PublicState *publicState;
    for (PublicState& pubState : subgame_->public_states) {
        if (pubState.public_tensor == public_observation) {
            publicState = &pubState;
            break;
        }
    }
    // const std::vector<Action> legal_actions = state.LegalActions(player_id_);
    // Can be just inferred from legal actions from the referee.
    const ActionsAndProbs uniform_actions = UniformStatePolicy(state);


    // TODO: Tabularization of any Bot to compute offline TabularPolicy.
    //       Bot base class will need to add a Clone() method.

    // TODO: keep particles from previous step along with beliefs.
    //       Currently can work only for one-step lookahead trees.
    std::cout << "# Generate particles for current public state\n";
    std::unique_ptr<ParticleSetPartition> partition = MakeParticleSetPartition(*publicState, pow(10,7), pow(10,-9),false,rnd_gen_);
    std::unique_ptr<ParticleSet> set = std::make_unique<ParticleSet>(partition->primary);
    //    std::unique_ptr<ParticleSet> set = GenerateParticles(
    //        infostate_observation,
    //        player_id_,
    //        subgame_factory_->max_particles,
    //        subgame_factory_->max_particles,
    //        // Make sure we always have 1 particle in the current infostate.
    //        // Using this removes the strong global consistency guarantee,
    //        // but it makes the algorithm always capable of playing the game.
    //        /*infostate_particles=*/1,
    //        rnd_gen_);
    SPIEL_CHECK_FALSE(set->particles.empty());

    //subgame_factory_->game->NewInitialState();

    // TODO: proper management of beliefs between steps. This is just
    //       a dummy initialization.
    for (auto& particle: set->particles) {
      particle.chance_reach = 1.;
      particle.player_reach[0] = 1.;
      particle.player_reach[1] = 1.;
    }

    std::cout << "# Making subgame\n";
    std::shared_ptr<Subgame> subgame = subgame_factory_->MakeSubgame(*set);
    // TODO: implement continual resolving.
    //  Update subgame's infostate trees: subgame->trees[1-player_id_]
    //  such that they begin with the choice for the opponent
    //  to follow or not into this subgame. This could be done by careful
    //  manipulation with the (already constructed) infostate tree,
    //  or with changing how the trees are constructed. Plumb this through
    //  MakeSubgame to affect infostate tree construction.

    std::cout << "# Making solver\n";
    std::unique_ptr<SubgameSolver> solver = solver_factory_->MakeSolver(subgame, nullptr, "", true);

//    // Code for opponent fixation:
//    TabularPolicy opponent_policy;  // Needs to be provided.
//    int opponent = 1 - player_id_;
//    algorithms::BanditVector& opponent_bandits = solver->bandits()[opponent];
//    for (algorithms::DecisionId id : opponent_bandits.range()) {
//      algorithms::InfostateNode* node = subgame->trees[opponent]->decision_infostate(id);
//      ActionsAndProbs infostate_policy = opponent_policy.GetStatePolicy(node->infostate_string());
//      std::vector<double> probs = GetProbs(infostate_policy);
//      auto fixable_bandit = std::make_unique<algorithms::bandits::FixableStrategy>(probs);
//      opponent_bandits[id] = std::move(fixable_bandit);
//    }

    std::cout << "# Solving!\n";
    solver->RunSimultaneousIterations(solver_factory_->cfr_iterations);

    auto policy = std::make_shared<algorithms::BanditsAveragePolicy>(
        subgame->trees, solver->bandits());
    ActionsAndProbs actions_and_probs = policy->GetStatePolicy(infostate);
    SPIEL_CHECK_FALSE(actions_and_probs.empty());

    double p = std::uniform_real_distribution<>(0., 1.)(rnd_gen_);
    std::pair<Action, double> outcome = SampleAction(actions_and_probs, p);
    return {actions_and_probs, outcome.first};
  }

  // Not implemented yet.
  void InformAction(const State& state,
                    Player player_id,
                    Action action) override {
    Bot::InformAction(state, player_id, action);
  }
  void InformActions(const State& state,
                     const std::vector<Action>& actions) override {
    Bot::InformActions(state, actions);
  }
  void Restart() override {
    Bot::Restart();
  }
  void RestartAt(const State& state) override {
    Bot::RestartAt(state);
  }
  bool ProvidesForceAction() override {
    return Bot::ProvidesForceAction();
  }
  void ForceAction(const State& state, Action action) override {
    Bot::ForceAction(state, action);
  }
  bool ProvidesPolicy() override {
    return Bot::ProvidesPolicy();
  }
  ActionsAndProbs GetPolicy(const State& state) override {
    return Bot::GetPolicy(state);
  }
};

std::unique_ptr<Bot> MakeSherlockBot(
    std::unique_ptr<SubgameFactory> subgame_factory,
    std::unique_ptr<SolverFactory> solver_factory,
    Player player_id, int seed) {
  return std::make_unique<SherlockBot>(std::move(subgame_factory),
                                       std::move(solver_factory),
                                       player_id, seed);
}

namespace {

template<typename T>
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
    bool zero_sum_regression = GetParameterValue(bot_params, "zero_sum_regression", false);
    int cfr_iterations = GetParameterValue(bot_params, "cfr_iterations", kDefaultCfrIterations);
    int max_move_ahead_limit = GetParameterValue(bot_params, "max_move_ahead_limit", 1);
    int max_particles = GetParameterValue(bot_params, "max_particles", 1000);
    std::string device_spec = GetParameterValue<std::string>(bot_params, "device", "auto");
    std::string use_bandits_for_cfr = GetParameterValue<std::string>(bot_params, "use_bandits_for_cfr", kDefaultDlCfrBandit);
    std::string save_values_policy = GetParameterValue<std::string>(bot_params, "save_values_policy", "average");
    std::string game_model = absl::StrCat(game->GetType().short_name, ".model");
    std::string load_from = GetParameterValue<std::string>(bot_params, "load_from", game_model);

    // -- Create all necessary structures --------------------------------------
    torch::Device device(device_spec);
    //
    auto subgame_factory = std::make_unique<SubgameFactory>();
    subgame_factory->game = game;
    subgame_factory->infostate_observer   = game->MakeObserver(kInfoStateObsType, {});
    subgame_factory->public_observer      = game->MakeObserver(kPublicStateObsType, {});
    subgame_factory->hand_observer        = game->MakeObserver(kHandHistoryObsType, {});
    subgame_factory->max_move_ahead_limit = max_move_ahead_limit;
    subgame_factory->max_particles        = max_particles;
    //
    std::shared_ptr<BasicDims> dims = DeduceBasicDims(
        NetArchitecture::kParticle, *game,
        subgame_factory->public_observer,
        subgame_factory->hand_observer);
    auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
    particle_dims->max_parviews = subgame_factory->max_particles * 2;
    //
    std::shared_ptr<ValueNet> model = MakeModel(
        NetArchitecture::kParticle,
        dims, num_layers, num_width, num_inputs_regression, zero_sum_regression);
    LoadNetSnapshot(model, load_from);
    model->eval();  // Set only eval mode.
    //
    auto eval_batch = std::make_shared<BatchData>(1,
                                                  dims->point_input_size(),
                                                  dims->point_output_size());
    //
    auto solver_factory = std::make_unique<SolverFactory>();
    solver_factory->cfr_iterations = cfr_iterations;
    solver_factory->use_bandits_for_cfr  = use_bandits_for_cfr;
    solver_factory->save_values_policy   = GetSaveValuesPolicy(save_values_policy);
    solver_factory->terminal_evaluator   = std::make_shared<TerminalEvaluator>();
    solver_factory->leaf_evaluator = MakeNetEvaluator(
        dims, model, eval_batch, device, nullptr, subgame_factory->hand_observer);
    //
    return MakeSherlockBot(std::move(subgame_factory),
                           std::move(solver_factory),
                           player_id, seed);
  }
};
REGISTER_SPIEL_BOT("sherlock", SherlockBotFactory);
}  // namespace

std::unique_ptr<Bot> MakeBot();

}  // namespace papers_with_code
}  // namespace open_spiel

