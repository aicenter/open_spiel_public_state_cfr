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

#include "open_spiel/games/game_picker.h"

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace game_picker {
namespace {

const GameType kGameType{
    /*short_name=*/"game_picker",
    /*long_name=*/"First player picks a game to play.",
    GameType::Dynamics::kSimultaneous,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/false,
    /*parameter_specification=*/
    {{"games",
      GameParameter(std::string("matrix_mp,matrix_rps"))}},
    /*default_loadable=*/false};


std::vector<std::shared_ptr<const Game>> MakeGames(
    const std::string& game_names) {
  std::vector<std::shared_ptr<const Game>> out;
  for (std::string_view game_name : absl::StrSplit(game_names, ":")) {
    out.push_back(LoadGame(std::string(game_name)));
  }
  return out;
}

}  // namespace

GamePicker::GamePicker(std::vector<std::shared_ptr<const Game>> games,
                       Player picking_player)
    : Game(kGameType, {}),
      games_(std::move(games)),
      picking_player_(picking_player) {
#define SAVE_PROPERTY(COMPARATOR, METHOD, SAVE_AS)        \
  for(const std::shared_ptr<const Game>& game : games_) {  \
    SAVE_AS = COMPARATOR(game->METHOD(), SAVE_AS);        \
  }

  SAVE_PROPERTY(std::max, NumDistinctActions, num_distinct_actions_)
  SAVE_PROPERTY(std::max, MaxChanceOutcomes, max_chance_outcomes_)
  SAVE_PROPERTY(std::max, NumPlayers, num_players_)
  SAVE_PROPERTY(std::min, MinUtility, min_utility_)
  SAVE_PROPERTY(std::max, MaxUtility, max_utility_)
  SAVE_PROPERTY(std::max, MaxGameLength, max_game_length_)
  SAVE_PROPERTY(std::max, MaxChanceNodesInHistory, max_chance_nodes_in_history_)

#undef SAVE_PROPERTY

  if (picking_player_ >= 0) {
    num_distinct_actions_ = std::max(num_distinct_actions_,
                                     static_cast<int>(games_.size()));
  } else if (picking_player_ == kChancePlayerId) {
    max_chance_outcomes_ = std::max(max_chance_outcomes_,
                                    static_cast<int>(games_.size()));
  } else {
    SpielFatalError("Exhausted pattern match! "
                    "Player can be only chance or valid game player.");
  }
  max_game_length_++;
}


GamePicker::GamePicker(const GameParameters& params)
    : GamePicker(MakeGames(params.at("games").string_value()),
                 params.at("picking_player").int_value()) {}

std::unique_ptr<State> GamePicker::NewInitialState() const {
  return std::make_unique<GamePickerState>(
      shared_from_this(),
      std::make_unique<GamePickerDispatchingState>(shared_from_this()));
}

std::shared_ptr<Observer> GamePicker::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  SPIEL_CHECK_TRUE(iig_obs_type.has_value());
  return std::make_unique<GamePickerObserver>(
      shared_from_this(), *iig_obs_type);
}

std::vector<const Game*> GamePicker::games() const {
  std::vector<const Game*> out;
  out.reserve(games_.size());
  for (int i = 0; i < games_.size(); ++i) {
    out.push_back(games_[i].get());
  }
  return out;
}

GamePickerObserver::GamePickerObserver(
    std::shared_ptr<const Game> picking_game,
    IIGObservationType iig_obs_type)
    : Observer(true, true),
      picking_game_(std::static_pointer_cast<const GamePicker>(picking_game)),
      iig_obs_type_(iig_obs_type) {
  SPIEL_DCHECK_TRUE( // Check in debug if indeed the cast is correct.
      std::dynamic_pointer_cast<const GamePicker>(picking_game) != nullptr);

  // Let's find the largest observations.
  max_tensor_size_ = 0;
  game_tensor_sizes_.reserve(picking_game_->games().size());
  for (const Game* game : picking_game_->games()) {
    std::shared_ptr<Observer> game_observer =
        game->MakeObserver(iig_obs_type_, {});
    game_observers_.push_back(game_observer);
    Observation game_observation(*game, game_observer);
    int game_tensor_size = game_observation.Tensor().size();
    max_tensor_size_ = std::max(game_tensor_size, max_tensor_size_);
    game_tensor_sizes_.push_back(game_tensor_size);
  }
}

void GamePickerObserver::WriteTensor(const State& state,
                                     Player player,
                                     Allocator* allocator) const {
  const auto& wrapper = open_spiel::down_cast<const WrappedState&>(state);
  const auto& wrapped_state = wrapper.GetWrappedState();

  const int num_games = game_observers_.size();
  const std::vector<State::PlayerAction>& full_history = state.FullHistory();
  absl::optional<size_t> picked_game = {};
  if (!full_history.empty()) {
    picked_game = full_history[0].action;
    SPIEL_CHECK_GE(picked_game, 0);
    SPIEL_CHECK_LT(picked_game, num_games);
  }
  if (iig_obs_type_.public_info) {
    auto out = allocator->Get("picked_game", {num_games});
    if (picked_game) {
      out.at(*picked_game) = 1.;
    }
  }
  if (picked_game) {
    game_observers_[*picked_game]->WriteTensor(
        wrapped_state, player, allocator);
  } else {
    // Allocate size for the largest game observations.
    allocator->Get("game_tensor", {max_tensor_size_});
  }
}

std::string GamePickerObserver::StringFrom(
    const State& state, Player player) const {
  const auto& wrapper = open_spiel::down_cast<const WrappedState&>(state);
  const auto& wrapped_state = wrapper.GetWrappedState();

  const size_t num_games = game_observers_.size();
  const std::vector<State::PlayerAction>& full_history = state.FullHistory();
  absl::optional<size_t> picked_game = {};
  if (!full_history.empty()) {
    picked_game = full_history[0].action;
    SPIEL_CHECK_GE(picked_game, 0);
    SPIEL_CHECK_LT(picked_game, num_games);
  }
  std::string out;
  if (iig_obs_type_.public_info) {
    if (picked_game) {
      out = absl::StrCat("[picked_game=", *picked_game, "]");
    } else {
      out = absl::StrCat("[picked_game=none yet]");
    }
  }
  if (picked_game) {
    out = absl::StrCat(
        out, game_observers_[*picked_game]->StringFrom(wrapped_state, player));
  }
  return out;
}

void GamePickerState::DoApplyAction(Action action_id) {
  if (history_.empty()) {
    auto picker = std::static_pointer_cast<const GamePicker>(game_);
    SPIEL_CHECK_GE(action_id, 0);
    SPIEL_CHECK_LT(action_id, picker->num_games());
    state_ = picker->select_game(action_id)->NewInitialState();
  } else {
    state_->ApplyAction(action_id);
  }
}

std::unique_ptr<State> GamePickerState::Clone() const {
  return std::make_unique<GamePickerState>(*this);
}

std::shared_ptr<const Game> MakeGamePicker(
    const std::vector<std::shared_ptr<const Game>>& games,
    Player picking_player) {
  return std::make_shared<GamePicker>(games, picking_player);
}

const GamePicker* GamePickerDispatchingState::game_picker() const {
  SPIEL_DCHECK_TRUE(dynamic_cast<const GamePicker*>(game_.get()));
  return dynamic_cast<const GamePicker*>(game_.get());
}

Player GamePickerDispatchingState::CurrentPlayer() const {
  return game_picker()->picking_player();
}

std::vector<Action> GamePickerDispatchingState::LegalActions(
    Player player) const {
  if (player != game_picker()->picking_player()) {
    return {};
  }
  std::vector<Action> out(game_picker()->num_games(), 0);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

std::vector<Action> GamePickerDispatchingState::LegalActions() const {
  return LegalActions(game_picker()->picking_player());
}

ActionsAndProbs GamePickerDispatchingState::ChanceOutcomes() const {
  auto picker = game_picker();
  if (picker->picking_player() == kChancePlayerId) {
    const size_t n = game_picker()->num_games();
    ActionsAndProbs out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
      out.push_back({i, 1. / n});
    }
    return out;
  } else {
    return {};
  }
}

std::vector<Action> GamePickerDispatchingState::LegalChanceOutcomes() const {
  if (game_picker()->picking_player() == kChancePlayerId) {
    std::vector<Action> out(game_picker()->num_games(), 0);
    std::iota(out.begin(), out.end(), 0);
    return out;
  } else {
    return {};
  }
}

std::string GamePickerDispatchingState::ActionToString(Player player,
                                                       Action action_id) const {
  return absl::StrCat("Pick game #", action_id, ": ",
                      game_picker()->games().at(
                          action_id)->GetType().short_name);
}
}  // namespace game_picker
}  // namespace open_spiel
