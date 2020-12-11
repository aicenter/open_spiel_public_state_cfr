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

#ifndef OPEN_SPIEL_GAMES_GAME_PICKER_H_
#define OPEN_SPIEL_GAMES_GAME_PICKER_H_

#include <memory>
#include <string>

#include "open_spiel/spiel.h"
#include "open_spiel/game_transforms/game_wrapper.h"

namespace open_spiel {
namespace game_picker {

class GamePicker;

class GamePickerDispatchingState : public State {
 public:
  GamePickerDispatchingState(std::shared_ptr<const Game> game)
      : State(game) {}
  GamePickerDispatchingState(const GamePickerDispatchingState& other)
      : State(other) {}
  const GamePicker* game_picker() const;

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions(Player player) const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override { return "Picking game."; }
  bool IsTerminal() const override { return false; }
  std::vector<double> Rewards() const override {
    return std::vector<double>(NumPlayers(), 0);
  }
  std::vector<double> Returns() const override {
    return std::vector<double>(NumPlayers(), 0);
  }
  std::string InformationStateString(Player player) const override {
    SpielFatalError("Use the Observer.");
  }
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override {
    SpielFatalError("Use the Observer.");
  }
  std::string ObservationString(Player player) const override {
    SpielFatalError("Use the Observer.");
  }
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override {
    SpielFatalError("Use the Observer.");
  }
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<GamePickerDispatchingState>(*this);
  }
  ActionsAndProbs ChanceOutcomes() const override;
  std::vector<Action> LegalChanceOutcomes() const override;

 protected:
  void DoApplyAction(Action action_id) override {
    SpielFatalError("Cannot apply action directly, use GamePickerState");
  }
  void DoApplyActions(const std::vector<Action>& actions) override {
    SpielFatalError("Cannot apply action directly, use GamePickerState");
  }
};

class GamePickerState : public WrappedState {
 public:
  GamePickerState(
      std::shared_ptr<const Game> game, std::unique_ptr<State> state)
      : WrappedState(std::move(game), std::move(state)) {}
  GamePickerState(const GamePickerState& other) : WrappedState(other) {}
  std::unique_ptr<State> Clone() const override;
 protected:
  void DoApplyAction(Action action_id) override;
};

class GamePicker : public Game {
 public:
  explicit GamePicker(std::vector<std::shared_ptr<const Game>> games,
                      Player picking_player);
  explicit GamePicker(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override;
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
  std::vector<const Game*> games() const;
  const Game* select_game(int game_index) const {
    return games_.at(game_index).get();
  }
  size_t num_games() const { return games_.size(); }
  Player picking_player() const { return picking_player_; }

  int NumDistinctActions() const override { return num_distinct_actions_; }
  int MaxChanceOutcomes() const override { return max_chance_outcomes_; }
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return min_utility_; }
  double MaxUtility() const override { return max_utility_; }
  int MaxGameLength() const override { return max_game_length_; }
  int MaxChanceNodesInHistory() const override {
    return max_chance_nodes_in_history_;
  }

 private:
  const std::vector<std::shared_ptr<const Game>> games_;
  const Player picking_player_;
  int num_distinct_actions_;
  int max_chance_outcomes_;
  int num_players_;
  double min_utility_;
  double max_utility_;
  int max_game_length_;
  int max_chance_nodes_in_history_;
};

class GamePickerObserver : public Observer {
  const std::shared_ptr<const GamePicker> picking_game_;
  const IIGObservationType iig_obs_type_;
  size_t max_tensor_size_;
  std::vector<size_t> game_tensor_sizes_;
  std::vector<std::shared_ptr<Observer>> game_observers_;
 public:
  GamePickerObserver(std::shared_ptr<const Game> picking_game,
                     IIGObservationType iig_obs_type);
  void WriteTensor(const State& state, Player player,
                   Allocator* allocator) const override;
  std::string StringFrom(const State& state, Player player) const override;
};

std::shared_ptr<const Game> MakeGamePicker(
    const std::vector<std::shared_ptr<const Game>>& games,
    Player picking_player = kChancePlayerId);

}  // namespace game_picker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_GAME_PICKER_H_
