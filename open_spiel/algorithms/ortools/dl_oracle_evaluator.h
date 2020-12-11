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


#include "open_spiel/algorithms/history_tree.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

struct OraclePublicStateContext : public dlcfr::PublicStateContext {
  const std::vector<HistoryTree> subgame_root_histories;
  const std::map<
      /* history string */ std::string, // stringified std::vector<Action>
      /* range index of player */std::array<size_t, 2>> subgame_range_indexing;

  // Specifications are not const because they will be refined for the value
  // computation, for each public state evaluation.
  std::array<SequenceFormLpSpecification, 2> specifications;

  OraclePublicStateContext(
      std::array<SequenceFormLpSpecification, 2> specifications,
      std::vector<HistoryTree> root_histories,
      std::map<std::string, std::array<size_t, 2>> subgame_range_indexing)
      : subgame_root_histories(std::move(root_histories)),
        subgame_range_indexing(std::move(subgame_range_indexing)),
        specifications(std::move(specifications)) {}
};

struct OracleEvaluator : public dlcfr::LeafEvaluator {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;
  OracleEvaluator(std::shared_ptr<const Game> game,
                  std::shared_ptr<Observer> infostate_observer);
  std::unique_ptr<dlcfr::PublicStateContext> CreateContext(
      const dlcfr::LeafPublicState& leaf_state) const override;
  void EvaluatePublicState(dlcfr::LeafPublicState* s,
                           dlcfr::PublicStateContext* context) const override;
};

// GLOP may have some numerical stability issues for the formulated LPs!
double ComputeRootValueWhileFixingStrategy(
    SequenceFormLpSpecification* solver, const Policy& fixed_policy,
    Player fixed_player);

// Based on Proposition 3.11 in the value functions paper [1], as the average of
// the individual player exploitabilities -- we don't need to know the game
// value.
double TrunkExploitability(SequenceFormLpSpecification* solver,
                           const Policy& trunk_policy);

// Based on Proposition 3.11 in the value functions paper [1].
// If you do not supply game value, it will be calculated automatically.
double TrunkPlayerExploitability(
    SequenceFormLpSpecification* solver, const Policy& trunk_policy, Player p,
    absl::optional<double> maybe_game_value = {});

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
