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


#include "open_spiel/algorithms/infostate_dl_cfr.h"

namespace open_spiel {
namespace algorithms {
namespace ortools {

struct OraclePublicState : public dlcfr::EncodedPublicState {
  const dlcfr::LeafPublicState& public_state;
  std::array<std::vector<float>, 2> root_cfvs;
  explicit OraclePublicState(const dlcfr::LeafPublicState& s);
};

struct OracleEvaluator : public dlcfr::LeafEvaluator {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;
  OracleEvaluator(
      std::shared_ptr<const Game> game,
      std::shared_ptr<Observer> infostate_observer)
      : game(std::move(game)),
        infostate_observer(std::move(infostate_observer)) {}

  std::unique_ptr<dlcfr::EncodedPublicState> EncodeLeafPublicState(
      const dlcfr::LeafPublicState& leaf_state) const override;
  std::array<absl::Span<const float>, 2> EvaluatePublicState(
      dlcfr::EncodedPublicState* public_state,
      std::array<absl::Span<const float>, 2>) const override;
};

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
