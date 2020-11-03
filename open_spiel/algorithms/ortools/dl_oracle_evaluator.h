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

struct OracleEvaluator : public dlcfr::LeafEvaluator {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> infostate_observer;
  OracleEvaluator(std::shared_ptr<const Game> game,
                  std::shared_ptr<Observer> infostate_observer);
  void EvaluatePublicState(dlcfr::LeafPublicState* s,
                           dlcfr::PublicStateContext* context) const override;
};

}  // namespace ortools
}  // namespace algorithms
}  // namespace open_spiel
