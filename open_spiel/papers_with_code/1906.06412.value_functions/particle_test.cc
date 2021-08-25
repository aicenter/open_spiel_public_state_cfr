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

#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "subgame_factory.h"

namespace open_spiel {
namespace papers_with_code {
namespace {


// Print out bunch of particles to see how diverse they are.
// Not a test yet.
void TestStatesInSupport() {
  std::shared_ptr <const Game> game = LoadGame(
      "goofspiel(imp_info=true,players=2,points_order=descending,num_turns=3,num_cards=6)");

  SequenceFormLpSpecification lp({
     algorithms::MakeInfostateTree(*game, 0, algorithms::kNoMoveAheadLimit, algorithms::kStoreAllStatesPolicy),
     algorithms::MakeInfostateTree(*game, 1, algorithms::kNoMoveAheadLimit, algorithms::kStoreAllStatesPolicy),
  });
  TabularPolicy policy;
  for (int pl = 0; pl < 1; ++pl) {
    lp.SpecifyLinearProgram(pl);
    lp.Solve();
    policy.ImportPolicy(lp.OptimalPolicy(pl));
  }
  std::cout << policy.ToString() << "\n";

  std::vector<std::unique_ptr<State>> ss = GetStatesInSupport(*game, policy);
  for (const std::unique_ptr<State>& s : ss) {
    std::cout << s->HistoryString() << "\n";
  }

}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestStatesInSupport();
}

