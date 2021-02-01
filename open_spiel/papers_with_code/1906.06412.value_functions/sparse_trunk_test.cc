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


#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

using namespace algorithms;

void CheckSparseTrunksKuhnDepth2(
    const std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 3);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    // One char, i.e. card 0/1/2
    SPIEL_CHECK_EQ(sparse_trunks[i]->eval_infostate.size(), 1);

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                     ->leaf_nodes().size(), num_states * 5);
    }
  }
}

void CheckSparseTrunksKuhnDepth3(
    const std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 6);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    // Two chars, i.e. card 0/1/2 followed by p/b
    SPIEL_CHECK_EQ(sparse_trunks[i]->eval_infostate.size(), 2);

    for (int pl = 0; pl < 2; ++pl) {
      if (sparse_trunks[i]->eval_infostate[1] == 'p') {
        // Pass infostate
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                       ->leaf_nodes().size(), num_states * 3);
      } else if (sparse_trunks[i]->eval_infostate[1] == 'b') {
        // Bet infostate
        SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                       ->leaf_nodes().size(), num_states * 2);
      } else {
        SpielFatalError("Exhausted pattern match!");
      }
    }
  }
}

void CheckSparseTrunksKuhnDepth4(
    const std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks,
    int num_states) {
  SPIEL_CHECK_EQ(sparse_trunks.size(), 3);
  for (int i = 0; i < sparse_trunks.size(); ++i) {
    // Three chars, i.e. card 0/1/2 followed by pb
    SPIEL_CHECK_EQ(sparse_trunks[i]->eval_infostate.size(), 3);

    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_EQ(sparse_trunks[i]->dlcfr->trees()[pl]
                     ->leaf_nodes().size(), num_states * 2);
    }
  }
}

void TestMakeSparseTrunks() {
  std::shared_ptr<const Game> game = LoadGame("kuhn_poker");
  std::shared_ptr<Observer> infostate_observer =
      game->MakeObserver(kInfoStateObsType, {});
  std::shared_ptr<Observer> public_observer =
      game->MakeObserver(kPublicStateObsType, {});
  std::shared_ptr<dlcfr::LeafEvaluator> terminal_evaluator =
      dlcfr::MakeTerminalEvaluator();
  std::shared_ptr<dlcfr::LeafEvaluator> dummy_eval =
      dlcfr::MakeDummyEvaluator();
  std::string bandits_for_cfr = "RegretMatchingPlus";
  std::mt19937 rnd_gen(0);

  // Use any valid number of states.
  // -1 are for full trees test (just as 6) -- there is no sparsification.
  for (int limit = -1; limit <= 6; ++limit) {
    if (limit == 0) continue;

    for (int depth = 2; depth <= 4; ++depth) {
      std::vector<std::unique_ptr<SparseTrunk>> sparse_trunks =
          MakeSparseTrunks(game, infostate_observer, public_observer, depth,
                           dummy_eval, terminal_evaluator,
                           limit, bandits_for_cfr, rnd_gen);
      int num_states = limit > 0 ? limit : 6;

      switch (depth) {
        case 2: CheckSparseTrunksKuhnDepth2(sparse_trunks, num_states); break;
        case 3: CheckSparseTrunksKuhnDepth3(sparse_trunks, num_states); break;
        case 4: CheckSparseTrunksKuhnDepth4(sparse_trunks, num_states); break;
      }
    }
  }
}

}  // namespace
}  // papers_with_code
}  // open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestMakeSparseTrunks();
}
