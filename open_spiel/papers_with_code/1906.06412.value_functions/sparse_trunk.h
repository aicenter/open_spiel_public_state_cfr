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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SPARSE_TRUNK_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SPARSE_TRUNK_

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/sequence_form_lp.h"

namespace open_spiel {
namespace papers_with_code {

// See comment below.
struct SparseTrunk {
  std::unique_ptr<algorithms::dlcfr::DepthLimitedCFR> dlcfr;
  std::vector<std::string> eval_infostates;
};

// For each eval_infostate within each public state at the specified depth of
// the game, make a new SparseTrunk. These sparse trunks are constructed based
// on a set of histories: we need a mechanism to specify what this set of
// histories is. There is always at least one history that belongs to
// eval_infostate. The other histories will be taken randomly according to
// limit_initial_states, within the public state of eval_infostate.
std::vector<std::unique_ptr<SparseTrunk>> MakeSparseTrunks(
    std::shared_ptr<const Game> game,
    std::shared_ptr<Observer> infostate_observer,
    std::shared_ptr<Observer> public_observer,
    int roots_depth, int trunk_depth,
    std::shared_ptr<const algorithms::dlcfr::LeafEvaluator> net_evaluator,
    std::shared_ptr<const algorithms::dlcfr::LeafEvaluator> terminal_evaluator,
    int limit_initial_states, const std::string& bandits_for_cfr,
    std::mt19937& rnd_gen);


}  // papers_with_code
}  // open_spiel



#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SPARSE_TRUNK_
