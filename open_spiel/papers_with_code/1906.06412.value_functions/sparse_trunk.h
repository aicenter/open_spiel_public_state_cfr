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

struct SparseTrunk {
  // Equilibrium ranges for each leaf public state, each player
  // and each infostate of the trunk.
  std::vector<std::array<std::vector<double>, 2>> eq_ranges;

  bool IsReachable(const algorithms::dlcfr::LeafPublicState& state,
                   Player p, int infostate_index) const {
    return IsReachable(eq_ranges.at(state.public_id).at(p).at(infostate_index));
  }
  bool IsReachable(double prob) const { return prob > 0.; }

  std::array<std::vector<bool>, 2> StateMask(
      const algorithms::dlcfr::LeafPublicState& state);
  void PrintMasks() const;
};

std::unique_ptr<SparseTrunk> FindSparseTrunk(
    algorithms::ortools::SequenceFormLpSpecification* whole_game,
    algorithms::dlcfr::DepthLimitedCFR* fixable_trunk);

}  // papers_with_code
}  // open_spiel



#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SPARSE_TRUNK_
