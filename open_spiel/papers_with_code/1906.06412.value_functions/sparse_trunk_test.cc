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


#include "open_spiel/algorithms/ortools/sequence_form_lp.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

using namespace algorithms;

class MaskedEvaluator : public dlcfr::CFREvaluator {
  std::unique_ptr<SparseTrunk> sparse_trunk_;

 public:
  MaskedEvaluator(std::shared_ptr<const Game> game, int depth_limit,
                  std::shared_ptr<const LeafEvaluator> leaf_evaluator,
                  std::shared_ptr<const LeafEvaluator> terminal_evaluator,
                  std::shared_ptr<Observer> public_observer,
                  std::shared_ptr<Observer> infostate_observer,
                  std::unique_ptr<SparseTrunk> sparse_trunk)
      : dlcfr::CFREvaluator(game, depth_limit, leaf_evaluator,
                            terminal_evaluator, public_observer,
                            infostate_observer),
        sparse_trunk_(std::move(sparse_trunk)) {}

  void EvaluatePublicState(LeafPublicState* public_state,
                           PublicStateContext* context) const override {

    CFREvaluator::EvaluatePublicState(public_state, context);

    // Mask the output values.
    std::array<std::vector<bool>, 2> mask = sparse_trunk_->StateMask(*public_state);
    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_DCHECK_EQ((*reachable_mask)[pl].size(), state.ranges[pl].size());

      for (int j = 0; j < public_state->leaf_nodes[pl].size(); j++) {
        if(!mask[pl][j]) {
          // Write a big negative value for this infostate.
          // This should be a value larger than any utility in the game.
          public_state->values[pl][j] = kSparseTrunkDoNotFollowValue;
          continue;  // Skip copying in this infostate!
        }
      }
    }
  }
};

void TestDlCFRConvergenceWithSparseTrunk() {
  auto trunk = MakeTrunk(
//      "leduc_poker", 6,
      "goofspiel(players=2,num_cards=4,imp_info=True,points_order=descending)", 1,
      "RegretMatching");

  ortools::SequenceFormLpSpecification whole_game(*trunk->game, "CLP");
  std::unique_ptr<SparseTrunk> sparse_trunk = FindSparseTrunk(
      &whole_game, trunk->fixable_trunk_with_oracle.get());
  sparse_trunk->PrintMasks();

  auto masked_evaluator = std::make_shared<const MaskedEvaluator>(
      trunk->game, /*full_subgame_depth=*/100, /*no_leaf_evaluator=*/nullptr,
      trunk->terminal_evaluator, trunk->public_observer,
      trunk->infostate_observer, std::move(sparse_trunk));

  dlcfr::DepthLimitedCFR masked_dlcfr(
      trunk->game, trunk->trunk_trees, masked_evaluator,
      trunk->terminal_evaluator, trunk->public_observer,
      MakeBanditVectors(trunk->trunk_trees, "RegretMatching"));

  for (int i = 0; i < 500; ++i) {
    masked_dlcfr.RunSimultaneousIterations(50);
    double expl = TrunkExploitability(&whole_game, *masked_dlcfr.AveragePolicy());
    std::cout << expl << "\n";
    PrintTrunkStrategies(&masked_dlcfr);
  }
}

}  // namespace
}  // papers_with_code
}  // open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestDlCFRConvergenceWithSparseTrunk();
}
