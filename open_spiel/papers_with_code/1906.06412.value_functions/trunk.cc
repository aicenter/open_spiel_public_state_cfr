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


#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"

#include "open_spiel/utils/format_observation.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

void PrintRangeTables(const std::vector<dlcfr::RangeTable>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# List of private hands for pl " << pl << "\n";
    const dlcfr::RangeTable& table = tables[pl];
    for (int i = 0; i < table.private_hands.size(); ++i) {
      std::cout << "#   private_hand[" << i << "]:\n#      "
                << ObservationToString(table.private_hands[i], "\n#      ")
                << "\n";
    }

    std::cout << "# List of bijections (tree -> net) for pl " << pl << "\n";
    for (size_t i = 0; i < table.bijections.size(); ++i) {
      std::cout << "#  Public state " << i << "\n";
      const std::map<size_t, size_t>& tree_to_net =
          table.bijections[i].tree_to_net();
      for (auto&[key, val] : tree_to_net) {
        std::cout << "#   " << key << " -> " << val << "\n";
      }
    }
  }
}

void PrintBatchData(const BatchData& batch,
                    const std::vector<dlcfr::LeafPublicState>& states) {
  std::cout << "# Made BatchData with sizes:\n"
            << "#   batch_size=" << batch.batch_size << "\n"
            << "#   input_size=" << batch.input_size << "\n"
            << "#   output_size=" << batch.output_size << "\n"
            << "#   public_features_size=" << batch.public_features_size << "\n"
            << "#   ranges_size=" << batch.ranges_size << "\n";
  std::cout << "# Public features:\n";
  for (int i = 0; i < states.size(); ++i) {
    std::cout << "#   states[" << i << "].public_tensor\n#     "
              << ObservationToString(states[i].public_tensor, "\n#     ")
              << "\n";
  }
  std::cout << "# BatchData after feature copying:\n";
  std::cout << "#   " << batch.data << "\n";
}

std::unique_ptr<Trunk> MakeTrunk(const std::string& game_name,
                                        int trunk_depth) {
  return std::make_unique<Trunk>(game_name, trunk_depth);
}

Trunk::Trunk(const std::string& game_name, int trunk_depth) {
  // 1. Prepare the game, observers and depth-limited (trunk) trees.
  game = LoadGame(game_name);
  infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  public_observer = game->MakeObserver(kPublicStateObsType, {});
  hand_observer = game->MakeObserver(kHandHistoryObsType, {});
  trunk_trees = {MakeInfostateTree(*game, 0, trunk_depth),
                 MakeInfostateTree(*game, 1, trunk_depth)};

  // 2. Create value oracle for the trunk.
  terminal_evaluator = dlcfr::MakeTerminalEvaluator();
  oracle_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      game, /*full_subgame_depth=*/100, /*no_leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator,
      public_observer,
      MakeBanditVectors(trunk_trees, "FixableStrategy"));

  // 3. Make a Batch of data that encompasses all leaf public states.
  tables = CreateRangeTables(*game, hand_observer,
                             trunk_with_oracle->public_leaves());
  const dlcfr::LeafPublicState& some_leaf =
      trunk_with_oracle->public_leaves().at(0);
  const size_t encoding_size = some_leaf.public_tensor.Tensor().size();
  std::array<size_t, 2> ranges_size = {tables[0].largest_range(),
                                       tables[1].largest_range()};
  const size_t range_size_sum = ranges_size[0] + ranges_size[1];
  const size_t input_size = encoding_size + range_size_sum;
  const size_t output_size = range_size_sum;
  batch = std::make_unique<BatchData>(trunk_with_oracle->public_leaves(),
                                      input_size, output_size, encoding_size,
                                      ranges_size);
}

}  // namespace papers_with_code
}  // namespace open_spiel

