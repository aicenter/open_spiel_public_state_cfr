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

std::unique_ptr<Trunk> MakeTrunk(const std::string& game_name,
                                 int trunk_depth, std::string use_bandits_for_cfr) {
  return std::make_unique<Trunk>(game_name, trunk_depth, use_bandits_for_cfr);
}

Trunk::Trunk(const std::string& game_name, int depth, std::string use_bandits_for_cfr) {
  // 1. Prepare the game, observers and depth-limited (trunk) trees.
  game = LoadGame(game_name);
  trunk_depth = depth;
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
  oracle_evaluator->bandit_name = use_bandits_for_cfr;
  fixable_trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator,
      public_observer,
      MakeBanditVectors(trunk_trees, "FixableStrategy"));
  iterable_trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator,
      public_observer,
      MakeBanditVectors(trunk_trees, use_bandits_for_cfr));

  // 3. Make a Batch of data that encompasses all leaf public states.
  tables = CreateRangeTables(*game, hand_observer,
                             fixable_trunk_with_oracle->public_leaves());
  const dlcfr::LeafPublicState& some_leaf =
      fixable_trunk_with_oracle->public_leaves().at(0);


  dims.public_features_size = some_leaf.public_tensor.Tensor().size();
  for (int pl = 0; pl < 2; ++pl) {
    dims.net_ranges_size[pl] = tables[pl].largest_range();
  }
  num_leaves = fixable_trunk_with_oracle->public_leaves().size();
  num_non_terminal_leaves = 0;
  for (auto& leaf: fixable_trunk_with_oracle->public_leaves()) {
    if (!leaf.IsTerminal()) num_non_terminal_leaves++;
  }
}

void AddExperiencesFromTrunk(std::vector<dlcfr::LeafPublicState>& public_leaves,
                             const std::vector<dlcfr::RangeTable>& tables,
                             PositionalDataDims dims,
                             ExperienceReplay* replay) {
  for (const dlcfr::LeafPublicState& leaf : public_leaves) {
    PositionalData data_point = replay->AddExperience(dims);
    data_point.Reset();
    CopyFeatures(leaf.public_tensor.Tensor(), data_point);
    CopyDataTreeToNet(leaf, data_point, tables);
  }
}

void CopyFeatures(absl::Span<const float> features, PositionalData data_point) {
  std::copy(features.begin(), features.end(),
            data_point.public_features.begin());
}

void CopyDataTreeToNet(const dlcfr::LeafPublicState& leaf,
                       PositionalData data_point,
                       const std::vector<dlcfr::RangeTable>& tables) {
  CopyRangesTreeToNet(leaf, data_point, tables);
  CopyValuesTreeToNet(leaf, data_point, tables);
}

// Copy non-contiguous vectors using a permutation map.
// This also converts float <-> double as needed.
template<typename From, typename To>
void PlacementCopy(const std::vector<From>& from, absl::Span<To> to,
                   const std::map<size_t, size_t>& from_to) {
  for (const auto&[f, t] : from_to) {
    to[t] = from[f];
  }
}
template<typename From, typename To>
void PlacementCopy(absl::Span<From> from, std::vector<To>& to,
                   const std::map<size_t, size_t>& from_to) {
  for (const auto&[f, t] : from_to) {
    to[t] = from[f];
  }
}

void CopyRangesTreeToNet(const dlcfr::LeafPublicState& leaf,
                         PositionalData data_point,
                         const std::vector<dlcfr::RangeTable>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    PlacementCopy<float_tree, float_net>(
        /*tree=*/ leaf.ranges[pl],
        /*net=*/  data_point.net_ranges[pl],
                  tables[pl].bijections[leaf.public_id].tree_to_net());
  }
}

void CopyValuesTreeToNet(const dlcfr::LeafPublicState& leaf,
                         PositionalData data_point,
                         const std::vector<dlcfr::RangeTable>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    PlacementCopy<float_tree, float_net>(
        /*tree=*/ leaf.values[pl],
        /*net=*/  data_point.net_values[pl],
                  tables[pl].bijections[leaf.public_id].tree_to_net());
  }
}

void CopyValuesNetToTree(PositionalData data_point,
                         dlcfr::LeafPublicState& leaf,
                         const std::vector<dlcfr::RangeTable>& tables) {
  for (int pl = 0; pl < 2; ++pl) {
    PlacementCopy<float_net, float_tree>(
        /*net=*/  data_point.net_values[pl],
        /*tree=*/ leaf.values[pl],
                  tables[pl].bijections[leaf.public_id].net_to_tree());
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

