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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRUNK_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRUNK_

#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

struct Trunk {
  std::shared_ptr<const Game> game;
  int trunk_depth;
  std::shared_ptr<Observer> infostate_observer;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> hand_observer;
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trunk_trees;
  std::shared_ptr<const algorithms::dlcfr::LeafEvaluator> terminal_evaluator;
  std::shared_ptr<algorithms::dlcfr::CFREvaluator> oracle_evaluator;
  std::unique_ptr<algorithms::dlcfr::DepthLimitedCFR> fixable_trunk_with_oracle;
  std::unique_ptr<algorithms::dlcfr::DepthLimitedCFR> iterable_trunk_with_oracle;
  std::vector<algorithms::dlcfr::RangeTable> tables;
  PositionalDataDims dims;
  int num_leaves;
  int num_non_terminal_leaves;

  Trunk(const std::string& game_name, int trunk_depth,
        std::string use_bandits_for_cfr = "RegretMatchingPlus");
};

std::unique_ptr<Trunk> MakeTrunk(const std::string& game_name, int trunk_depth,
    std::string use_bandits_for_cfr = "RegretMatchingPlus");

void AddExperiencesFromTrunk(
    std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves,
    const std::vector<algorithms::dlcfr::RangeTable>& tables,
    PositionalDataDims dims,
    ExperienceReplay* replay);

void CopyFeatures(absl::Span<const float> features, PositionalData data_point);

void CopyDataTreeToNet(const algorithms::dlcfr::LeafPublicState& leaf,
                       PositionalData data_point,
                       const std::vector<algorithms::dlcfr::RangeTable>& tables);

void CopyRangesTreeToNet(const algorithms::dlcfr::LeafPublicState& leaf,
                         PositionalData data_point,
                         const std::vector<algorithms::dlcfr::RangeTable>& tables);

void CopyValuesTreeToNet(const algorithms::dlcfr::LeafPublicState& leaf,
                         PositionalData data_point,
                         const std::vector<algorithms::dlcfr::RangeTable>& tables);

void CopyValuesNetToTree(PositionalData data_point,
                         algorithms::dlcfr::LeafPublicState& leaf,
                         const std::vector<algorithms::dlcfr::RangeTable>& tables);



}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRUNK_
