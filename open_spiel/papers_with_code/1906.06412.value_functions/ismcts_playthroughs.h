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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_ISMCTS_PLAYTHROUGHS_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_ISMCTS_PLAYTHROUGHS_

#include "open_spiel/algorithms/is_mcts.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

struct NodeStats{
  int visits;
  int move_number;
  int player;
};

using InfostateStats = std::unordered_map<Observation, NodeStats>;

// Implemented only for imp. info GoofSpiel.
struct IsmctsPlaythroughs {
  std::unique_ptr<algorithms::ISMCTSBot> bot;
  // Settings:
  int num_matches = 2;
  double uct_c = 10.;
  int max_simulations = 100;
  algorithms::ISMCTSFinalPolicyType policy_type =
      algorithms::ISMCTSFinalPolicyType::kNormalizedVisitCount;

  // For storing infostate stats.
  InfostateStats infostate_stats;
  // Cumulative distribution function based on the visit counts,
  // for each move number in the game.
  std::vector</*move_number*/
    std::map</*cumul=*/double,
             /*stats_ref=*/InfostateStats::iterator>> cdfs;

  void MakeBot(std::mt19937& rnd_gen);;
  void GenerateNodes(const Game& game, std::mt19937& rnd_gen);
  InfostateStats::iterator SampleInfostate(int move_number, std::mt19937& rnd_gen);
};

}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_ISMCTS_PLAYTHROUGHS_
