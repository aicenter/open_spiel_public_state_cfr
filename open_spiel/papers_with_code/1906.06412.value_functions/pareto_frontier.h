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


#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARETO_FRONTIER_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARETO_FRONTIER_

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace papers_with_code {

// Does "a" dominate "b"?
template<size_t N>
bool Dominates(const std::array<double, N>& a, const std::array<double, N>& b) {
  for (int i = 0; i < N; ++i) {
    if (a[i] < b[i]) return false;
  }
  return true;
}

template<size_t N>
std::vector<int> SortByParetoFrontier(
    const std::vector<std::array<double, N>>& xs, int max_points) {

  std::vector<int> frontier;
  int num_points = xs.size();
  max_points = std::min(max_points, num_points);

  enum { REMOVED    = 0
       , DOMINATED  = 1
       , DOMINATING = 2 };
  std::vector<int> domination(xs.size(), DOMINATING);

  while (frontier.size() != max_points) {
    for (int& el : domination) {  // Reset.
      if (el != REMOVED) el = DOMINATING;
    }

    for (int i = 0; i < xs.size(); ++i) {
      if (domination[i] != DOMINATING) continue;
      for (int j = 0; j < xs.size(); ++j) {
        if (i == j) continue;
        if (domination[j] != DOMINATING) continue;

        if (Dominates(xs[i], xs[j])) {
          domination[j] = DOMINATED;
        } else {
          domination[i] = DOMINATED;
          break;
        }
      }
    }

    // Add the frontier and mark it as removed.
    for (int i = 0; i < xs.size() && frontier.size() < max_points; ++i) {
      if (domination[i] == DOMINATING) {
        domination[i] = REMOVED;
        frontier.push_back(i);
      }
    }
  }

  SPIEL_CHECK_EQ(frontier.size(), max_points);
  return frontier;
}



} // namespace papers_with_code
} // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_PARETO_FRONTIER_
