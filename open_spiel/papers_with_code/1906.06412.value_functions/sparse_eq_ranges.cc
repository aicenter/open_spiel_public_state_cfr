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


#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_eq_ranges.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;
using namespace algorithms::dlcfr;
using namespace algorithms::ortools;

std::unique_ptr<SparseEqRanges> FindSparseEqRanges(
    SequenceFormLpSpecification* whole_game, DepthLimitedCFR* fixable_trunk) {

  std::vector<BanditVector>& bandits = fixable_trunk->bandits();
  fixable_trunk->Reset();

  for (int pl = 0; pl < 2; ++pl) {
    whole_game->SpecifyLinearProgram(pl);
    whole_game->Solve();
    TabularPolicy policy = whole_game->OptimalPolicy(pl);

    std::cout << "# Trunk eq strategies -- player " << pl << std::endl;
    for (DecisionId id : bandits[pl].range()) {
      bandits::Bandit* bandit = bandits[pl][id].get();
      auto* fixable_bandit =
          open_spiel::down_cast<bandits::FixableStrategy*>(bandit);
      absl::Span<double> bandit_policy = fixable_bandit->mutable_strategy();
      SPIEL_DCHECK_EQ(bandit_policy.size(), bandit->num_actions());

      // The underlying trees are not identical! So we cannot
      // copy the strategy using decision ids and resort to infostate strings.
      const InfostateNode* node =
          fixable_trunk->trees()[pl]->decision_infostate(id);
      const std::string& infostate_str = node->infostate_string();

      ActionsAndProbs state_policy = policy.GetStatePolicy(infostate_str);
      SPIEL_DCHECK_EQ(node->num_children(), bandit_policy.size());
      SPIEL_DCHECK_EQ(state_policy.size(), bandit_policy.size());
      for (int i = 0; i < bandit_policy.size(); ++i) {
        bandit_policy[i] = state_policy[i].second;
      }

      std::cout << "# " << node->ToString() << " " << bandit_policy << std::endl;
    }
  }

  // Compute the ranges from the trunk.
  fixable_trunk->UpdateReachProbs();
  fixable_trunk->EvaluateLeaves();

  // Copy the ranges.
  auto sparse_eq_ranges = std::make_unique<SparseEqRanges>();
  sparse_eq_ranges->eq_ranges.reserve(fixable_trunk->public_leaves().size());
  for (const LeafPublicState& state : fixable_trunk->public_leaves()) {
    sparse_eq_ranges->eq_ranges.push_back(state.ranges);
  }
  return sparse_eq_ranges;
}

std::array<std::vector<bool>, 2> SparseEqRanges::StateMask(
    const LeafPublicState& state) {

  const std::array<std::vector<double>, 2>& ranges =
      eq_ranges.at(state.public_id);

  std::array<std::vector<bool>, 2> mask = {
      std::vector<bool>(ranges[0].size()),
      std::vector<bool>(ranges[1].size())
  };

  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < ranges[pl].size(); ++i) {
      mask[pl][i] = IsReachable(ranges[pl][i]);
    }
  }
  return mask;
}

void SparseEqRanges::PrintMasks() const {
  std::cout << "# Equilibrium support mask:" << std::endl;
  int max_support = 0;
  for (const auto& state_ranges : eq_ranges) {
    std::cout << "# ";
    int num_reachable = 0;
    for (int pl = 0; pl < 2; ++pl) {
      for (int i = 0; i < state_ranges[pl].size(); ++i) {
        bool reachable = IsReachable(state_ranges[pl][i]);
        if (reachable) {
          num_reachable++;
          std::cout << "◉";
        } else {
          std::cout << "◯";
        }

      }
      std::cout << ' ';
    }
    std::cout << " in support: " << num_reachable << std::endl;
    max_support = std::max(max_support, num_reachable);
  }
  std::cout << "# Max support size: " << max_support << std::endl;
}
}  // papers_with_code
}  // open_spiel



