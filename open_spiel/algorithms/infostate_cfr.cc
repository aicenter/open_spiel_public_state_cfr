// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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


#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/utils/functional.h"


namespace open_spiel {
namespace algorithms {

void TopDown(
    const InfostateTree& tree, absl::Span<double> reach_probs,
    std::function<std::vector<double>(
        DecisionId, /*current_reach=*/double)> policy_fn) {
  const std::vector<std::vector<InfostateNode*>>& nodes_at_depths =
      tree.nodes_at_depths();
  SPIEL_CHECK_EQ(reach_probs.size(), nodes_at_depths.back().size());
  const int tree_depth = nodes_at_depths.size();
  // Loop over all depths, except for the first two depths:
  // - Depth 0: corresponds to the dummy observation node, which is used mainly
  //   for computing the sum of all cf values. As it is not a decision node,
  //   it is not involved in the calculation of reach probs and can be skipped.
  // - Depth 1: now there may be some first decision nodes. The caller must fill
  //   reach probs for this depth. This is done so that we can support arbitrary
  //   depth-limited infostate trees.
  for (int d = 2; d < tree_depth; d++) {
    // Loop over all parents of current nodes.
    // We do it in reverse, i.e. from the last parent index to the first one.
    // As we update reach probs, we overwrite the same buffer so we lose the
    // current reach. However, because the tree is balanced and the usage
    // of the buffer only monotically grows with depth, doing it in reverse we
    // do not overwrite the current reach prob.
    int right_offset = nodes_at_depths[d].size();
    for (int parent_idx = nodes_at_depths[d - 1].size() - 1;
         parent_idx >= 0; parent_idx--) {
      const double current_reach = reach_probs[parent_idx];
      const InfostateNode* node = nodes_at_depths[d - 1][parent_idx];
      const int num_children = node->num_children();
      right_offset -= num_children;

      if (node->type() == kDecisionInfostateNode) {
        std::vector<double> policy =
            policy_fn(node->decision_id(), current_reach);
        SPIEL_DCHECK_EQ(policy.size(), num_children);
        SPIEL_DCHECK_TRUE(IsValidProbDistribution(policy));
        for (int i = 0; i < num_children; i++) {
          reach_probs[right_offset + i] = policy[i] * current_reach;
        }
      } else {
        SPIEL_DCHECK_EQ(node->type(), kObservationInfostateNode);
        // Copy the reach probs.
        for (int i = 0; i < num_children; i++) {
          reach_probs[right_offset + i] = current_reach;
        }
      }
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(right_offset, 0);
  }
}

void BottomUp(
    const InfostateTree& tree, absl::Span<double> cf_values,
    std::function<
        void(DecisionId, /*loss=*/absl::Span<const double>)> observe_loss_fn,
    std::function<std::vector<double>(DecisionId)> policy_fn) {
  const std::vector<std::vector<InfostateNode*>>& nodes_at_depths =
      tree.nodes_at_depths();
  SPIEL_CHECK_EQ(cf_values.size(), nodes_at_depths.back().size());
  const int tree_depth = nodes_at_depths.size();
  // Loop over all depths, except for the last one, as it is already set
  // by calling the leaf evaluation.
  for (int d = tree_depth - 2; d >= 1; d--) {
    // Loop over all parents of current nodes.
    // We do it in forward mode, i.e. from the first parent index to the last
    // one. As we update cf values, we overwrite the same buffer, so we lose
    // the children values. However, because the tree is balanced and
    // the usage of the buffer only monotically grows with depth, doing it in
    // forward we do not overwrite the parent's node cf value.
    int left_offset = 0;
    // Loop over all parents of current nodes.
    for (int parent_idx = 0; parent_idx < nodes_at_depths[d].size();
         parent_idx++) {
      const InfostateNode* node = nodes_at_depths[d][parent_idx];
      const int num_children = node->num_children();
      double node_sum = 0.;
      if (node->type() == kDecisionInfostateNode) {
        observe_loss_fn(
            node->decision_id(),
            absl::Span<const double>(&cf_values[left_offset], num_children));
        std::vector<double> policy = policy_fn(node->decision_id());
        // Propagate child values by multiplying with current policy.
        for (int i = 0; i < num_children; i++) {
          node_sum += policy[i] * cf_values[left_offset + i];
        }
      } else {
        SPIEL_DCHECK_EQ(node->type(), kObservationInfostateNode);
        // Just sum the child values, no policy weighing is needed.
        for (int i = 0; i < num_children; i++) {
          node_sum += cf_values[left_offset + i];
        }
      }

      cf_values[parent_idx] = node_sum;
      left_offset += num_children;
    }
    // Check that we passed over all of the children.
    SPIEL_DCHECK_EQ(left_offset, nodes_at_depths[d + 1].size());
  }
}

double RootCfValue(int root_branching_factor,
                   absl::Span<const double> cf_loss,
                   absl::Span<const double> range) {
  SPIEL_CHECK_TRUE(range.empty() ||
                   (range.size() == root_branching_factor
                    && range.size() == cf_loss.size()));
  double root_value = 0.;
  if (range.empty()) {
    for (int i = 0; i < root_branching_factor; ++i) {
      root_value += cf_loss[i];
    }
  } else {
    for (int i = 0; i < root_branching_factor; ++i) {
      root_value += range[i] * cf_loss[i];
    }
  }
  return -root_value;
}

InfostateCFR::InfostateCFR(std::vector<std::shared_ptr<InfostateTree>> trees)
    : trees_(std::move(trees)),
      cf_values_({
                     std::vector<double>(trees_[0]->num_leaves(), 0.),
                     std::vector<double>(trees_[1]->num_leaves(), 0.)
                 }),
      reach_probs_({
                       std::vector<double>(trees_[0]->num_leaves(), 0.),
                       std::vector<double>(trees_[1]->num_leaves(), 0.)
                   }),
      bandits_(MakeBanditVectors(trees_)) {
  SPIEL_CHECK_EQ(trees_.size(), 2);
  PrepareTerminals();
}

InfostateCFR::InfostateCFR(const Game& game)
    : InfostateCFR({MakeInfostateTree(game, 0), MakeInfostateTree(game, 1)}) {}

void InfostateCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;
    PrepareRootReachProbs();
    for (int pl = 0; pl < 2; ++pl) {
      TopDownCurrentPolicyWithCompute(
          *trees_[pl], bandits_[pl],
          absl::MakeSpan(reach_probs_[pl]), num_iterations_);
    }
    SPIEL_CHECK_FLOAT_NEAR(TerminalReachProbSum(), 1.0, 1e-3);

    EvaluateLeaves();
    for (int pl = 0; pl < 2; ++pl) {
      BottomUp(*trees_[pl], bandits_[pl], absl::MakeSpan(cf_values_[pl]));
    }
    SPIEL_CHECK_FLOAT_NEAR(
        RootCfValue(trees_[0]->root_branching_factor(), cf_values_[0]),
        -RootCfValue(trees_[1]->root_branching_factor(), cf_values_[1]), 1e-6);
  }
}
void InfostateCFR::RunAlternatingIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;
    for (int pl = 0; pl < 2; ++pl) {
      PrepareRootReachProbs(1 - pl);
      TopDownCurrentPolicyWithCompute(
          *trees_[1 - pl], bandits_[1 - pl],
          absl::MakeSpan(reach_probs_[1 - pl]), num_iterations_);

      EvaluateLeaves(pl);
      BottomUp(*trees_[pl], bandits_[pl], absl::MakeSpan(cf_values_[pl]));
    }
  }
}

void InfostateCFR::PrepareRootReachProbs() {
  for (int pl = 0; pl < 2; ++pl) PrepareRootReachProbs(pl);
}

void InfostateCFR::PrepareRootReachProbs(Player pl) {
  for (int i = 0; i < trees_[pl]->root_branching_factor(); ++i) {
    reach_probs_[pl][i] = 1.;
  }
}

void InfostateCFR::EvaluateLeaves() {
  for (int i = 0; i < trees_[0]->num_leaves(); ++i) {
    const int j = terminal_permutation_[i];
    cf_values_[0][i] = - terminal_values_[i] * reach_probs_[1][j];
    cf_values_[1][j] =   terminal_values_[i] * reach_probs_[0][i];
  }
}
void InfostateCFR::EvaluateLeaves(Player pl) {
  if (pl == 0) {
    for (int i = 0; i < trees_[0]->num_leaves(); ++i) {
      const int j = terminal_permutation_[i];
      cf_values_[0][i] = - terminal_values_[i] * reach_probs_[1][j];
    }
  } else {
    for (int i = 0; i < trees_[1]->num_leaves(); ++i) {
      const int j = terminal_permutation_[i];
      cf_values_[1][j] = terminal_values_[i] * reach_probs_[0][i];
    }
  }
}
void InfostateCFR::PrepareTerminals() {
  const std::vector<InfostateNode*>& leaves_a = trees_[0]->leaf_nodes();
  const std::vector<InfostateNode*>& leaves_b = trees_[1]->leaf_nodes();
  SPIEL_CHECK_EQ(leaves_a.size(), leaves_b.size());

  const int num_terminals = leaves_a.size();
  terminal_values_.reserve(num_terminals);
  terminal_ch_reaches_.reserve(num_terminals);
  terminal_permutation_.reserve(num_terminals);

  using History = absl::Span<const Action>;
  std::map<History, int> player1_map;
  for (int i = 0; i < num_terminals; ++i) {
    player1_map[leaves_b[i]->TerminalHistory()] = i;
  }
  SPIEL_CHECK_EQ(player1_map.size(), leaves_b.size());

  for (int i = 0; i < num_terminals; ++i) {
    // This CFR variant works only with leaf nodes being terminal nodes.
    const InfostateNode* const a = leaves_a[i];
    SPIEL_CHECK_TRUE(a->type() == kTerminalInfostateNode);
    const int permutation_index = player1_map.at(a->TerminalHistory());
    const InfostateNode* const b = leaves_b[permutation_index];
    SPIEL_CHECK_TRUE(b->type() == kTerminalInfostateNode);
    SPIEL_DCHECK_EQ(a->TerminalHistory(), b->TerminalHistory());

    const InfostateNode* const leaf = leaves_a[i];
    const double v = leaf->terminal_utility();
    const double chn = leaf->terminal_chance_reach_prob();
    terminal_values_.push_back(v * chn);
    terminal_ch_reaches_.push_back(chn);
    terminal_permutation_.push_back(permutation_index);
  }
  SPIEL_DCHECK_EQ(
  // A quick check to see if the permutation is ok
  // by computing the arithmetic sum.
      std::accumulate(terminal_permutation_.begin(),
                      terminal_permutation_.end(), 0),
      num_terminals * (num_terminals - 1) / 2);
}
double InfostateCFR::TerminalReachProbSum() {
  const int num_terminals = terminal_values_.size();
  double reach_sum = 0.;
  for (int i = 0; i < num_terminals; ++i) {
    const int j = terminal_permutation_[i];
    const double leaf_reach = terminal_ch_reaches_[i]
        * reach_probs_[0][i] * reach_probs_[1][j];
    SPIEL_CHECK_GE(leaf_reach, 0.);
    SPIEL_CHECK_LE(leaf_reach, 1.);
    reach_sum += leaf_reach;
  }
  return reach_sum;
}

std::shared_ptr<Policy> InfostateCFR::AveragePolicy() {
  return std::make_shared<BanditsAveragePolicy>(trees_, bandits_);
}
std::shared_ptr<Policy> InfostateCFR::CurrentPolicy() {
  return std::make_shared<BanditsCurrentPolicy>(trees_, bandits_);
}
double InfostateCFR::RootValue() const {
  return RootCfValue(trees_[0]->root_branching_factor(), cf_values_[0]);
}

}  // namespace algorithms
}  // namespace open_spiel
