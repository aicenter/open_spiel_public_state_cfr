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


namespace open_spiel {
namespace algorithms {

void TopDown(
    const std::vector<std::vector<InfostateNode*>>& nodes_at_depth,
    DecisionVector<CFRInfoStateValues>& node_values,
    absl::Span<float> reach_probs) {
  const int tree_depth = nodes_at_depth.size();
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
    int right_offset = nodes_at_depth[d].size();
    for (int parent_idx = nodes_at_depth[d - 1].size() - 1;
         parent_idx >= 0; parent_idx--) {
      const float current_reach = reach_probs[parent_idx];
      const int num_children = nodes_at_depth[d - 1][parent_idx]->num_children();
      right_offset -= num_children;
      const InfostateNode* node = nodes_at_depth[d - 1][parent_idx];
      if (node->type() == kDecisionInfostateNode) {
        CFRInfoStateValues& values = node_values[node->decision_id()];
        const std::vector<double>& policy = values.current_policy;
        const std::vector<double>& regrets = values.cumulative_regrets;
        std::vector<double>& avg_policy = values.cumulative_policy;

        SPIEL_DCHECK_EQ(policy.size(), num_children);
        // Copy the policy and update with reach probs.
        // Update cumulative policy, as we now have the appropriate reaches.
        for (int i = 0; i < num_children; i++) {
          avg_policy[i] += policy[i] * current_reach;
        }

        for (int i = 0; i < num_children; i++) {
          reach_probs[right_offset + i] = policy[i] * current_reach;
        }

      } else {
        SPIEL_DCHECK_EQ(node->type(), kObservationInfostateNode);
        // Copy only the reach probs.
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
    const std::vector<std::vector<InfostateNode*>>& nodes_at_depth,
    DecisionVector<CFRInfoStateValues>& node_values,
    absl::Span<float> cf_values) {
  const int tree_depth = nodes_at_depth.size();
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
    for (int parent_idx = 0; parent_idx < nodes_at_depth[d].size();
         parent_idx++) {
      const InfostateNode* node = nodes_at_depth[d][parent_idx];
      const int num_children = node->num_children();
      double node_sum = 0.;
      if (node->type() == kDecisionInfostateNode) {
        CFRInfoStateValues& values = node_values[node->decision_id()];
        std::vector<double>& regrets = values.cumulative_regrets;
        std::vector<double>& policy = values.current_policy;
        SPIEL_DCHECK_EQ(policy.size(), num_children);
        SPIEL_DCHECK_EQ(regrets.size(), num_children);
        // Propagate child values by multiplying with current policy.
        for (int i = 0; i < num_children; i++) {
          node_sum += policy[i] * cf_values[left_offset + i];
        }
        // TODO: abstract away RM!
        // Update regrets.
        for (int i = 0; i < num_children; i++) {
          regrets[i] += cf_values[left_offset + i] - node_sum;
        }
        // Apply RM: compute current policy.
        double sum_positive_regrets = 0.;
        for (int i = 0; i < num_children; i++) {
          if (regrets[i] > 0) {
            sum_positive_regrets += regrets[i];
          }
        }
        for (int i = 0; i < num_children; ++i) {
          if (sum_positive_regrets > 0) {
            policy[i] = regrets[i] > 0
                        ? regrets[i] / sum_positive_regrets
                        : 0;
          } else {
            policy[i] = 1.0 / num_children;
          }
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
    SPIEL_DCHECK_EQ(left_offset, nodes_at_depth[d + 1].size());
  }
}

float RootCfValue(int root_branching_factor, absl::Span<const float> cf_values,
                  absl::Span<const float> range) {
  SPIEL_CHECK_TRUE(range.empty() ||
                   (range.size() == root_branching_factor
                    && range.size() == cf_values.size()));
  float root_value = 0.;
  if (range.empty()) {
    for (int i = 0; i < root_branching_factor; ++i) {
      root_value += cf_values[i];
    }
  } else {
    for (int i = 0; i < root_branching_factor; ++i) {
      root_value += range[i] * cf_values[i];
    }
  }
  return root_value;
}

namespace {
// Make sure we can get the average policy to compute expected values
// and exploitability.
class InfostateCFRAveragePolicy : public Policy {
  const CFRInfoStateValuesPtrTable infostate_ptr_table_;
 public:
  InfostateCFRAveragePolicy(const CFRInfoStateValuesPtrTable& ptable)
      : infostate_ptr_table_(ptable) {}
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    const CFRInfoStateValues& vs = *infostate_ptr_table_.at(info_state);
    double sum_prob = 0.0;
    for (int i = 0; i < vs.num_actions(); ++i) {
      sum_prob += vs.cumulative_policy[i];
    }

    ActionsAndProbs out;
    out.reserve(vs.num_actions());
    for (int i = 0; i < vs.num_actions(); ++i) {
      if (sum_prob > 0) {
        out.push_back({vs.legal_actions[i],
                       vs.cumulative_policy[i] / sum_prob});
      } else {
        // Return a uniform policy at this node
        out.push_back({vs.legal_actions[i],
                       vs.cumulative_policy[i] / vs.num_actions()});
      }
    }
    return out;
  }
};

}  // namespace

std::unordered_map<const InfostateNode*, CFRInfoStateValues> CreateTable(
    const std::array<std::shared_ptr<InfostateTree>, 2>& trees) {
  std::unordered_map<const InfostateNode*, CFRInfoStateValues> map;
  for (int pl = 0; pl < 2; ++pl) {
    const std::vector<std::vector<InfostateNode*>>& nodes =
        trees[pl]->nodes_at_depths();
    for (int d = 0; d < nodes.size() - 1; ++d) {
      for (const InfostateNode* node : nodes[d]) {
        if (node->type() != kDecisionInfostateNode) continue;
        map[node] = CFRInfoStateValues(node->legal_actions());
      }
    }
  }
  return map;
}

InfostateCFR::InfostateCFR(std::array<std::shared_ptr<InfostateTree>, 2> trees)
    : trees_(std::move(trees)),
      cf_values_({
        std::vector<float>(trees_[0]->num_leaves(), 0.),
        std::vector<float>(trees_[1]->num_leaves(), 0.)
      }),
      reach_probs_({
        std::vector<float>(trees_[0]->num_leaves(), 0.),
        std::vector<float>(trees_[1]->num_leaves(), 0.)
      }),
      node_values_({
        DecisionVector<CFRInfoStateValues>(trees_[0].get()),
        DecisionVector<CFRInfoStateValues>(trees_[1].get())
      }) {
  PrepareTerminals();
}

InfostateCFR::InfostateCFR(const Game& game)
    : InfostateCFR({MakeInfostateTree(game, 0), MakeInfostateTree(game, 1)}) {}

void InfostateCFR::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    PrepareRootReachProbs();
    TopDown(trees_[0]->nodes_at_depths(), node_values_[0], absl::MakeSpan(reach_probs_[0]));
    TopDown(trees_[1]->nodes_at_depths(), node_values_[1], absl::MakeSpan(reach_probs_[1]));
    SPIEL_CHECK_FLOAT_NEAR(TerminalReachProbSum(), 1.0, 1e-3);

    EvaluateLeaves();
    BottomUp(trees_[0]->nodes_at_depths(), node_values_[0], absl::MakeSpan(cf_values_[0]));
    BottomUp(trees_[1]->nodes_at_depths(), node_values_[1], absl::MakeSpan(cf_values_[1]));
    SPIEL_CHECK_FLOAT_NEAR(
        RootCfValue(trees_[0]->root_branching_factor(), cf_values_[0]),
        - RootCfValue(trees_[1]->root_branching_factor(), cf_values_[1]), 1e-6);
  }
}
void InfostateCFR::RunAlternatingIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    for (int pl = 0; pl < 2; ++pl) {
      PrepareRootReachProbs(1 - pl);
      TopDown(trees_[1 - pl]->nodes_at_depths(), node_values_[1 - pl], absl::MakeSpan(reach_probs_[1 - pl]));
      EvaluateLeaves(pl);
      BottomUp(trees_[pl]->nodes_at_depths(), node_values_[pl], absl::MakeSpan(cf_values_[pl]));
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
    cf_values_[0][i] =  terminal_values_[i] * reach_probs_[1][j];
    cf_values_[1][j] = -terminal_values_[i] * reach_probs_[0][i];
  }
}
void InfostateCFR::EvaluateLeaves(Player pl) {
  if (pl == 0) {
    for (int i = 0; i <  trees_[0]->num_leaves(); ++i) {
      const int j = terminal_permutation_[i];
      cf_values_[0][i] =  terminal_values_[i] * reach_probs_[1][j];
    }
  } else {
    for (int i = 0; i < trees_[1]->num_leaves(); ++i) {
      const int j = terminal_permutation_[i];
      cf_values_[1][j] = -terminal_values_[i] * reach_probs_[0][i];
    }
  }
}
CFRInfoStateValuesPtrTable InfostateCFR::InfoStateValuesPtrTable() {
  CFRInfoStateValuesPtrTable vec_ptable;
  for (int pl = 0; pl < 2; ++pl) {
    for (DecisionId id : trees_[pl]->AllDecisionIds()) {
      vec_ptable[trees_[pl]->decision_infostate(id)->infostate_string()]
        = &node_values_[pl][id];
    }
  }
  return vec_ptable;
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
float InfostateCFR::TerminalReachProbSum() {
  const int num_terminals = terminal_values_.size();
  float reach_sum = 0.;
  for (int i = 0; i < num_terminals; ++i) {
    const int j = terminal_permutation_[i];
    const float leaf_reach = terminal_ch_reaches_[i]
        * reach_probs_[0][i] * reach_probs_[1][j];
    SPIEL_CHECK_LE(leaf_reach, 1.0);
    reach_sum += leaf_reach;
  }
  return reach_sum;
}

std::shared_ptr<Policy> InfostateCFR::AveragePolicy(){
  return std::make_shared<InfostateCFRAveragePolicy>(InfoStateValuesPtrTable());
}
float InfostateCFR::RootValue() const {
  return RootCfValue(trees_[0]->root_branching_factor(), cf_values_[0]);
}

}  // namespace algorithms
}  // namespace open_spiel
