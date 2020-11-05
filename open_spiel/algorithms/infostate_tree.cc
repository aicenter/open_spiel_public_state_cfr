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

#include "open_spiel/algorithms/infostate_tree.h"

namespace open_spiel {
namespace algorithms {

void CollectInfostateLookupTable(CFRNode* node,
                                 CFRInfoStateValuesPtrTable* out) {
  if (node->is_leaf_node()) return;
  if (node->type() == kDecisionInfostateNode) {
    (*out)[node->infostate_string()] = &node->values();
  }
  for (CFRNode& child : node->child_iterator()) {
    CollectInfostateLookupTable(&child, out);
  }
}

CFRNode::CFRNode(const CFRTree& tree, CFRNode* parent, int incoming_index,
                 InfostateNodeType type, const std::string& infostate_string,
                 double terminal_utility, double terminal_chn_reach_prob,
                 const State* originating_state) :
    InfostateNode<CFRNode>(
        tree, parent, incoming_index, type, infostate_string,
        terminal_utility, terminal_chn_reach_prob, originating_state)  {
  SPIEL_DCHECK_TRUE(
      !(originating_state && type == kDecisionInfostateNode)
          || originating_state->IsPlayerActing(tree.acting_player()));
  if (originating_state) {
    if (type_ == kDecisionInfostateNode) {
      values_ = CFRInfoStateValues(
          originating_state->LegalActions(tree.acting_player()));
    }
  }
}

CFRInfoStateValues* CFRNode::operator->() {
  SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
  return &values_;
}
const CFRInfoStateValues& CFRNode::values() const {
  SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
  return values_;
}
CFRInfoStateValues& CFRNode::values() {
  SPIEL_CHECK_EQ(type_, kDecisionInfostateNode);
  return values_;
}

}  // namespace algorithms
}  // namespace open_spiel
