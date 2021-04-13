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

#ifndef OPEN_SPIEL_UTILS_TREE_H_
#define OPEN_SPIEL_UTILS_TREE_H_

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"

namespace open_spiel {

// A helper container that stores some leaf values, as identified by a tree
// trace to this leaf. This is intended as a replacement for
// std::map<std::vector<Edge>, LeafValue>
template<typename Edge, typename LeafValue>
class TreeMap {
  std::vector<std::pair<Edge, std::unique_ptr<TreeMap>>> children_;
  LeafValue value_;
 public:
  LeafValue& operator[](absl::Span<const Edge> trace) {
    if (trace.empty()) return value_;

    const Edge& head = trace[0];
    for (const auto&[edge, subtree] : children_) {
      if (edge == head) {
        return subtree->operator[](trace.subspan(1));
      }
    }
    children_.push_back({head, std::make_unique<TreeMap>()});
    return children_.back().second->operator[](trace.subspan(1));
  }

  absl::optional<LeafValue> at(absl::Span<const Edge> trace) const {
    if (trace.empty()) return value_;

    const Edge& head = trace[0];
    for (const auto&[edge, subtree] : children_) {
      if (edge == head) {
        return subtree->at(trace.subspan(1));
      }
    }
    return {};
  }

  LeafValue fold_sum(LeafValue initial_acc) const {
    if (children_.empty()) return value_;

    LeafValue node_acc = initial_acc;
    for (const auto&[edge, subtree] : children_) {
      node_acc = node_acc + subtree->fold_sum(initial_acc);
    }
    return node_acc;
  }
};

}  // open_spiel

#endif  // OPEN_SPIEL_UTILS_TREE_H_
