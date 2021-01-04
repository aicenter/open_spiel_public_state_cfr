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

#ifndef OPEN_SPIEL_UTILS_DATA_STRUCTURES_
#define OPEN_SPIEL_UTILS_DATA_STRUCTURES_

#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/time/time.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"

namespace open_spiel {

template<class T>
struct BijectiveContainer {
  std::map<T, T> x2y;
  std::map<T, T> y2x;

  void put(std::pair<T, T> xy) {
    const T& x = xy.first;
    const T& y = xy.second;
    SPIEL_CHECK_TRUE(x2y.find(x) == x2y.end());
    SPIEL_CHECK_TRUE(y2x.find(y) == y2x.end());
    x2y[x] = y;
    y2x[y] = x;
  }
  // Direction is equivalent to player id.
  const std::map<T, T>& association(int direction) const {
    SPIEL_CHECK_TRUE(direction == 0 || direction == 1);
    if (direction == 0) return x2y;
    else return y2x;
  }

  size_t size() const {
    SPIEL_CHECK_EQ(x2y.size(), y2x.size());
    return x2y.size();
  }

  // TODO: remove
  const std::map<T, T>& tree_to_net() const { return x2y; }
  const std::map<T, T>& net_to_tree() const { return y2x; }
};

}  // open_spiel

#endif  // OPEN_SPIEL_UTILS_DATA_STRUCTURES_
