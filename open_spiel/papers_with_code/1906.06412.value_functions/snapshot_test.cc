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

#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"

namespace open_spiel {
namespace papers_with_code {

void TestFindSnapshot() {
  std::string current_dir = __FILE__;
  current_dir.resize(current_dir.rfind("/"));
  SPIEL_CHECK_FALSE(
      FindSnapshot(absl::StrCat(current_dir, "/snapshots/test")).empty());
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestFindSnapshot();
}
