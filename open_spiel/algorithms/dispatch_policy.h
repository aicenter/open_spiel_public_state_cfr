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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_POLICY_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_POLICY_

#include "open_spiel/policy.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"

namespace open_spiel {
namespace papers_with_code {

class DispatchPolicy : public Policy {
  std::map<std::string, std::shared_ptr<Policy>> dispatch_table_;
 public:
  DispatchPolicy() {}
  DispatchPolicy(const std::string& infostate, std::shared_ptr<Policy> policy) {
    AddDispatch(infostate, policy);
  }
  DispatchPolicy(const std::vector<std::string>& infostates,
                 std::shared_ptr<Policy> policy) {
    AddDispatch(infostates, policy);
  }
  void AddDispatch(const std::vector<std::string>& infostates,
                   std::shared_ptr<Policy> policy) {
    for (const std::string& infostate : infostates) {
      AddDispatch(infostate, policy);
    }
  }
  void
  AddDispatch(const std::string& infostate, std::shared_ptr<Policy> policy) {
    SPIEL_CHECK_TRUE(dispatch_table_.find(infostate) == dispatch_table_.end());
    dispatch_table_[infostate] = policy;
  }
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    auto it = dispatch_table_.find(info_state);
    if (it == dispatch_table_.end()) {
      return {};
    } else {
      auto policy = it->second->GetStatePolicy(info_state);
      SPIEL_CHECK_FALSE(policy.empty());
      return policy;
    }
  }
};

}  // papers_with_code
}  // open_spiel



#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TABULARIZE_POLICY_
