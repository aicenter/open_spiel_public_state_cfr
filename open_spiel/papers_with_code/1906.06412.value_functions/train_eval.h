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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRAIN_EVAL_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRAIN_EVAL_

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/algorithms/ortools/dl_oracle_evaluator.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"

namespace open_spiel {
namespace papers_with_code {

double TrainNetwork(ParticleValueNet* model, torch::Device* device,
                    torch::optim::Optimizer* optimizer,
                    BatchData* batch);

std::vector<double> EvaluateNetwork(
    std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks_with_net,
    algorithms::ortools::SequenceFormLpSpecification* whole_game,
    const std::vector<int>& evaluate_iters);

}  //  papers_with_code
}  //  open_spiel

#endif // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRAIN_EVAL_
