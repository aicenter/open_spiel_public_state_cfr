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

#include "open_spiel/games/goofspiel.h"
#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/algorithms/dispatch_policy.h"
#include "open_spiel/algorithms/ortools/trunk_exploitability.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_architectures.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"

namespace open_spiel {
namespace papers_with_code {

struct Metric {
  virtual ~Metric() = default;
  virtual std::string name() const = 0;
  virtual void PrintHeader(std::ostream& os) const = 0;
  virtual void PrintMetric(std::ostream& os) const = 0;
  virtual void Reset() = 0;
  virtual void Evaluate(std::ostream& progress) = 0;
};

std::unique_ptr<Metric> MakeFullTrunkExplMetric(
    std::vector<int> evaluate_iters,
    SubgameSolver* trunk_with_net,
    algorithms::ortools::SequenceFormLpSpecification* whole_game);

std::unique_ptr<Metric> MakeIigsApproxBrMetric(
    std::unique_ptr<Bot> bot,
    std::shared_ptr<const goofspiel::GoofspielGame> game);

std::unique_ptr<Metric> MakeReplayVisitsMetric(
    ExperienceReplay* replay, int window);

std::unique_ptr<Metric> MakeTrackTimeMetric();
std::unique_ptr<Metric> MakeTrackLearningRate(torch::optim::Optimizer* optimizer);

void ComputeMetrics(std::vector<std::unique_ptr<Metric>>& metrics);
void PrintHeaders(const std::vector<std::unique_ptr<Metric>>& metrics);
void PrintMetrics(const std::vector<std::unique_ptr<Metric>>& metrics);

}  //  papers_with_code
}  //  open_spiel

#endif // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_TRAIN_EVAL_
