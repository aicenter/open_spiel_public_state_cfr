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

#include "open_spiel/papers_with_code/1906.06412.value_functions/metrics.h"

#include <chrono>

#include "open_spiel/algorithms/dispatch_policy.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/tabularize_bot.h"
#include <open_spiel/algorithms/infostate_tree.h>
#include <algorithms/best_response.h>
#include <game_transforms/turn_based_simultaneous_game.h>

namespace open_spiel {
namespace papers_with_code {

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.
namespace or_algs = algorithms::ortools;

class FullTrunkExplMetric : public Metric {
  std::vector<int> evaluate_iters_;
  std::vector<double> expls_;
  SubgameSolver* trunk_with_net_;
  or_algs::SequenceFormLpSpecification* whole_game_;

 public:
  FullTrunkExplMetric(std::vector<int> evaluate_iters,
                      SubgameSolver* trunk_with_net,
                      or_algs::SequenceFormLpSpecification* whole_game)
     : evaluate_iters_(std::move(evaluate_iters)),
       expls_(evaluate_iters_.size()),
       trunk_with_net_(trunk_with_net),
       whole_game_(whole_game) {}

  std::string name() const override { return "full_trunk_expl"; }

  void Reset() override { std::fill(expls_.begin(), expls_.end(), 0.); }

  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<Policy> eval_policy = trunk_with_net_->AveragePolicy();
    int j = 0;

    // Important!! We must reset all the bandits & other memory for proper eval.
    trunk_with_net_->Reset();
    for (int i = 1; i <= evaluate_iters_.back(); ++i) {
      trunk_with_net_->RunSimultaneousIterations(1);
      if (should_evaluate_at_iter(i)) {
        expls_[j++] = or_algs::TrunkExploitability(whole_game_, *eval_policy,
                                                   /*strategy_epsilon=*/0.);
        progress << '.' << std::flush;
      }
    }
  }

  void PrintHeader(std::ostream& os) const override {
    bool first = true;
    for (int iter: evaluate_iters_) {
      if (!first) os << ",";
      else first = false;
      os << "expl[" << iter << "]";
    }
  }

  void PrintMetric(std::ostream& os) const override {
    bool first = true;
    for (double expl: expls_) {
      if (!first) os << ",";
      else first = false;
      os << expl;
    }
  }

 private:
  bool should_evaluate_at_iter(int iter) const {
    for (auto j : evaluate_iters_)
      if (iter == j) return true;
    return false;
  }
};

class IigsApproxBrMetric : public Metric {
  std::unique_ptr<Bot> bot_;
  std::shared_ptr<const Game> orig_game_;
  std::shared_ptr<const Game> br_game_;
  std::shared_ptr<algorithms::InfostateTree> player_tree_;
  double br_;
 public:
  IigsApproxBrMetric(std::unique_ptr<Bot> bot,
                     std::shared_ptr<const goofspiel::GoofspielGame> game)
      : bot_(std::move(bot)),
        br_game_(LoadGameAsTurnBased(absl::StrCat("goofspiel("
          "players=2,"
          "num_turns=", game->NumTurns(), ","
          "num_cards=", game->NumCards(), ","
          "opponent_br_deck=True,"  // <-- This is important change.
          "imp_info=True,"
          "points_order=descending"
        ")"))),
        player_tree_(algorithms::MakeInfostateTree(
            *game, Player{0},
            algorithms::kNoMoveAheadLimit,
            algorithms::kStoreAllStatesPolicy)) {}
  std::string name() const override { return "br"; }
  void Reset() override {}
  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<TabularPolicy> policy =
        TabularizeOnlinePolicy(bot_.get(), player_tree_);
//    std::cout << policy->ToString() << "\n";
    algorithms::TabularBestResponse br(*br_game_, Player{1}, policy.get());
    br_ = br.Value("");
  }
  void PrintHeader(std::ostream& os) const override { os << "br"; }
  void PrintMetric(std::ostream& os) const override { os << br_; }
};


class ReplayVisitsMetric : public Metric {
  ExperienceReplay* replay_;
  int window_;
  double avg_visits_;
 public:
  ReplayVisitsMetric(ExperienceReplay* replay, int window)
      : replay_(replay), window_(window) {}

  std::string name() const override { return "replay_visits"; }

  void Evaluate(std::ostream& progress) override {
    double total_visits = 0;
    const int head = replay_->head();
    const std::vector<int>& visit_cnt = replay_->visit_cnt();
    const int size = replay_->size();
    SPIEL_CHECK_EQ(replay_->size(), visit_cnt.size());
    for (int i = 0; i < window_; ++i) {
      total_visits += visit_cnt[(size + head - i) % size];
    }
    avg_visits_ = total_visits / window_;
  }

  void PrintHeader(std::ostream& os) const override { os << name(); }
  void PrintMetric(std::ostream& os) const override { os << avg_visits_; }
  void Reset() override { avg_visits_ = 0;  }
};


class TrackTimeMetric : public Metric {
  std::chrono::time_point<std::chrono::system_clock> prev;
  std::chrono::duration<double> dt;
 public:
  TrackTimeMetric() : prev(std::chrono::system_clock::now()) {}
  std::string name() const override { return "time_elapsed"; }
  void Evaluate(std::ostream& progress) override {
    auto now = std::chrono::system_clock::now();
    dt = now - prev;
    prev = now;
  }

  void PrintHeader(std::ostream& os) const override { os << name(); }
  void PrintMetric(std::ostream& os) const override { os << dt.count(); }
  void Reset() override {}
};

class TrackLearningRate : public Metric {
  torch::optim::Optimizer* optimizer_;
  double lr;
 public:
  TrackLearningRate(torch::optim::Optimizer* optimizer)
    : optimizer_(optimizer) {}
  std::string name() const override { return "lr"; }
  void Evaluate(std::ostream& progress) override {
    for (auto &group : optimizer_->param_groups()) {
      if(group.has_options()) {
        auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
        lr = options.lr();
      }
    }
  }

  void PrintHeader(std::ostream& os) const override { os << name(); }
  void PrintMetric(std::ostream& os) const override { os << lr; }
  void Reset() override {}
};

std::unique_ptr<Metric> MakeFullTrunkExplMetric(
    std::vector<int> evaluate_iters, SubgameSolver* trunk_with_net,
    or_algs::SequenceFormLpSpecification* whole_game) {
  return std::make_unique<FullTrunkExplMetric>(std::move(evaluate_iters),
                                               trunk_with_net, whole_game);
}

std::unique_ptr<Metric> MakeIigsApproxBrMetric(
    std::unique_ptr<Bot> bot,
    std::shared_ptr<const goofspiel::GoofspielGame> game) {
  return std::make_unique<IigsApproxBrMetric>(std::move(bot), game);
}


std::unique_ptr<Metric> MakeReplayVisitsMetric(ExperienceReplay* replay,
                                               int window) {
  return std::make_unique<ReplayVisitsMetric>(replay, window);
}

std::unique_ptr<Metric> MakeTrackTimeMetric() {
  return std::make_unique<TrackTimeMetric>();
}

std::unique_ptr<Metric> MakeTrackLearningRate(torch::optim::Optimizer* optimizer) {
  return std::make_unique<TrackLearningRate>(optimizer);
}


void ComputeMetrics(std::vector<std::unique_ptr<Metric>>& metrics) {
  for (std::unique_ptr<Metric>& evaluator : metrics) {
    evaluator->Reset();
    std::cout << "# " << evaluator->name() << " " << std::flush;
    evaluator->Evaluate(std::cout);
    std::cout << std::endl;
  }
}
void PrintHeaders(const std::vector<std::unique_ptr<Metric>>& metrics) {
  for (const std::unique_ptr<Metric>& metric : metrics) {
    std::cout << ',';
    metric->PrintHeader(std::cout);
  }
}
void PrintMetrics(const std::vector<std::unique_ptr<Metric>>& metrics) {
  for (const std::unique_ptr<Metric>& metric : metrics) {
    std::cout << ',';
    metric->PrintMetric(std::cout);
  }
}


}  // namespace papers_with_code
}  // namespace open_spiel

