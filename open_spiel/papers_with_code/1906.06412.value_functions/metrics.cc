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
#include "open_spiel/algorithms/expected_returns.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/tabularize_bot.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/infostate_tree_br.h"
#include <open_spiel/algorithms/infostate_tree.h>
#include <game_transforms/turn_based_simultaneous_game.h>

namespace open_spiel {
namespace papers_with_code {

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.
namespace or_algs = algorithms::ortools;

class FullTrunkExplMetric : public Metric {
  std::vector<int> evaluate_iters_;
  std::vector<double> expls_;
  SubgameSolver* trunk_with_vf_;
  or_algs::SequenceFormLpSpecification* whole_game_;

 public:
  FullTrunkExplMetric(std::vector<int> evaluate_iters,
                      SubgameSolver* trunk_with_vf,
                      or_algs::SequenceFormLpSpecification* whole_game)
     : evaluate_iters_(std::move(evaluate_iters)),
       expls_(evaluate_iters_.size()),
       trunk_with_vf_(trunk_with_vf),
       whole_game_(whole_game) {}

  std::string name() const override { return "full_trunk_expl"; }

  void Reset() override { std::fill(expls_.begin(), expls_.end(), 0.); }

  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<Policy> eval_policy = trunk_with_vf_->AveragePolicy();
    int j = 0;

    // Important!! We must reset all the bandits & other memory for proper eval.
    trunk_with_vf_->Reset();
    for (int i = 1; i <= evaluate_iters_.back(); ++i) {
      trunk_with_vf_->RunSimultaneousIterations(1);
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

class IigsBrMetric : public Metric {
  std::unique_ptr<Bot> bot_;
  std::shared_ptr<const Game> br_game_;
  std::shared_ptr<const Game> turn_br_game_;
  absl::optional<int> max_actions_ = {};
  std::vector<std::shared_ptr<algorithms::InfostateTree>> player_trees_;
  double br_;
  double returns_;
 public:
  IigsBrMetric(std::unique_ptr<Bot> bot,
               std::shared_ptr<const goofspiel::GoofspielGame> game,
               bool approx_response)
      : bot_(std::move(bot)),
        br_game_(approx_response
          ? LoadGame(absl::StrCat("goofspiel("
                "players=2,"
                "num_turns=", game->NumTurns(), ","
                "num_cards=", game->NumCards(), ","
                "opponent_br_deck=True,"  // <-- This is important change.
                "imp_info=True,"
                "points_order=descending"
              ")"))
          : game
        ),
        turn_br_game_(ConvertToTurnBased(*br_game_)),
        max_actions_(approx_response
          ? absl::optional<int>{game->NumTurns() + 1}
          : absl::optional<int>{}),
        player_trees_(algorithms::MakeInfostateTrees(
            *br_game_,
            algorithms::kNoMoveAheadLimit,
            algorithms::kStoreAllStatesPolicy)) {}
  std::string name() const override { return "br"; }
  void Reset() override {}
  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<TabularPolicy> policy =
        TabularizeOnlinePolicy(bot_.get(), player_trees_[0], max_actions_);
    progress << '.';

    br_ = BestResponse(player_trees_, *policy)[1];
    progress << '.';

    auto uniform = GetUniformPolicy(*br_game_);
    std::vector<double> returns =
        algorithms::ExpectedReturns(*br_game_->NewInitialState(),
                                    {policy.get(), &uniform}, 1000);
    returns_ = returns[0];
    progress << '.';
  }
  void PrintHeader(std::ostream& os) const override { os << "br,returns"; }
  void PrintMetric(std::ostream& os) const override { os << br_ << ',' << returns_; }
};


class BrMetric : public Metric {
  std::unique_ptr<Bot> bot_;
  std::shared_ptr<const Game> br_game_;
  std::shared_ptr<const Game> turn_br_game_;
  std::vector<std::shared_ptr<algorithms::InfostateTree>> player_trees_;
  double br_;
  double returns_;
 public:
  BrMetric(std::unique_ptr<Bot> bot, std::shared_ptr<const Game> game)
      : bot_(std::move(bot)),
        br_game_(game),
        turn_br_game_(game->GetType().dynamics == GameType::Dynamics::kSimultaneous
                      ? ConvertToTurnBased(*br_game_)
                      : br_game_),
        player_trees_(algorithms::MakeInfostateTrees(
            *br_game_,
            algorithms::kNoMoveAheadLimit,
            algorithms::kStoreAllStatesPolicy)) {}
  std::string name() const override { return "br"; }
  void Reset() override {}
  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<TabularPolicy> policy =
        TabularizeOnlinePolicy(bot_.get(), player_trees_[0], absl::nullopt);
    progress << '.';

    br_ = BestResponse(player_trees_, *policy)[1];
    progress << '.';

    auto uniform = GetUniformPolicy(*turn_br_game_);
    std::vector<double> returns =
        algorithms::ExpectedReturns(*turn_br_game_->NewInitialState(),
                                    {policy.get(), &uniform}, 1000);
    returns_ = returns[0];
    progress << '.';
  }
  void PrintHeader(std::ostream& os) const override { os << "br,returns"; }
  void PrintMetric(std::ostream& os) const override { os << br_ << ',' << returns_; }
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


class ValidationLossMetric : public Metric {
  ExperienceReplay validation_data_;
  std::shared_ptr<ValueNet> model_;
  torch::Device* device_;
  float loss_;
 public:
  ValidationLossMetric(ReplayFiller filler,
                       std::shared_ptr<ValueNet> model,
                       torch::Device* device,
                       ReplayFillerPolicy fill_policy,
                       int num_experiences)
    : validation_data_(num_experiences,
                       filler.dims->point_input_size(),
                       filler.dims->point_output_size()),
      model_(model),
      device_(device) {
    // Modify local filler copy.
    filler.replay = &validation_data_;
    filler.CreateExperiences(fill_policy, num_experiences);
  }
  std::string name() const override { return "val_loss"; }
  void Evaluate(std::ostream& progress) override {
    SPIEL_DCHECK_FALSE(model_->is_training());
    torch::NoGradGuard no_grad_guard;  // We run only inference.

    torch::Tensor data = validation_data_.data.to(*device_);
    torch::Tensor target = model_->PrepareTarget(&validation_data_).to(*device_);
    SPIEL_DCHECK_TRUE(torch::isfinite(data).all().item<bool>());
    SPIEL_DCHECK_TRUE(torch::isfinite(target).all().item<bool>());
    torch::Tensor output = model_->forward(data);
    SPIEL_DCHECK_TRUE(torch::isfinite(output).all().item<bool>());
    loss_ = torch::mse_loss(output, target).item<float>();
  }

  void PrintHeader(std::ostream& os) const override { os << name(); }
  void PrintMetric(std::ostream& os) const override { os << loss_; }
  void Reset() override {}
};

std::unique_ptr<Metric> MakeFullTrunkExplMetric(
    std::vector<int> evaluate_iters, SubgameSolver* trunk_with_vf,
    or_algs::SequenceFormLpSpecification* whole_game) {
  return std::make_unique<FullTrunkExplMetric>(std::move(evaluate_iters),
                                               trunk_with_vf, whole_game);
}

std::unique_ptr<Metric> MakeIigsBrMetric(
    std::unique_ptr<Bot> bot,
    std::shared_ptr<const goofspiel::GoofspielGame> game,
    bool approx_response) {
  return std::make_unique<IigsBrMetric>(std::move(bot), game, approx_response);
}

std::unique_ptr<Metric> MakeBrMetric(std::unique_ptr<Bot> bot,
                                     std::shared_ptr<const Game> game) {
  return std::make_unique<BrMetric>(std::move(bot), game);
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

std::unique_ptr<Metric> MakeValidationLossMetric(const ReplayFiller& filler,
                                                 std::shared_ptr<ValueNet> model,
                                                 torch::Device* device,
                                                 ReplayFillerPolicy fill_policy,
                                                 int num_experiences) {
  return std::make_unique<ValidationLossMetric>(
      filler, model, device, fill_policy, num_experiences);
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

