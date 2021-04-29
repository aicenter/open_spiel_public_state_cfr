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
#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame_factory.h"
#include "open_spiel/algorithms/dispatch_policy.h"

namespace open_spiel {
namespace papers_with_code {

using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.
namespace or_algs = algorithms::ortools;

class FullTrunkExplMetric : public Metric {
  std::vector<int> evaluate_iters_;
  std::vector<double> expls_;
  Subgame* trunk_with_net_;
  or_algs::SequenceFormLpSpecification* whole_game_;

 public:
  FullTrunkExplMetric(std::vector<int> evaluate_iters,
                      Subgame* trunk_with_net,
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
        expls_[j++] = or_algs::TrunkExploitability(whole_game_, *eval_policy);
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


class UniformPolicyForInfostateTrees : public Policy {
  const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees_;
 public:
  UniformPolicyForInfostateTrees(
      const std::vector<std::shared_ptr<algorithms::InfostateTree>>& trees)
      : trees_(trees) {}
  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    for (int pl = 0; pl < 2; ++pl) {
      const algorithms::InfostateNode* node =
          trees_[pl]->DecisionNodeFromInfostateString(info_state);
      if (node) {
        const std::vector<Action>& actions = node->legal_actions();
        const double p = 1. / actions.size();
        ActionsAndProbs ap;
        ap.reserve(actions.size());
        for (int i = 0; i < actions.size(); ++i) ap.push_back({actions[i], p});
        return ap;
      }
    }
    
    return {};  // Node not found, return empty vector.
  }
};

const std::string kUseBanditsForCfr = "RegretMatchingPlus";
const int kTrunkNoMoveLimit = 1000;

class SparseRootsExplMetric : public Metric {
  std::vector<int> evaluate_iters_;
  std::vector<double> expls_;

  std::unique_ptr<SparseTrunk> sparse_eq_trunk_with_net_;

  std::unique_ptr<DispatchPolicy> eval_policy_;
  std::unique_ptr<SparseTrunk> eval_trunk_;
  std::unique_ptr<or_algs::SequenceFormLpSpecification> eval_lp_;
  std::shared_ptr<UniformPolicyForInfostateTrees> uniform_policy_;
  
 public:
  SparseRootsExplMetric(
      // Needed to construct proper evaluations.
      SubgameFactory* factory,
      or_algs::SequenceFormLpSpecification* whole_game,
      // Settings.
      std::vector<int> evaluate_iters,
      int roots_depth, double support_threshold, bool prune_chance_histories)
      : evaluate_iters_(std::move(evaluate_iters)),
        expls_(evaluate_iters_.size()),
        uniform_policy_(
            std::make_shared<UniformPolicyForInfostateTrees>(whole_game->trees())) {
    
    auto[eq_policy, game_value] = or_algs::MakeEquilibriumPolicy(whole_game);
    sparse_eq_trunk_with_net_ = MakeSparseTrunkWithEqSupport(
        eq_policy, factory->game, factory->infostate_observer,
        factory->public_observer,
        roots_depth, factory->max_move_ahead_limit,
        factory->leaf_evaluator, factory->terminal_evaluator, kUseBanditsForCfr,
        support_threshold, prune_chance_histories);

    // The sparse trunk is constructed as replacing the players' equilibrium
    // policies as a chance in the upper game. By constructing the trunk with no
    // move limit, we make an evaluation trunk.
    eval_trunk_ = MakeSparseTrunkWithEqSupport(
        eq_policy, factory->game,
        factory->infostate_observer, factory->public_observer,
        roots_depth, kTrunkNoMoveLimit,
        nullptr, factory->terminal_evaluator,
        kUseBanditsForCfr, /*support_threshold=*/1e-5, 
        /*prune_chance_histories=*/false);
    eval_lp_ = std::make_unique<or_algs::SequenceFormLpSpecification>(
             eval_trunk_->dlcfr->trees(), "CLP");
    eval_policy_ = std::make_unique<DispatchPolicy>();
    eval_policy_->AddDispatch(sparse_eq_trunk_with_net_->fixate_infostates,
                              sparse_eq_trunk_with_net_->dlcfr->AveragePolicy());
    eval_policy_->AddDispatch(sparse_eq_trunk_with_net_->uniform_infostates,
                              uniform_policy_);
  }

  std::string name() const override { return "sparse_roots_expl"; }

  void Evaluate(std::ostream& progress) override {
    int j = 0;
    
    // Important!! We must reset all the bandits & other memory for proper eval.
    sparse_eq_trunk_with_net_->dlcfr->Reset();

    for (int i = 1; i <= evaluate_iters_.back(); ++i) {
      Subgame* trunk_with_net =
          sparse_eq_trunk_with_net_->dlcfr.get();
      trunk_with_net->RunSimultaneousIterations(1);
      if (should_evaluate_at_iter(i)) {
        expls_[j++] = or_algs::TrunkExploitability(eval_lp_.get(),
                                                   *eval_policy_);
        progress << '.' << std::flush;
      }
    }
  }

  void PrintHeader(std::ostream& os) const override {
    bool first = true;
    for (int iter: evaluate_iters_) {
      if (!first) os << ",";
      else first = false;
      os << name() << "[" << iter << "]";
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

  void Reset() override { std::fill(expls_.begin(), expls_.end(), 0.); }

 private:
  bool should_evaluate_at_iter(int iter) const {
    for (auto j : evaluate_iters_)
      if (iter == j) return true;
    return false;
  }
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

std::unique_ptr<Metric> MakeFullTrunkExplMetric(
    std::vector<int> evaluate_iters, Subgame* trunk_with_net,
    or_algs::SequenceFormLpSpecification* whole_game) {
  return std::make_unique<FullTrunkExplMetric>(std::move(evaluate_iters),
                                               trunk_with_net, whole_game);
}

std::unique_ptr<Metric> MakeSparseRootsExplMetric(
    SubgameFactory* factory,
    or_algs::SequenceFormLpSpecification* whole_game,
    std::vector<int> evaluate_iters,
    int roots_depth, double support_threshold, bool prune_chance_histories) {
  return std::make_unique<SparseRootsExplMetric>(
      factory, whole_game, evaluate_iters,
      roots_depth, support_threshold, prune_chance_histories);
}

std::unique_ptr<Metric> MakeReplayVisitsMetric(ExperienceReplay* replay,
                                               int window) {
  return std::make_unique<ReplayVisitsMetric>(replay, window);
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

