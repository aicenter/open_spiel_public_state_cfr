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
#include "open_spiel/papers_with_code/1906.06412.value_functions/dispatch_policy.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;
using namespace torch::indexing;  // Load all of the Slice, Ellipsis, etc.

//auto [eq_policy, game_value] = ortools::MakeEquilibriumPolicy(&whole_game);
//std::vector<std::unique_ptr<SparseTrunk>> sparse_eq_trunk_with_net;
//sparse_eq_trunk_with_net.push_back(MakeSparseTrunkWithEqSupport(
//    eq_policy, t->game, t->infostate_observer, t->public_observer,
//    roots_depth, t->trunk_depth,
//    net_evaluator, t->terminal_evaluator, use_bandits_for_cfr,
//    support_threshold, prune_chance_histories));
//std::cout << "# Equilibrium sparse trunk:"
//<< "\n# - Infostate leaves: "
//<< sparse_eq_trunk_with_net.back()->dlcfr->trees()[0]->num_leaves()
//<< "\n# - Eval infostates: "
//<< sparse_eq_trunk_with_net.back()->fixate_infostates.size()
//<< "\n# Full trunk infostate leaves: "
//<< t->fixable_trunk_with_oracle->trees()[0]->num_leaves() << "\n";
//
//// The sparse trunk is constructed as replacing the players' equilibrium
//// policies as a chance in the upper game. By constructing the trunk with no
//// move limit, we make an evaluation trunk.
//std::unique_ptr<SparseTrunk> eval_trunk =
//    MakeSparseTrunkWithEqSupport(eq_policy, t->game,
//                                 t->infostate_observer, t->public_observer,
//                                 roots_depth, no_move_limit,
//                                 nullptr, t->terminal_evaluator,
//                                 use_bandits_for_cfr, 1e-5, false);
//ortools::SequenceFormLpSpecification eq_fixed_as_chance_lp(
//    eval_trunk->dlcfr->trees(), "CLP");

//std::unique_ptr<Evaluator> MakeSparseTrunkEvaluator() {
//  std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks_with_net,
//  ortools::SequenceFormLpSpecification* whole_game,
//  const std::vector<int>& evaluate_iters) {
//
//    auto should_evaluate = [&](int i){
//        for (auto j : evaluate_iters) {
//          if (i == j) return true;
//        }
//        return false;
//    };
//
//    std::vector<double> expls;
//    expls.reserve(evaluate_iters.size());
//    int trunk_iters = *std::max_element(evaluate_iters.begin(),
//                                        evaluate_iters.end());
//
//    auto uniform_policy = std::make_shared<UniformISTreePolicy>(
//        whole_game->trees());
//    DispatchPolicy eval_policy;
//    for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//      // Important!! We must reset all the bandits & other memory for proper eval.
//      sparse_trunk->dlcfr->Reset();
//      eval_policy.AddDispatch(sparse_trunk->fixate_infostates,
//                              sparse_trunk->dlcfr->AveragePolicy());
//      eval_policy.AddDispatch(sparse_trunk->uniform_infostates,
//                              uniform_policy);
//    }
//
//    for (int i = 1; i <= trunk_iters; ++i) {
//      for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//        dlcfr::DepthLimitedCFR* trunk_with_net = sparse_trunk->dlcfr.get();
//        ++trunk_with_net->num_iterations_;
//        trunk_with_net->UpdateReachProbs();
//        trunk_with_net->EvaluateLeaves();
//      }
//
//      if (should_evaluate(i)) {
//        expls.push_back(ortools::TrunkExploitability(whole_game, eval_policy));
//        std::cout << '.' << std::flush;
//      }
//
//      for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//        sparse_trunk->dlcfr->UpdateTrunk();
//      }
//    }
//}

class FullTrunkExplMetric : public Metric {
  std::vector<int> evaluate_iters_;
  std::vector<double> expls_;
  dlcfr::DepthLimitedCFR* trunk_with_net_;
  ortools::SequenceFormLpSpecification* whole_game_;

 public:
  FullTrunkExplMetric(std::vector<int> evaluate_iters,
                      dlcfr::DepthLimitedCFR* trunk_with_net,
                      ortools::SequenceFormLpSpecification* whole_game)
     : evaluate_iters_(std::move(evaluate_iters)),
       expls_(evaluate_iters_.size()),
       trunk_with_net_(trunk_with_net),
       whole_game_(whole_game) {}

  std::string name() const override { return "full_trunk_expl"; }

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

  void Reset() override { std::fill(expls_.begin(), expls_.end(), 0.); }

  void Evaluate(std::ostream& progress) override {
    std::shared_ptr<Policy> eval_policy = trunk_with_net_->AveragePolicy();
    int j = 0;

    // Important!! We must reset all the bandits & other memory for proper eval.
    trunk_with_net_->Reset();

    for (int i = 1; i <= evaluate_iters_.back(); ++i) {
      ++trunk_with_net_->num_iterations_;
      trunk_with_net_->UpdateReachProbs();
      trunk_with_net_->EvaluateLeaves();

      if (should_evaluate_at_iter(i)) {
        expls_[j++] = ortools::TrunkExploitability(whole_game_, *eval_policy);
        progress << '.' << std::flush;
      }
      trunk_with_net_->UpdateTrunk();
    }
  }

 private:
  bool should_evaluate_at_iter(int iter) const {
    for (auto j : evaluate_iters_)
      if (iter == j) return true;
    return false;
  }
};

std::unique_ptr<Metric> MakeFullTrunkExplMetric(
    std::vector<int> evaluate_iters, dlcfr::DepthLimitedCFR* trunk_with_net,
    ortools::SequenceFormLpSpecification* whole_game) {
  return std::make_unique<FullTrunkExplMetric>(std::move(evaluate_iters),
                                               trunk_with_net, whole_game);
}

void ComputeMetrics(std::vector<std::unique_ptr<Metric>>& metrics) {
  for (std::unique_ptr<Metric>& evaluator : metrics) {
    evaluator->Reset();
    std::cout << "# " << evaluator->name() << " " << std::flush;
    evaluator->Evaluate(std::cout);
    std::cout << std::endl;
  }
}

//class UniformISTreePolicy : public Policy {
//  const std::vector<std::shared_ptr<InfostateTree>>& trees_;
// public:
//  UniformISTreePolicy(const std::vector<std::shared_ptr<InfostateTree>>& trees)
//    : trees_(trees) {}
//  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
//    for (int pl = 0; pl < 2; ++pl) {
//      const InfostateNode* node =
//          trees_[pl]->DecisionNodeFromInfostateString(info_state);
//      if (node) {
//        const std::vector<Action>& actions = node->legal_actions();
//        const double p = 1. / actions.size();
//        ActionsAndProbs ap;
//        ap.reserve(actions.size());
//        for (int i = 0; i < actions.size(); ++i) {
//          ap.push_back({actions[i], p});
//        }
//        return ap;
//      }
//    }
//    return {};
//  }
//};
//
//
//std::vector<double> ComputeMetrics(
//    std::vector<std::unique_ptr<SparseTrunk>>& sparse_trunks_with_net,
//    ortools::SequenceFormLpSpecification* whole_game,
//    const std::vector<int>& evaluate_iters) {
//
//  auto should_evaluate = [&](int i){
//    for (auto j : evaluate_iters) {
//      if (i == j) return true;
//    }
//    return false;
//  };
//
//  std::vector<double> expls;
//  expls.reserve(evaluate_iters.size());
//  int trunk_iters = *std::max_element(evaluate_iters.begin(),
//                                      evaluate_iters.end());
//
//  auto uniform_policy = std::make_shared<UniformISTreePolicy>(
//      whole_game->trees());
//  DispatchPolicy eval_policy;
//  for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//    // Important!! We must reset all the bandits & other memory for proper eval.
//    sparse_trunk->dlcfr->Reset();
//    eval_policy.AddDispatch(sparse_trunk->fixate_infostates,
//                            sparse_trunk->dlcfr->AveragePolicy());
//    eval_policy.AddDispatch(sparse_trunk->uniform_infostates,
//                            uniform_policy);
//  }
//
//  for (int i = 1; i <= trunk_iters; ++i) {
//    for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//      dlcfr::DepthLimitedCFR* trunk_with_net = sparse_trunk->dlcfr.get();
//      ++trunk_with_net->num_iterations_;
//      trunk_with_net->UpdateReachProbs();
//      trunk_with_net->EvaluateLeaves();
//    }
//
//    if (should_evaluate(i)) {
//      expls.push_back(ortools::TrunkExploitability(whole_game, eval_policy));
//      std::cout << '.' << std::flush;
//    }
//
//    for (std::unique_ptr<SparseTrunk>& sparse_trunk: sparse_trunks_with_net) {
//      sparse_trunk->dlcfr->UpdateTrunk();
//    }
//  }
//
//  return expls;
//}

}  // namespace papers_with_code
}  // namespace open_spiel

