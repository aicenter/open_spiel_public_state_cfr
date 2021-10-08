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


#include "open_spiel/abseil-cpp/absl/hash/hash.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/solver.h"
#include "open_spiel/algorithms/bandits.h"
#include "open_spiel/algorithms/bandits_policy.h"

namespace open_spiel {
namespace papers_with_code {

namespace {

void CheckConsistency(const PublicState& s) {
  // All leaf nodes must be indeed leaf nodes and belong to correct players.
  // They should all be terminal or non-terminal.
  // The set of corresponding states must be the same across players.
  int num_terminals = 0, num_nonterminals = 0;

  using History = std::vector<Action>;
  std::unordered_set<History, absl::Hash<History>> state_histories;
  for (int pl = 0; pl < 2; ++pl) {
    for (const algorithms::InfostateNode* node : s.nodes[pl]) {
      SPIEL_CHECK_TRUE(!s.IsLeaf() || node->is_leaf_node());
      SPIEL_CHECK_EQ(node->tree().acting_player(), pl);
      if (node->type() == algorithms::kTerminalInfostateNode) num_terminals++;
      else num_nonterminals++;

      for (const std::unique_ptr<State>& state : node->corresponding_states()) {
        std::unique_ptr<std::vector<Action>> h;
        if (node->type() == algorithms::kTerminalInfostateNode) {
          h = std::make_unique<std::vector<Action>>(node->TerminalHistory());
        } else {
          h = std::make_unique<std::vector<Action>>(state->History());
        }
        if (pl == 0) {
          SPIEL_CHECK_TRUE(state_histories.find(*h) == state_histories.end());
          state_histories.insert(*h);
        } else {
          SPIEL_CHECK_TRUE(state_histories.find(*h) != state_histories.end());
        }
      }
    }
  }
  SPIEL_CHECK_FALSE(num_terminals > 0 && num_nonterminals > 0);
  // We must count terminals twice (2 players).
  SPIEL_CHECK_FALSE(num_terminals % 2 != 0 && num_nonterminals % 2 != 0);
  // All OK! Yay!
}

std::vector<std::unique_ptr<PublicStateContext>> MakeContexts(
    std::shared_ptr<Subgame> subgame,
    const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator) {
  std::vector<std::unique_ptr<PublicStateContext>> contexts;
  contexts.reserve(subgame->public_states.size());
  for (const PublicState& state : subgame->public_states) {
    if (!state.IsLeaf()) {
      contexts.push_back(nullptr);
    } else {
      SPIEL_DCHECK(CheckConsistency(state));
      if (state.IsTerminal()) {
        contexts.push_back(terminal_evaluator->CreateContext(state));
      } else {
        contexts.push_back(nonterminal_evaluator
                           ? nonterminal_evaluator->CreateContext(state)
                           : nullptr);
      }
    }
  }
  return contexts;
}

}  // namespace

SubgameSolver::SubgameSolver(
    std::shared_ptr<Subgame> subgame,
    const std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator,
    const std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
    const std::shared_ptr<std::mt19937> rnd_gen,
    const std::string& bandit_name,
    algorithms::PolicySelection save_values_policy,
    bool safe_resolving,
    bool beliefs_for_average,
    double noisy_values
) : subgame_(subgame),
    nonterminal_evaluator_(nonterminal_evaluator),
    terminal_evaluator_(terminal_evaluator),
    rnd_gen_(rnd_gen),
    safe_resolving_(safe_resolving),
    beliefs_for_average_(beliefs_for_average),
    noisy_values_(noisy_values),
    bandits_(algorithms::MakeBanditVectors(subgame_->trees, bandit_name)),
    reach_probs_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                  std::vector<double>(subgame_->trees[1]->num_leaves())}),
    cf_values_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                std::vector<double>(subgame_->trees[1]->num_leaves())}),
    contexts_(MakeContexts(subgame, nonterminal_evaluator, terminal_evaluator)),
    num_iterations_(0),
    init_save_values_(save_values_policy) {}

SubgameSolver::SubgameSolver(
    std::shared_ptr<Subgame> subgame,
    std::vector<algorithms::BanditVector> bandits
) : subgame_(subgame),
    nonterminal_evaluator_(nullptr),
    terminal_evaluator_(MakeTerminalEvaluator()),
    rnd_gen_(nullptr),
    safe_resolving_(false),
    beliefs_for_average_(false),
    noisy_values_(0.),
    bandits_(std::move(bandits)),
    reach_probs_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                  std::vector<double>(subgame_->trees[1]->num_leaves())}),
    cf_values_({std::vector<double>(subgame_->trees[0]->num_leaves()),
                std::vector<double>(subgame_->trees[1]->num_leaves())}),
    contexts_(MakeContexts(subgame, nullptr, terminal_evaluator_)),
    num_iterations_(0),
    init_save_values_(algorithms::kDefaultPolicySelection) {}

std::shared_ptr<Policy> SubgameSolver::AveragePolicy() {
  return std::make_shared<algorithms::BanditsAveragePolicy>(subgame()->trees,
                                                            bandits_);
}

std::shared_ptr<Policy> SubgameSolver::CurrentPolicy() {
  return std::make_shared<algorithms::BanditsCurrentPolicy>(subgame()->trees,
                                                            bandits_);
}

void SubgameSolver::RunSimultaneousIterations(int iterations) {
  for (int t = 0; t < iterations; ++t) {
    ++num_iterations_;

    // 1. Prepare initial reach probs, based on beliefs in initial state.
    std::array<std::vector<double>, 2>& beliefs = initial_state().beliefs;
    for (int pl = 0; pl < 2; ++pl) {
      std::copy(beliefs[pl].begin(), beliefs[pl].end(),
                reach_probs_[pl].begin());
    }

    // 2. Compute reach probs to the terminals.
    for (int pl = 0; pl < 2; ++pl) {
      TopDownCurrent(*subgame_->trees[pl], bandits_[pl],
                     absl::MakeSpan(reach_probs_[pl]), num_iterations_);
    }

    // Optionally instead of current reach probs, use average reach probs.
    // This corresponds to using CFR-AVE in [Appendix E, 1].
    //
    // [1] Combining Deep Reinforcement Learning and Search
    //     for Imperfect-Information Games
    //     Noam Brown, Anton Bakhtin, Adam Lerer, Qucheng Gong
    if (beliefs_for_average_) {
      // 1. Prepare initial reach probs, based on beliefs in initial state.
      for (int pl = 0; pl < 2; ++pl) {
        std::copy(beliefs[pl].begin(), beliefs[pl].end(),
                  reach_probs_[pl].begin());
      }
      // 2. Compute reach probs of avg strategy to the terminals.
      for (int pl = 0; pl < 2; ++pl) {
        TopDownAverage(*subgame_->trees[pl], bandits_[pl],
                       absl::MakeSpan(reach_probs_[pl]));
      }
    }

    // 3. Evaluate leaves using current reach probs.
    EvaluateLeaves();

    // 4. Propagate updated values up the tree.
    for (int pl = 0; pl < 2; ++pl) {
      BottomUp(*subgame_->trees[pl], bandits_[pl],
               absl::MakeSpan(cf_values_[pl]));
    }
    // Holds for oracle values, but not for the ones coming from NN (not yet).
//    SPIEL_DCHECK_FLOAT_NEAR(initial_state().Value(0),
//                            -initial_state().Value(1), 1e-6);

    if (init_save_values_ == algorithms::PolicySelection::kAveragePolicy) {
      IncrementallyAverageValuesInState(&initial_state());
    }
  }

  if (init_save_values_ == algorithms::PolicySelection::kCurrentPolicy) {
    CopyCurrentValuesToInitialState();
  }
}

void SubgameSolver::EvaluateLeaves() {
  SPIEL_CHECK_EQ(subgame()->public_states.size(), contexts_.size());
  for (int i = 0; i < subgame()->public_states.size(); ++i) {
    PublicState* state = &subgame()->public_states[i];
    if (!state->IsLeaf()) continue;
    PublicStateContext* context = contexts_[i].get();
    EvaluateLeaf(state, context);
  }
}

void SubgameSolver::EvaluateLeaf(PublicState* state,
                                 PublicStateContext* context) {
  SPIEL_CHECK_TRUE(state);
  SPIEL_CHECK_TRUE(state->IsLeaf());

  // 1. Prepare beliefs
  for (int pl = 0; pl < 2; pl++) {
    const int num_leaves = state->nodes[pl].size();
    for (int j = 0; j < num_leaves; ++j) {
      const algorithms::InfostateNode* leaf_node = state->nodes[pl][j];
      const int trunk_position = state->nodes_positions.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, subgame()->trees[pl]->num_leaves());
      // Copy reach prob (player belief) from the trunk
      // to the leaf public state->
      state->beliefs[pl][j] = reach_probs_[pl][trunk_position];
    }
  }

  // 2. Evaluate: compute cfvs.
  if (state->IsTerminal()) {
    SPIEL_CHECK_TRUE(terminal_evaluator_);
    terminal_evaluator_->EvaluatePublicState(state, context);
  } else {
    SPIEL_CHECK_TRUE(nonterminal_evaluator_);
    nonterminal_evaluator_->EvaluatePublicState(state, context);
  }

  // 3. Update cfvs for propagators.
  for (int pl = 0; pl < 2; pl++) {
    const int num_leaves = state->nodes[pl].size();
    for (int j = 0; j < num_leaves; ++j) {
      const algorithms::InfostateNode* leaf_node = state->nodes[pl][j];
      const int trunk_position = state->nodes_positions.at(leaf_node);
      SPIEL_DCHECK_GE(trunk_position, 0);
      SPIEL_DCHECK_LT(trunk_position, subgame()->trees[pl]->num_leaves());
      // Copy value from the leaf public state to the trunk.
      double noise = 0.;
      if (noisy_values_ > 0) {
        SPIEL_CHECK_TRUE(rnd_gen_);
        std::normal_distribution<double> dist(0, noisy_values_);
        noise = dist(*rnd_gen_);
      }
      cf_values_[pl][trunk_position] = state->values[pl][j] + noise;
    }
  }

  // 4. Incrementally update average CFVs.
  if (safe_resolving_) {
    for (int pl = 0; pl < 2; pl++) {
      const int num_leaves = state->nodes[pl].size();
      for (int j = 0; j < num_leaves; ++j) {
        state->average_values[pl][j] +=
            (state->values[pl][j] - state->average_values[pl][j])
                / num_iterations_;
      }
    }
  }
}

void SubgameSolver::Reset() {
  // Reset trunk
  num_iterations_ = 0;
  for (int pl = 0; pl < 2; ++pl) {
    std::fill(cf_values_[pl].begin(), cf_values_[pl].end(), 0.);
    std::fill(reach_probs_[pl].begin(), reach_probs_[pl].end(), 0.);
  }
  for (algorithms::BanditVector& bandits : bandits_) {
    for (algorithms::DecisionId id : bandits.range()) {
      bandits[id]->Reset();
    }
  }
  // Reset subgames
  for (int i = 0; i < subgame_->public_states.size(); ++i) {
    PublicState& state = subgame_->public_states[i];
    for (int pl = 0; pl < 2; ++pl) {
      // Conserve beliefs for initial state.
      if (!state.IsInitial()) {
        std::fill(state.beliefs[pl].begin(), state.beliefs[pl].end(), 0.);
      }
      std::fill(state.values[pl].begin(), state.values[pl].end(), 0.);
      std::fill(state.average_values[pl].begin(),
                state.average_values[pl].end(), 0.);
    }
    if (nonterminal_evaluator_.get()) {
      std::unique_ptr<PublicStateContext>& context = contexts_[i];
      if (!state.IsTerminal() && context.get()) {
        nonterminal_evaluator_->ResetContext(context.get());
      }
    }
  }
}

void SubgameSolver::ResetCumulValues() {
  for (int pl = 0; pl < 2; ++pl) {
    for (auto& nodes : subgame_->trees[pl]->nodes_at_depths()) {
      for (auto& node: nodes) node->cumul_value = 0;
    }
  }
}


void SubgameSolver::CopyCurrentValuesToInitialState() {
  for (int pl = 0; pl < 2; ++pl) {
    int branching = subgame()->trees[pl]->root_branching_factor();
    SPIEL_CHECK_EQ(initial_state().values[pl].size(), branching);
    std::copy(cf_values_[pl].begin(), cf_values_[pl].begin() + branching,
              initial_state().values[pl].begin());
  }
}

void SubgameSolver::IncrementallyAverageValuesInState(PublicState* state) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < state->values[pl].size(); ++i) {
      state->values[pl][i] +=
          (cf_values_[pl][i] - state->values[pl][i]) / num_iterations_;
    }
  }
}



}  // namespace papers_with_code
}  // namespace open_spiel
