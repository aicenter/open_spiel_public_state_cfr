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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EVALUATOR_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EVALUATOR_

#include "open_spiel/algorithms/bandits_policy.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

// Derived classes specify members as they need for their specific
// public state evaluators.
//
// It is **strongly advised** to make the contexts immutable, or if the state
// evaluator mutates the context for the purposes of its computation, it should
// restore it back to the original. This is because it is beneficial to share
// the same context between multiple evaluator implementations (for example
// the sequence-form oracle evaluators). If mutation were to happen, each
// evaluator then must take this into account, making implementation more
// difficult.
struct PublicStateContext {
  // Make sure PublicStateContext is polymorphic.
  virtual ~PublicStateContext() = default;
};

// Public state evaluator can create appropriate contexts for later evaluation
// of public states. The derived classes should down_cast the context as needed.
// Beliefs and values are saved within the public state.
class PublicStateEvaluator {
 public:
  virtual ~PublicStateEvaluator() = default;
  virtual std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const { return nullptr; };
  virtual void ResetContext(PublicStateContext* context) const {}
  virtual void EvaluatePublicState(
      PublicState* public_state, PublicStateContext* context) const = 0;
};

// -- Terminal evaluator -------------------------------------------------------

struct TerminalPublicStateContext final : public PublicStateContext {
  // Map from player 0 index (key) to player 1 (value).
  std::vector<int> permutation;
  // For the player 0 and already multiplied by chance reach probs.
  std::vector<double> utilities;
  explicit TerminalPublicStateContext(const PublicState& state);
};

class TerminalEvaluator final : public PublicStateEvaluator {
 public:
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void EvaluatePublicState(
      PublicState* state, PublicStateContext* context) const override;
};


// -- Dummy evaluator ----------------------------------------------------------

// Evaluator that does nothing.
struct DummyEvaluator : public PublicStateEvaluator {
  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override { return nullptr; };
  void ResetContext(PublicStateContext* context) const override {};
  void EvaluatePublicState(PublicState* public_state,
                           PublicStateContext* context) const override {};
};


// -- CFR evaluator ------------------------------------------------------------


struct CFREvaluator : public PublicStateEvaluator {
  std::shared_ptr<const Game> game;
  int depth_limit;
  std::shared_ptr<const PublicStateEvaluator> nonterminal_evaluator;
  std::shared_ptr<const PublicStateEvaluator> terminal_evaluator;
  std::shared_ptr<Observer> public_observer;
  std::shared_ptr<Observer> infostate_observer;
  bool reset_subgames_on_evaluation = true;
  int num_cfr_iterations;
  std::string bandit_name = "RegretMatchingPlus";
  algorithms::PolicySelection save_values_policy = algorithms::kDefaultPolicySelection;

  CFREvaluator(std::shared_ptr<const Game> game, int depth_limit,
               std::shared_ptr<const PublicStateEvaluator> leaf_evaluator,
               std::shared_ptr<const PublicStateEvaluator> terminal_evaluator,
               std::shared_ptr<Observer> public_observer,
               std::shared_ptr<Observer> infostate_observer,
               int cfr_iterations = 100);

  std::unique_ptr<PublicStateContext> CreateContext(
      const PublicState& state) const override;
  void ResetContext(PublicStateContext* context) const override;
  void EvaluatePublicState(PublicState* state,
                           PublicStateContext* context) const override;
};


std::shared_ptr<PublicStateEvaluator> MakeTerminalEvaluator();
std::shared_ptr<PublicStateEvaluator> MakeDummyEvaluator();
// CFR evaluator that makes a large number of iterations.
std::shared_ptr<PublicStateEvaluator> MakeApproxOracleEvaluator(
    std::shared_ptr<const Game> game, int cfr_iterations = 100000);



}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_EVALUATOR_
