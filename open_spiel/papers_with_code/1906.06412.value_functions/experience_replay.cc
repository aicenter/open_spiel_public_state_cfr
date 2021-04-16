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

#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/infostate_dl_cfr.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"


namespace open_spiel {
namespace papers_with_code {

namespace {

double PlayerReachProb(const algorithms::InfostateNode* node,
                       const algorithms::BanditVector& bandits, double prob) {
  using Bandit = algorithms::bandits::Bandit;

  if (node->is_root_node()) return prob;
  if (node->parent()->is_root_node()) return prob;
  if (node->parent()->type() == algorithms::kDecisionInfostateNode) {
    Bandit* bandit = bandits[node->parent()->decision_id()].get();
    prob *= bandit->current_strategy()[node->incoming_index()];
  }
  return PlayerReachProb(node->parent(), bandits, prob);
}

void UpdateBeliefs(PublicState& state,
                   const std::vector<algorithms::BanditVector>& bandits) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < state.nodes[pl].size(); ++i) {
      const algorithms::InfostateNode* node = state.nodes[pl][i];
      state.beliefs[pl][i] = PlayerReachProb(node, bandits[pl], 1.);
    }
  }
}

}  // namespace

ParticleDataPoint ExperienceReplay::AddExperience(const ParticleDims& dims) {
  ParticleDataPoint point = point_at(head_, dims);
  AdvanceHead();
  return point;
}

PositionalDataPoint ExperienceReplay::AddExperience(const PositionalDims& dims) {
  PositionalDataPoint point = point_at(head_, dims);
  AdvanceHead();
  return point;
}

void ExperienceReplay::AdvanceHead() {
  visit_cnt_[head_] = 0;
  ++head_;
  if (head_ >= size()) {
    head_ = 0;
    ++overflow_cnt_;
  }
}

void ExperienceReplay::SampleBatch(BatchData* batch, std::mt19937& rnd_gen) {
  // Do not sample non-filled experiences.
  const int n = overflow_cnt_ == 0 ? head_ : size();
  const int k = batch->size();
  SPIEL_CHECK_GE(n, k);

  // Full gradient descent -- no batching.
  if (batch->size() == size()) {
    batch->data = data;
    batch->target = target;
    return;
  }

  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rnd_gen);

  for (int i = 0; i < k; ++i) {
    ++visit_cnt_[perm[i]];
    batch->data[i].copy_(data[perm[i]]);
    batch->target[i].copy_(target[perm[i]]);
  }
}

ReplayFillerPolicy GetReplayFillerPolicy(const std::string& s) {
  if (s == "nothing")           return kNothing;
  if (s == "trunk_dlcfr")       return kTrunkDlcfr;
  if (s == "trunk_random")      return kTrunkRandom;
  if (s == "pbs_random")        return kPbsRandom;
  if (s == "sparse_pbs_random") return kSparsePbsRandom;
  SpielFatalError("Exhausted pattern match: ReplayFillerPolicy.");
}

void ReplayFiller::AddExperience(const PublicState& leaf,
                                 const NetContext* net_context) {
  switch (arch) {
    case NetArchitecture::kParticle: {
      auto particle_dims = open_spiel::down_cast<const ParticleDims&>(*dims);
      ParticleDataPoint data_point = replay->AddExperience(particle_dims);
      WriteParticleDataPoint(leaf, particle_dims, &data_point,
                             factory->hand_observer,
                             randomizer->rnd_gen, shuffle_input_output);
      break;
    }
    case NetArchitecture::kPositional: {
      auto pos_dims = open_spiel::down_cast<const PositionalDims&>(*dims);
      PositionalDataPoint data_point = replay->AddExperience(pos_dims);
      SPIEL_CHECK_TRUE(net_context);
      WritePositionalDataPoint(leaf, *net_context, pos_dims, &data_point);
      break;
    }
  }
}

void ReplayFiller::AddExperiencesFromPublicStates(
    const std::vector<algorithms::dlcfr::PublicState>& states) {
  for (int i = 0; i < states.size(); ++i) {
    const algorithms::dlcfr::PublicState& state = states[i];
    // Add experiences only for non-terminals leaves.
    if (state.IsTerminal() || !state.IsLeaf()) continue;
    auto context = factory->leaf_evaluator->CreateContext(state);
    auto net_context = open_spiel::down_cast<NetContext*>(context.get());
    AddExperience(state, net_context);
  }
}

void ReplayFiller::AddTrunkRandomPbsSolution() {
  Subgame* fixable_trunk_with_oracle = reuse->GetFixableTrunkWithOracle();
  int num_states = fixable_trunk_with_oracle->public_states().size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);

  // Reset trunk.
  fixable_trunk_with_oracle->Reset();
  // Randomize strategy in the trunk.
  randomizer->Randomize(fixable_trunk_with_oracle->bandits());
  // Compute the reach probs from the trunk.
  fixable_trunk_with_oracle->UpdateReachProbs();
  // Do not call bottom-up, just evaluate leaves.
  fixable_trunk_with_oracle->EvaluateLeaves();
  // Pick some valid public state.
  // Loop until we find one. There should be always one -- or perhaps
  // the trunk is too deep, getting into only terminal states.
  PublicState* state = nullptr;
  int pick_public_state;
  while (!state || state->IsTerminal() || state->IsInitial()) {
    pick_public_state = public_state_dist(*randomizer->rnd_gen);
    state = &fixable_trunk_with_oracle->public_states()[pick_public_state];
  }

  // Add experience for that state.
  auto context = factory->leaf_evaluator->CreateContext(*state);
  auto net_context = open_spiel::down_cast<NetContext*>(context.get());
  AddExperience(*state, net_context);
}

// The network should imitate DL-CFR at each iteration
// when we use this generation method.
void ReplayFiller::FillReplayWithTrunkDlCfrPbsSolutions() {
  Subgame* iterable_trunk_with_oracle = reuse->GetIterableTrunkWithOracle();
  SequenceFormLpSpecification* sf_lp = reuse->GetSfLp();

  std::cout << "# Computing reference expls for given trunk iterations.\n";
  int num_non_terminal_leaves = 0;
  for (auto& state: iterable_trunk_with_oracle->public_states()) {
    if (!state.IsTerminal() && state.IsLeaf()) num_non_terminal_leaves++;
  }
  SPIEL_CHECK_GT(num_non_terminal_leaves, 0);
  int num_trunks = replay->size() / num_non_terminal_leaves;
  // This replay filling is intended only for special test use cases -- you
  // should know the size of the buffer before hand.
  SPIEL_CHECK_EQ(num_trunks * num_non_terminal_leaves, replay->size());

  std::cout << "# <ref_expl>\n";
  std::cout << "# trunk_iter,expl\n";

  iterable_trunk_with_oracle->Reset();
  for (int iter = 1; iter <= num_trunks; ++iter) {
    ++iterable_trunk_with_oracle->num_iterations_;
    iterable_trunk_with_oracle->UpdateReachProbs();
    iterable_trunk_with_oracle->EvaluateLeaves();

    bool should_evaluate = std::find(eval_iters.begin(), eval_iters.end(),
                                     iter) != eval_iters.end();
    if (should_evaluate) {
      double expl = algorithms::ortools::TrunkExploitability(
          sf_lp, *iterable_trunk_with_oracle->AveragePolicy());
      std::cout << "# " << iter << "," << expl << std::endl;
    }

    AddExperiencesFromPublicStates(iterable_trunk_with_oracle->public_states());
    iterable_trunk_with_oracle->UpdateTrunk();
  }

  std::cout << "# </ref_expl>\n";
  SPIEL_CHECK_TRUE(replay->IsFilled() && replay->IsAtBeginning());
}

void ReplayFiller::AddRandomPbsSolution() {
  PublicStatesInGame* all_states = reuse->GetAllPublicStates();
  std::vector<algorithms::BanditVector>& bandits =
      reuse->GetFixableBanditsForAllPublicStates();
  const int num_states = all_states->public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);
  SPIEL_CHECK_TRUE(randomizer->rnd_gen);

  // 1. Pick a public state
  PublicState* state = nullptr;  // TODO: make sure we can also add terminals
  while (!state || state->IsTerminal()) {
    const int pick_public_state = public_state_dist(*randomizer->rnd_gen);
    state = &all_states->public_states[pick_public_state];
  }

  // 2. Assign randomized beliefs.
  randomizer->Randomize(bandits);
  UpdateBeliefs(*state, bandits);

  // 3. Build subgame and solve it.
  std::unique_ptr<Subgame> subgame = factory->MakeSubgame(*state, 1000);
  subgame->RunSimultaneousIterations(100);
  std::array<absl::Span<const double>, 2> root_values =
      subgame->RootChildrenCfValues();
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(state->values[pl].size(), root_values[pl].size());
    std::copy(root_values[pl].begin(), root_values[pl].end(),
              state->values[pl].begin());
  }

  // 4. Add solution to the experiences.
  auto context = factory->leaf_evaluator->CreateContext(*state);
  auto net_context = open_spiel::down_cast<NetContext*>(context.get());
  AddExperience(*state, net_context);
}

void ReplayFiller::AddRandomSparsePbsSolution() {
  PublicStatesInGame* all_states = reuse->GetAllPublicStates();
  std::vector<algorithms::BanditVector>& bandits =
      reuse->GetFixableBanditsForAllPublicStates();
  const int num_states = all_states->public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);
  SPIEL_CHECK_TRUE(randomizer->rnd_gen);
  SPIEL_CHECK_GT(sparse_particles, 0);

  // 1. Pick a "nice" public state.
  PublicState* state = nullptr;
  // TODO: check IsReachable() is correct wrt cfvs
  while (!state || !state->IsReachable()) {
    const int pick_public_state = public_state_dist(*randomizer->rnd_gen);
    state = &all_states->public_states.at(pick_public_state);
    // TODO: make sure we can also add terminals
    if (state->IsTerminal()) continue;

    // 2. Assign randomized beliefs.
    randomizer->Randomize(bandits);
    UpdateBeliefs(*state, bandits);
  }

  // 3. Pick the most probable particles.
  std::unique_ptr<ParticleSetPartition> particle_partition =
      MakeParticleSetPartition(*state, sparse_particles, sparse_epsilon,
          /*save_secondary=*/false, *randomizer->rnd_gen);

  // 5. Build subgame and solve it.
  std::unique_ptr<Subgame> subgame =
      factory->MakeSubgame(particle_partition->primary, 1000);
  subgame->RunSimultaneousIterations(100);
  PublicState& result = subgame->initial_state();

  // 6. Copy the solution to state.
  std::array<absl::Span<const double>, 2> root_values =
      subgame->RootChildrenCfValues();
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(result.values[pl].size(), root_values[pl].size());
    std::copy(root_values[pl].begin(), root_values[pl].end(),
              result.values[pl].begin());
  }

  // 7. Add solution to the experiences.
  auto context = factory->leaf_evaluator->CreateContext(result);
  auto net_context = open_spiel::down_cast<NetContext*>(context.get());
  AddExperience(result, net_context);
}

void ReplayFiller::CreateExperience(ReplayFillerPolicy fill_policy,
                                    int num_experiences) {
  SPIEL_CHECK_LE(num_experiences, replay->size());
  std::cout << "# ";
  for (int i = 0; i < num_experiences; ++i) {
    if (i % 10 == 0) std::cout << '.' << std::flush;
    switch (fill_policy) {
      case kTrunkRandom:     AddTrunkRandomPbsSolution();  break;
      case kPbsRandom:       AddRandomPbsSolution();       break;
      case kSparsePbsRandom: AddRandomSparsePbsSolution(); break;
      case kNothing:         break;
      default: SpielFatalError("Exhausted pattern match on ReplayFillerPolicy");
    }
  }
  std::cout << std::endl;
}

void ReplayFiller::FillReplay(ReplayFillerPolicy fill_policy) {
  replay->ResetHead();
  if (fill_policy == kTrunkDlcfr) return FillReplayWithTrunkDlCfrPbsSolutions();

  CreateExperience(fill_policy, replay->size());
  SPIEL_CHECK_TRUE(replay->IsFilled() && replay->IsAtBeginning());
}

}  // papers_with_code
}  // open_spiel
