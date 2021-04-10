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
  ++head_;
  if (head_ >= size()) {
    head_ = 0;
    ++overflow_cnt_;
  }
}

void ExperienceReplay::SampleBatch(BatchData* batch,
                                   std::mt19937& rnd_gen) const {
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
    batch->data[i].copy_(data[perm[i]]);
    batch->target[i].copy_(target[perm[i]]);
  }
}

ReplayFillerInit GetReplayInit(const std::string& s) {
  if (s == "trunk_dlcfr")  return kTrunkDlcfr;
  if (s == "trunk_random") return kTrunkRandom;
  if (s == "pbs_random")   return kPbsRandom;
  SpielFatalError("Exhausted pattern match: exp_init");
}

void ReplayFiller::AddExperience(const PublicState& leaf,
                                 const NetContext* net_context) {
  switch(arch) {
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

void ReplayFiller::FillReplayWithTrunkRandomPbsSolutions() {
  Subgame* fixable_trunk_with_oracle = reuse->GetFixableTrunkWithOracle();

  std::cout << "# Generating random trunks to fill experience replay.\n# ";
  int num_non_terminal_leaves = 0;
  for (auto& state: fixable_trunk_with_oracle->public_states()) {
    if (!state.IsTerminal() && state.IsLeaf()) num_non_terminal_leaves++;
  }
  SPIEL_CHECK_GT(num_non_terminal_leaves, 0);
  int num_trunks = replay->size() / num_non_terminal_leaves;

  for (int i = 0; i < num_trunks; ++i) {
    if (i % 10 == 0) std::cout << '.' << std::flush;

    fixable_trunk_with_oracle->Reset();
    // Randomize strategy in the trunk.
    randomizer->Randomize(fixable_trunk_with_oracle->bandits());
    // Compute the reach probs from the trunk.
    fixable_trunk_with_oracle->UpdateReachProbs();
    // Do not call bottom-up, just evaluate leaves.
    fixable_trunk_with_oracle->EvaluateLeaves();
    // Copy the leaves values to the experience replay.
    AddExperiencesFromPublicStates(fixable_trunk_with_oracle->public_states());
  }
  std::cout << std::endl;
}

// The network should imitate DL-CFR at each iteration
// when we use this generation method.
void ReplayFiller::FillReplayWithTrunkDlCfrPbsSolutions(
    const std::vector<int>& eval_iters) {
  Subgame* iterable_trunk_with_oracle = reuse->GetIterableTrunkWithOracle();
  SequenceFormLpSpecification* sf_lp = reuse->GetSfLp();

  std::cout << "# Computing reference expls for given trunk iterations.\n";
  int num_non_terminal_leaves = 0;
  for (auto& state: iterable_trunk_with_oracle->public_states()) {
    if (!state.IsTerminal() && state.IsLeaf()) num_non_terminal_leaves++;
  }
  SPIEL_CHECK_GT(num_non_terminal_leaves, 0);
  int num_trunks = replay->size() / num_non_terminal_leaves;

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
}

void ReplayFiller::FillReplayWithRandomPbsSolutions() {
  PublicStatesInGame* all_states = reuse->GetAllPublicStates();

  std::cout << "# Generating random PBS and finding their solutions ...\n# ";
  auto bandits = MakeBanditVectors(all_states->infostate_trees, "FixableStrategy");
  const int num_states = all_states->public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);
  SPIEL_CHECK_TRUE(randomizer->rnd_gen);

  int i = 0;
  while(i < replay->size()) {
    // 1. Pick a public state and compute according beliefs.
    const int pick_public_state = public_state_dist(*randomizer->rnd_gen);
    PublicState& state = all_states->public_states[pick_public_state];
    if (state.IsTerminal()) continue; // TODO: make sure we can also add terminals
    randomizer->Randomize(bandits);
    UpdateBeliefs(state, bandits);

    // 2. Build subgame and solve it.
    std::unique_ptr<Subgame> subgame =
        factory->MakeSubgame(state, 1000, reuse->pbs_oracle);
    subgame->RunSimultaneousIterations(100);
    std::array<absl::Span<const double>, 2> root_values =
        subgame->RootChildrenCfValues();
    for (int pl = 0; pl < 2; ++pl) {
      SPIEL_CHECK_EQ(state.values[pl].size(), root_values[pl].size());
      std::copy(root_values[pl].begin(), root_values[pl].end(),
                state.values[pl].begin());
    }

    // 3. Add solution to the experiences.
    auto context = factory->leaf_evaluator->CreateContext(state);
    auto net_context = open_spiel::down_cast<NetContext*>(context.get());
    AddExperience(state, net_context);
    if (i % 100 == 0) std::cout << '.' << std::flush;
    i++;
  }
  std::cout << std::endl;
  SPIEL_CHECK_TRUE(replay->IsFilled() && replay->IsAtBeginning());
}

}  // papers_with_code
}  // open_spiel
