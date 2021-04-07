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

ParticleDataPoint ExperienceReplay::AddExperience(const ParticleDims& dims) {
  ParticleDataPoint point = point_at(head_, dims);
  AdvanceHead();
  return point;
}

PositionalData ExperienceReplay::AddExperience(const PositionalDims& dims) {
  PositionalData point = point_at(head_, dims);
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

  std::vector<int> perm(n);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rnd_gen);

  for (int i = 0; i < k; ++i) {
    batch->data[i].copy_(data[perm[i]]);
    batch->target[i].copy_(target[perm[i]]);
  }
}

ExpReplayInitialization GetExpReplayInitialization(const std::string& s) {
  if (s == "trunk_dlcfr")  return kInitTrunkDlcfr;
  if (s == "trunk_random") return kInitTrunkRandom;
  SpielFatalError("Exhausted pattern match: exp_init");
}

void AddExperience(
    const algorithms::dlcfr::PublicState& leaf,
    NetContext* net_context,
    const BasicDims& dims,
    NetArchitecture arch,
    ExperienceReplay* replay,
    std::shared_ptr<Observer> hand_observer,
    std::mt19937& rnd_gen,
    bool shuffle_input_output
) {
  switch(arch) {
    case NetArchitecture::kParticle: {
      auto particle_dims = open_spiel::down_cast<const ParticleDims&>(dims);
      ParticleDataPoint data_point = replay->AddExperience(particle_dims);
      WriteParticleDataPoint(leaf, particle_dims, &data_point,
                             hand_observer,
                             &rnd_gen, shuffle_input_output);
      break;
    }
    case NetArchitecture::kPositional: {
      auto pos_dims = open_spiel::down_cast<const PositionalDims&>(dims);
      PositionalData data_point = replay->AddExperience(pos_dims);
      SPIEL_CHECK_TRUE(net_context);
      WritePositionalDataPoint(leaf, *net_context, pos_dims, &data_point);
      break;
    }
  }
}


void AddExperiencesFromTrunk(
    const std::vector<algorithms::dlcfr::PublicState>& states,
    const std::vector<NetContext*>& net_contexts,
    const BasicDims& dims,
    NetArchitecture arch,
    ExperienceReplay* replay,
    std::shared_ptr<Observer> hand_observer,
    std::mt19937& rnd_gen,
    bool shuffle_input_output
) {
  for (int i = 0; i < states.size(); ++i) {
    const algorithms::dlcfr::PublicState& state = states[i];
    // Add experiences only for the non-terminal leaves.
    if (!state.IsLeaf() || state.IsTerminal()) continue;
    AddExperience(state, net_contexts[i], dims, arch, replay,
                  hand_observer, rnd_gen, shuffle_input_output);
  }
}

void InitTrunkRandomBeliefs(
    Trunk* trunk,
    const std::vector<NetContext*>& contexts,
    const BasicDims& dims,
    NetArchitecture arch,
    ExperienceReplay* replay,
    double prob_pure_strat,
    double prob_fully_mixed,
    std::mt19937& rnd_gen,
    bool shuffle_input_output
) {
  trunk->fixable_trunk_with_oracle->Reset();

  // Randomize strategy in the trunk.
  RandomizeStrategy(trunk->fixable_trunk_with_oracle->bandits(),
                    prob_pure_strat, prob_fully_mixed, rnd_gen);
  // Compute the reach probs from the trunk.
  trunk->fixable_trunk_with_oracle->UpdateReachProbs();
  // Do not call bottom-up, just evaluate leaves.
  trunk->fixable_trunk_with_oracle->EvaluateLeaves();
  // Copy the leaves values to the experience replay.
  AddExperiencesFromTrunk(
      trunk->fixable_trunk_with_oracle->public_states(),
      contexts, dims, arch, replay,
      trunk->hand_observer, rnd_gen, shuffle_input_output);

//  if (verbose) {
//    for (int i = 0; i < batch->batch_size; ++i) {
//      std::cout << "# Public state " << i << std::endl;
//      std::cout << "#   Inputs:  " << batch->data_at(i) << std::endl;
//      std::cout << "#   Outputs: " << batch->targets_at(i) << std::endl;
//    }
//    std::cout << "\n# ";
//  }
}

// The network should imitate DL-CFR at each iteration
// when we use this generation method.
void InitTrunkDlCfrIterations(
    Trunk* trunk,
    const std::vector<NetContext*>& contexts,
    const BasicDims& dims,
    NetArchitecture arch,
    ExperienceReplay* replay,
    int trunk_iters,
    std::function<void(/*trunk_iter=*/int)> monitor_fn,
    std::mt19937& rnd_gen,
    bool shuffle_input_output
) {
  trunk->iterable_trunk_with_oracle->Reset();
  for (int iter = 1; iter <= trunk_iters; ++iter) {
    ++trunk->iterable_trunk_with_oracle->num_iterations_;
    trunk->iterable_trunk_with_oracle->UpdateReachProbs();
    trunk->iterable_trunk_with_oracle->EvaluateLeaves();

    AddExperiencesFromTrunk(trunk->iterable_trunk_with_oracle->public_states(),
                            contexts, dims, arch, replay, trunk->hand_observer,
                            rnd_gen, shuffle_input_output);
    monitor_fn(iter);

    trunk->iterable_trunk_with_oracle->UpdateTrunk();
  }
}

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

void InitSubgamesRandomBeliefs(
    SubgameFactory* factory,
    const BasicDims& dims,
    NetArchitecture arch,
    ExperienceReplay* replay,
    std::mt19937& rnd_gen,
    double prob_pure_strat,
    double prob_fully_mixed,
    bool shuffle_input_output
) {
  auto all = algorithms::dlcfr::MakeAllPublicStates(*factory->game);
  auto bandits = MakeBanditVectors(all->infostate_trees, "FixableStrategy");
  const int num_trunks = replay->size();
  const int num_states = all->public_states.size();
  auto public_state_dist = std::uniform_int_distribution<>(0, num_states - 1);

  for (int i = 0; i < num_trunks; ++i) {
    // 1. Pick a public state and compute according beliefs.
    RandomizeStrategy(bandits, prob_pure_strat, prob_fully_mixed, rnd_gen);
    const int public_state_idx = public_state_dist(rnd_gen);
    PublicState& state = all->public_states[public_state_idx];
    UpdateBeliefs(state, bandits);

    // 2. Build subgame and solve it.
    std::unique_ptr<Subgame> subgame = factory->MakeSubgame(state);
    subgame->RunSimultaneousIterations(100);

    // 3. Add solution to the experiences.
    auto context = factory->leaf_evaluator->CreateContext(state);
    auto net_context = open_spiel::down_cast<NetContext*>(context.get());
    AddExperience(state, net_context, dims, arch, replay,
                  factory->hand_observer, rnd_gen, shuffle_input_output);
  }
}


}  // papers_with_code
}  // open_spiel
