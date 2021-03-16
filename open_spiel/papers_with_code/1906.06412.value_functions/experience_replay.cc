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

void GenerateDataRandomRanges(
    Trunk* trunk, const std::vector<NetContext*>& contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    double prob_pure_strat, double prob_fully_mixed,
    std::mt19937& rnd_gen, bool shuffle_input_output) {
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
      contexts, dims, arch, replay, rnd_gen, shuffle_input_output);

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
void GenerateDataDLCfrIterations(
    Trunk* trunk, const std::vector<NetContext*>& contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    int trunk_iters,
    std::function<void(/*trunk_iter=*/int)> monitor_fn,
    std::mt19937& rnd_gen, bool shuffle_input_output) {
  trunk->iterable_trunk_with_oracle->Reset();
  for (int iter = 1; iter <= trunk_iters; ++iter) {
    ++trunk->iterable_trunk_with_oracle->num_iterations_;
    trunk->iterable_trunk_with_oracle->UpdateReachProbs();
    trunk->iterable_trunk_with_oracle->EvaluateLeaves();

    AddExperiencesFromTrunk(trunk->iterable_trunk_with_oracle->public_states(),
                            contexts, dims, arch, replay,
                            rnd_gen, shuffle_input_output);
    monitor_fn(iter);

    trunk->iterable_trunk_with_oracle->UpdateTrunk();
  }
}

ExpReplayInitPolicy GetInitPolicy(const std::string& s) {
  if (s == "dl_cfr") return kGenerateDlcfrIterations;
  if (s == "random") return kGenerateRandomRangesAndSubgameValues;
  SpielFatalError("Exhausted pattern match: data_generation");
}

void AddExperiencesFromTrunk(
    const std::vector<algorithms::dlcfr::PublicState>& public_leaves,
    const std::vector<NetContext*>& net_contexts,
    const BasicDims& dims, NetArchitecture arch, ExperienceReplay* replay,
    std::mt19937& rnd_gen, bool shuffle_input_output) {
  for (int i = 0; i < public_leaves.size(); ++i) {
    const algorithms::dlcfr::PublicState& leaf = public_leaves[i];
    if (leaf.IsTerminal()) continue;  // Add experiences only for non-terminals.
    const NetContext& net_context = *net_contexts[i];

    switch(arch) {
      case NetArchitecture::kParticle: {
        auto particle_dims = open_spiel::down_cast<const ParticleDims&>(dims);
        ParticleDataPoint data_point = replay->AddExperience(particle_dims);
        WriteParticleDataPoint(leaf, net_context, particle_dims, &data_point,
                               &rnd_gen, shuffle_input_output);
        break;
      }
      case NetArchitecture::kPositional: {
        auto pos_dims = open_spiel::down_cast<const PositionalDims&>(dims);
        PositionalData data_point = replay->AddExperience(pos_dims);
        WritePositionalDataPoint(leaf, net_context, pos_dims, &data_point);
        break;
      }
    }

  }
}


}  // papers_with_code
}  // open_spiel
