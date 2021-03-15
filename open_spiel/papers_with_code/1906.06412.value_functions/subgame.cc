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

#include "open_spiel/papers_with_code/1906.06412.value_functions/subgame.h"

namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

void Subgame::CopySolution() {
  // The parviews have a following invariant: they are ordered exactly the same
  // as root infostate nodes within the DL-CFR infostate trees.
  // Therefore the DepthLimitedCFR::RootChildrenCfValues() buffer can be written
  // directly to the data point targets.

  std::array<absl::Span<const double>, 2> cf_values =
      solver->RootChildrenCfValues();
  int num_root_infostates = cf_values[0].size() + cf_values[1].size();
  SPIEL_DCHECK_EQ(num_root_infostates, solution.num_parviews());
  int i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < cf_values[pl].size(); ++j) {
      solution.parview_at(i).value() = cf_values[pl][i];
      ++i;
    }
  }
  SPIEL_CHECK_EQ(i, num_root_infostates);
}

std::unique_ptr<Subgame> SubgameFactory::MakeSubgame(
    const ParticleSet& set, ParticleDataPoint write_solution_to) {

  SPIEL_DCHECK(CheckParticleSetConsistency(*game, public_observer,
                                           hand_observer, set));

  std::vector<std::unique_ptr<State>> start_states;
  std::vector<const State*> start_states_ptr;
  std::vector<double> chance_reach_probs;
  for (const Particle& particle : set.particles) {
    start_states.push_back(particle.MakeState(*game));
    start_states_ptr.push_back(start_states.back().get());
    chance_reach_probs.push_back(particle.chance_reach);
  }
  State* some_state = start_states.front().get();

  std::vector<std::shared_ptr<InfostateTree>> trees;
  for (int pl = 0; pl < 2; ++pl) {
    trees.push_back(MakeInfostateTree(
        start_states_ptr, chance_reach_probs,
        infostate_observer, pl, max_move_ahead_limit
    ));
  }
  SPIEL_DCHECK(CheckParticleSetConsistency(*game, infostate_observer,
                                           {trees[0]->root().children(),
                                            trees[1]->root().children()}, set));

  std::array<std::vector<double>, 2> beliefs = set.ComputeBeliefs();

  // Write public features
  ContiguousAllocator public_allocator(write_solution_to.public_features());
  public_observer->WriteTensor(*some_state, kDefaultPlayerId,
                               &public_allocator);
  // Write parviews
  int j = 0;
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(trees[pl]->root().num_children(), set.partition[pl].size());
    for (int i = 0; i < set.partition[pl].size(); ++i) {
      int k = set.partition[pl][i].front();
      State* parview_repr_state = start_states[k].get();
      ParviewDataPoint parview = write_solution_to.parview_at(j);
      ContiguousAllocator parview_allocator(parview.hand_features());
      hand_observer->WriteTensor(*parview_repr_state, pl, &parview_allocator);
      parview.player_features()[pl] = 1.;
      parview.player_features()[1-pl] = 0.;
      parview.range() = beliefs[pl][i];
      ++j;
    }
  }
  write_solution_to.num_parviews() = j;
  SPIEL_CHECK_EQ(j, set.partition[0].size() + set.partition[1].size());

  auto solver = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trees, leaf_evaluator, terminal_evaluator,
      public_observer, MakeBanditVectors(trees, use_bandits_for_cfr));
  solver->SetPlayerRanges(beliefs);
  return std::make_unique<Subgame>(std::move(solver), write_solution_to);
}


std::unique_ptr<ParticleSet> SubgameFactory::SampleParticleSet(
    const Subgame& subgame, std::mt19937& rnd_gen) {
  return std::unique_ptr<ParticleSet>();  // TODO.
}


}  // papers_with_code
}  // open_spiel
