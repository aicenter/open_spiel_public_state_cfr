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


#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"

#include "open_spiel/utils/format_observation.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

std::unique_ptr<Trunk> MakeTrunk(const std::string& game_name,
                                 int trunk_depth, std::string use_bandits_for_cfr) {
  return std::make_unique<Trunk>(game_name, trunk_depth, use_bandits_for_cfr);
}

Trunk::Trunk(const std::string& game_name, int depth, std::string use_bandits_for_cfr) {
  // 1. Prepare the game, observers and depth-limited (trunk) trees.
  game = LoadGame(game_name);
  trunk_depth = depth;
  infostate_observer = game->MakeObserver(kInfoStateObsType, {});
  public_observer = game->MakeObserver(kPublicStateObsType, {});
  hand_observer = game->MakeObserver(kHandHistoryObsType, {});
  trunk_trees = {MakeInfostateTree(*game, 0, trunk_depth),
                 MakeInfostateTree(*game, 1, trunk_depth)};

  // 2. Create value oracle for the trunk.
  terminal_evaluator = dlcfr::MakeTerminalEvaluator();
  oracle_evaluator = std::make_shared<dlcfr::CFREvaluator>(
      game, /*full_subgame_depth=*/100, /*no_leaf_evaluator=*/nullptr,
      terminal_evaluator, public_observer, infostate_observer);
  oracle_evaluator->bandit_name = use_bandits_for_cfr;
  fixable_trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator,
      public_observer,
      MakeBanditVectors(trunk_trees, "FixableStrategy"));
  iterable_trunk_with_oracle = std::make_unique<dlcfr::DepthLimitedCFR>(
      game, trunk_trees, oracle_evaluator, terminal_evaluator,
      public_observer,
      MakeBanditVectors(trunk_trees, use_bandits_for_cfr));

  // 3. Make a Batch of data that encompasses all leaf public states.
  tables = CreateHandTables(*game, hand_observer,
                            fixable_trunk_with_oracle->public_leaves());
  const dlcfr::LeafPublicState& some_leaf =
      fixable_trunk_with_oracle->public_leaves().at(0);

  dims = std::make_unique<ParticleDims>();
  dims->public_features_size = some_leaf.public_tensor.Tensor().size();
  for (int pl = 0; pl < 2; ++pl) {
    dims->net_ranges_size[pl] = tables[pl].num_hands();
  }
  num_leaves = fixable_trunk_with_oracle->public_leaves().size();
  num_non_terminal_leaves = 0;
  for (auto& leaf: fixable_trunk_with_oracle->public_leaves()) {
    if (!leaf.IsTerminal()) num_non_terminal_leaves++;
  }
}

void AddExperiencesFromTrunk(
    const std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves,
    const std::vector<std::unique_ptr<HandContext>>& hand_contexts,
    const ParticleDims& dims, ExperienceReplay* replay) {
  for (int i = 0; i < public_leaves.size(); ++i) {
    const dlcfr::LeafPublicState& leaf = public_leaves[i];
    if (leaf.IsTerminal()) continue;  // Add experiences only for non-terminals.
    ParticlesInContext data_point = replay->AddExperience(dims);
    SPIEL_DCHECK_TRUE(data_point.is_valid_view());
    WriteParticles(leaf, *hand_contexts[i], dims, &data_point);
  }
}

void WriteParticles(const algorithms::dlcfr::LeafPublicState& state,
                    const HandContext& hand_context, const ParticleDims& dims,
                    ParticlesInContext* point) {
  point->Reset();
  int particle_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParticleData particle = point->particle_at(dims, particle_index);

      Copy(state.public_tensor.Tensor(), particle.public_features());
      Copy(hand_context.hand_features[pl][j].Tensor(), particle.hand_features());
      particle.player_features()[pl] = 1.;
      particle.range() = state.ranges[pl][j];
      particle.value() = state.values[pl][j];
      particle_index++;
    }
  }
  point->num_particles() = particle_index;
}

HandContext::HandContext(const algorithms::dlcfr::LeafPublicState& leaf_state,
                         Observation& hand_observation) {
  for (int pl = 0; pl < 2; ++pl) {
    hand_features[pl].reserve(leaf_state.leaf_nodes[pl].size());

    for (const InfostateNode* leaf_node : leaf_state.leaf_nodes[pl]) {
      SPIEL_CHECK_FALSE(leaf_node->corresponding_states().empty());
      const auto& some_state = leaf_node->corresponding_states()[0];
      hand_observation.SetFrom(*some_state, pl);
      hand_features[pl].push_back(hand_observation);
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

