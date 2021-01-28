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
    if (!dims->write_hand_features_positionally) {
      dims->hand_features_size = tables[pl].hand_tensor_size();
    }
  }
  if (dims->write_hand_features_positionally) {
    // Max per table, because we already encode player id as a feature.
    dims->hand_features_size = std::max(tables[0].num_hands(),
                                        tables[1].num_hands());
  }
  dims->max_particles = tables[0].num_hands() + tables[1].num_hands();
  num_leaves = fixable_trunk_with_oracle->public_leaves().size();
  num_non_terminal_leaves = 0;
  for (auto& leaf: fixable_trunk_with_oracle->public_leaves()) {
    if (!leaf.IsTerminal()) num_non_terminal_leaves++;
  }
}

void AddExperiencesFromTrunk(
    const std::vector<algorithms::dlcfr::LeafPublicState>& public_leaves,
    const std::vector<HandTable>& hand_tables,
    const ParticleDims& dims, ExperienceReplay* replay,
    std::mt19937& rnd_gen, bool shuffle_input, bool shuffle_output) {
  for (int i = 0; i < public_leaves.size(); ++i) {
    const dlcfr::LeafPublicState& leaf = public_leaves[i];
    if (leaf.IsTerminal()) continue;  // Add experiences only for non-terminals.
    ParticlesInContext data_point = replay->AddExperience(dims);
    WriteParticles(leaf, hand_tables, dims, &data_point,
                   &rnd_gen, shuffle_input, shuffle_output);
  }
}

// Rewrite all private hands into one-hot encoded positional hands.
// The size of the encoding should be supplied and should be equal
// to the maximum of the number of the private hands over the players.
void WritePositionalHand(const HandMapping& map, int infostate_id,
                         absl::Span<float_net> write_to) {
  int net_id = map.tree_to_net().at(infostate_id);
  std::fill(write_to.begin(), write_to.end(), 0.);
  write_to[net_id] = 1.;
}

void WriteParticles(const algorithms::dlcfr::LeafPublicState& state,
                    const std::vector<HandTable>& hand_tables,
                    const ParticleDims& dims, ParticlesInContext* point,
                    std::mt19937* rnd_gen, bool shuffle_input, bool shuffle_output) {
  point->Reset();

  // Make a random permutation if something should be shuffled.
  int num_particles = state.leaf_nodes[0].size() + state.leaf_nodes[1].size();
  std::vector<int> particle_placement(num_particles);
  if (shuffle_input || shuffle_output) {
    SPIEL_CHECK_TRUE(rnd_gen);
    std::iota(particle_placement.begin(), particle_placement.end(), 0);
    std::shuffle(particle_placement.begin(), particle_placement.end(),
                 *rnd_gen);
  }

  // Write inputs
  int i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParticleData particle = point->particle_at(
          dims, shuffle_input ? particle_placement[i] : i);
      Copy(state.public_tensor.Tensor(), particle.public_features());
      // Hand features.
      if(dims.write_hand_features_positionally) {
        WritePositionalHand(hand_tables[pl].bijections[state.public_id], j,
                            particle.hand_features());
      } else {
        const Observation& hand_observation =
            hand_tables[pl].hand_observation_at(state.public_id, j);
        Copy(hand_observation.Tensor(), particle.hand_features());
      }
      particle.player_features()[pl] = 1.;
      particle.range() = state.ranges[pl][j];
      i++;
    }
  }
  SPIEL_CHECK_EQ(i, num_particles);

  // Write outputs
  i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParticleData particle = point->particle_at(
          dims, shuffle_output ? particle_placement[i] : i);
      particle.value() = state.values[pl][j];
      i++;
    }
  }

  SPIEL_CHECK_EQ(i, num_particles);
  point->num_particles() = num_particles;
}

void CopyValuesNetToTree(ParticlesInContext data_point,
                         algorithms::dlcfr::LeafPublicState& state,
                         const std::vector<HandTable>& hand_tables,
                         const ParticleDims& dims) {
  int particle_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParticleData particle = data_point.particle_at(dims, particle_index);
      state.values[pl][j] = particle.value();
      particle_index++;
    }
  }
  SPIEL_CHECK_EQ(data_point.num_particles(), particle_index);
}

}  // namespace papers_with_code
}  // namespace open_spiel

