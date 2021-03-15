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

Trunk::Trunk(const std::string& game_name, int depth,
             std::string use_bandits_for_cfr) {
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
  hand_info = CreateHandInfo(*game, hand_observer,
                             fixable_trunk_with_oracle->public_leaves());

  num_leaves = fixable_trunk_with_oracle->public_leaves().size();
  num_non_terminal_leaves = 0;
  for (auto& leaf: fixable_trunk_with_oracle->public_leaves()) {
    if (!leaf.IsTerminal()) num_non_terminal_leaves++;
  }
}

std::unique_ptr<BasicDims> DeduceDims(const Trunk& trunk, NetArchitecture arch) {
  const dlcfr::LeafPublicState& some_leaf =
      trunk.fixable_trunk_with_oracle->public_leaves().at(0);
  const HandInfo& hand_info = *trunk.hand_info;

  std::unique_ptr<BasicDims> dims;
  switch (arch) {
    case NetArchitecture::kParticle:
      dims = std::make_unique<ParticleDims>();
      break;
    case NetArchitecture::kPositional:
      dims = std::make_unique<PositionalDims>();
      break;
  }

  // Fill basic dims.
  dims->public_features_size = some_leaf.public_tensor.Tensor().size();
  for (int pl = 0; pl < 2; ++pl) {
    dims->net_ranges_size[pl] = hand_info.tables[pl].private_hands.size();
    if (!dims->write_hand_features_positionally()) {
      dims->hand_features_size = hand_info.hand_tensor_size();
    }
  }
  if (dims->write_hand_features_positionally()) {
    // Max per table, because we already encode player id as a feature.
    dims->hand_features_size = std::max(
        hand_info.tables[0].private_hands.size(),
        hand_info.tables[1].private_hands.size());
  }

  // Fill custom dims.
  if (arch == NetArchitecture::kParticle) {
    auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
    particle_dims->max_parviews = hand_info.num_hands();
  }

  return dims;
}

// Rewrite all private hands into one-hot encoded positional hands.
// The size of the encoding should be supplied and should be equal
// to the maximum of the number of the private hands over the players.
void WritePositionalHand(int net_id, absl::Span<float_net> write_to) {
  std::fill(write_to.begin(), write_to.end(), 0.);
  write_to[net_id] = 1.;
}

void WriteParticleDataPoint(const algorithms::dlcfr::LeafPublicState& state,
                            const NetContext& net_context,
                            const ParticleDims& dims, ParticleDataPoint* point,
                            std::mt19937* rnd_gen, bool shuffle_input_output) {
  // Important !!
  point->Reset();

  // Find out how many parviews we will write.
  int num_parviews = state.leaf_nodes[0].size() + state.leaf_nodes[1].size();
  // Make a random permutation if something should be shuffled.
  std::vector<int> parview_placement(num_parviews);
  if (shuffle_input_output) {
    SPIEL_CHECK_TRUE(rnd_gen);
    std::iota(parview_placement.begin(), parview_placement.end(), 0);
    std::shuffle(parview_placement.begin(), parview_placement.end(),
                 *rnd_gen);
  }

  // Write inputs
  int i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParviewDataPoint parview = point->parview_at(shuffle_input_output
                                                    ? parview_placement[i] : i);
      // Hand features.
      if(dims.write_hand_features_positionally()) {
        WritePositionalHand(net_context.net_index(pl, j),
                            parview.hand_features());
      } else {
        const Observation& hand_observation = net_context.hand_at(pl, j);
        Copy(hand_observation.Tensor(), parview.hand_features());
      }
      parview.player_features()[pl] = 1.;
      parview.range() = state.ranges[pl][j];
      i++;
    }
  }
  SPIEL_CHECK_EQ(i, num_parviews);
  point->num_parviews() = num_parviews;
  Copy(state.public_tensor.Tensor(), point->public_features());

  // Write outputs
  i = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParviewDataPoint parview = point->parview_at(shuffle_input_output
                                                   ? parview_placement[i] : i);
      parview.value() = state.values[pl][j];
      i++;
    }
  }

  SPIEL_CHECK_EQ(i, num_parviews);
}

void CopyValuesFromNetToTree(ParticleDataPoint data_point,
                             algorithms::dlcfr::LeafPublicState& state,
                             const ParticleDims& dims) {
  int parview_index = 0;
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      ParviewDataPoint parview = data_point.parview_at(parview_index);
      state.values[pl][j] = parview.value();
      parview_index++;
    }
  }
  SPIEL_CHECK_EQ(data_point.num_parviews(), parview_index);
}

void WritePositionalDataPoint(const algorithms::dlcfr::LeafPublicState& state,
                              const NetContext& net_context,
                              const PositionalDims& dims, PositionalData* point) {
  // Important !!
  point->Reset();

  // Write inputs
  Copy(state.public_tensor.Tensor(), point->public_features());
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      point->range_at(pl, net_context.net_index(pl, j)) = state.ranges[pl][j];
    }
  }
  // Write outputs
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      point->value_at(pl, net_context.net_index(pl, j)) = state.values[pl][j];
    }
  }
}

void CopyValuesNetToTree(PositionalData* point,
                         algorithms::dlcfr::LeafPublicState& state,
                         const NetContext& net_context) {
  for (int pl = 0; pl < 2; ++pl) {
    for (int j = 0; j < state.leaf_nodes[pl].size(); j++) {
      state.values[pl][j] = point->value_at(pl, net_context.net_index(pl, j));
    }
  }
}

//// Copy non-contiguous vectors using a permutation map.
//// This also converts float <-> double as needed.
//template<typename From, typename To>
//void PlacementCopy(const std::vector<From>& from, absl::Span<To> to,
//                   const std::map<size_t, size_t>& from_to) {
//  for (const auto&[f, t] : from_to) {
//    to[t] = from[f];
//  }
//}
//template<typename From, typename To>
//void PlacementCopy(absl::Span<From> from, std::vector<To>& to,
//                   const std::map<size_t, size_t>& from_to) {
//  for (const auto&[f, t] : from_to) {
//    to[t] = from[f];
//  }
//}
//
//void CopyRangesTreeToNet(const dlcfr::LeafPublicState& leaf,
//                         PositionalData data_point,
//                         const std::vector<dlcfr::RangeTable>& tables) {
//  for (int pl = 0; pl < 2; ++pl) {
//    PlacementCopy<float_tree, float_net>(
//        /*tree=*/ leaf.ranges[pl],
//        /*net=*/  data_point.net_ranges[pl],
//                  tables[pl].bijections[leaf.public_id].tree_to_net());
//  }
//}
//
//void CopyValuesTreeToNet(const dlcfr::LeafPublicState& leaf,
//                         PositionalData data_point,
//                         const std::vector<dlcfr::RangeTable>& tables) {
//  for (int pl = 0; pl < 2; ++pl) {
//    PlacementCopy<float_tree, float_net>(
//        /*tree=*/ leaf.values[pl],
//        /*net=*/  data_point.net_values[pl],
//                  tables[pl].bijections[leaf.public_id].tree_to_net());
//  }
//}
//
//void CopyValuesFromNetToTree(PositionalData data_point,
//                         dlcfr::LeafPublicState& leaf,
//                         const std::vector<dlcfr::RangeTable>& tables) {
//  for (int pl = 0; pl < 2; ++pl) {
//    PlacementCopy<float_net, float_tree>(
//        /*net=*/  data_point.net_values[pl],
//        /*tree=*/ leaf.values[pl],
//                  tables[pl].bijections[leaf.public_id].net_to_tree());
//  }
//}


void PrintTrunkStrategies(algorithms::dlcfr::DepthLimitedCFR* trunk_with_net) {
  auto& bandits = trunk_with_net->bandits();
  auto& trees = trunk_with_net->trees();
  for (int pl = 0; pl < 2; ++pl) {
    std::cout << "# Trunk eq strategies -- player " << pl << std::endl;
    for(DecisionId id : bandits[pl].range()) {
      std::cout << "# " << trees[pl]->decision_infostate(id)->ToString()
                << " " << bandits[pl][id]->AverageStrategy() << std::endl;
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

