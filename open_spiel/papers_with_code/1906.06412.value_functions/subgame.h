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

#ifndef OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_
#define OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "open_spiel/algorithms/bandits_policy.h"
#include "open_spiel/algorithms/infostate_cfr.h"
#include "open_spiel/algorithms/infostate_tree.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/data_structures.h"

// Depth-limited CFR is conceptually similar to what `infostate_cfr.h` does.
// Additionally it saves structure of public states in the root and leaves
// of the depth-limited infostate tree. It collects these into a set of
// `PublicState`s. DL-CFR can be instantiated with custom state evaluators,
// which may need to save different structures associated to these states.
//
// Therefore the state evaluator provides these two methods:
//
// - CreateContext: takes a public state and return any special context
//   representation needed for this state evaluator. These will be saved by
//   DL-CFR to provide them later as necessary.
// - EvaluatePublicState: receives a pointer to the public state along with
//   a context. In the state the evaluator can find the current beliefs, and it
//   should update the counterfactual values to continue the DL-CFR iterations.
//
// DL-CFR can be queried for `Policy` only in the depth-limited constructed
// parts of the tree.

namespace open_spiel {
namespace papers_with_code {

// -- Public state -------------------------------------------------------------

enum PublicStateType {
  // Initial public state of the depth-limited lookahead tree. There might be
  // multiple initial states (i.e. the lookahead tree is more precisely a forest
  // of trees).
  kInitialPublicState,
  // Leaf public state of the depth-limited lookahead tree. All such public
  // states are at the end of the lookahead. Some of them might be terminal,
  // i.e. they contain only `open_spiel::State`s that are terminal.
  kLeafPublicState,
};

struct PublicState {
  // An identification of the public state: a tensor of perfect recall
  // public observation.
  const Observation public_tensor;
  // Public state type. Note that only the pair of (public_tensor, state_type)
  // uniquely identifies public_id: this is for technical reasons so that we can
  // build 1-step lookaheads. There would be a single public state that is both
  // initial and leaf. We need to distinguish between the two, so we save them
  // redundantly as two different public states. The distinction is needed
  // because while the public observations are the same, the infostate node
  // pointers are not, and so are not all the other members derived from the
  // infostate nodes (beliefs and values).
  const PublicStateType state_type;
  // Position in the vector of Subgame::public_states()
  const size_t public_id;
  // TODO: careful about using this! maybe should be removed or properly handled
  //       this is just temporary field
  /*const*/ std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  // For each player, store a pointer to the infostate nodes for this public
  // state, within the depth-limited infostate tree. If needed, you can get
  // access to underlying perfect-information `State`s
  // via `InfostateNode::corresponding_states()`.
  // These are assigned once, and no subsequent change should be made.
  // The top nodes are saved for initial states and bottom nodes are saved for
  // leaf states. Both are listed for the special case if the public tree is
  // a singleton.
  /*const*/ std::array<std::vector<const algorithms::InfostateNode*>, 2> nodes;
  // Store a map between infostate nodes and positions in the vectors
  // of beliefs and values. The map is shared across the players.
  /*const*/ std::map<const algorithms::InfostateNode*, int> nodes_positions;
  // Store the move number associated with all the states that belong to this
  // public state.
  /*const*/ int move_number = -1;
  // For each player, store beliefs for the top-most infostate nodes, which
  // correspond to bottom_nodes of the DL infostate tree.
  std::array<std::vector<double>, 2> beliefs;
  // For each player, store current counterfactual values for the top-most
  // infostate nodes, which correspond to bottom_nodes of the DL infostate tree.
  std::array<std::vector<double>, 2> values;
  // For each player, store average counterfactual values for the same
  // infostates as the current cf. values.
  std::array<std::vector<double>, 2> average_values;

  PublicState(const Observation& public_observation,
              const PublicStateType state_type, const size_t public_id);
  // Check if the public state if initial, i.e. the depth-limited subgame
  // is rooted in this public state.
  bool IsInitial() const { return state_type == kInitialPublicState; }
  // Check if the public state if a leaf within the depth-limited subgame.
  bool IsLeaf() const { return state_type == kLeafPublicState; }
  // Check if the public state is terminal, i.e. it contains only states
  // that satisfy `State::IsTerminal()`.
  bool IsTerminal() const;
  // Compute the reach probability of this public state.
  double ReachProbability() const;
  // Check the state has non-zero reach probability.
  bool IsReachable() const { return ReachProbability() > 0; }
  // Check the state has non-zero beliefs for given player.
  bool IsReachableByPlayer(int player) const;
  // Check the state has non-zero beliefs for some player.
  bool IsReachableBySomePlayer() const {
    return IsReachableByPlayer(0) || IsReachableByPlayer(1);
  };
  // Compute value of this state based on current policy (for the given player).
  double CurrentValue(int player = 0) const;
  // Compute value of this state based on average policy (for the given player).
  double AverageValue(int player = 0) const;
  // Check if the public state is zero-sum.
  bool IsZeroSum() const {
    return fabs(CurrentValue(0) + CurrentValue(1)) < 1e-10;
  }
  // Set new beliefs and check they have consistent sizes.
  void SetBeliefs(const std::array<std::vector<double>, 2>& new_beliefs);
  // Return a map of infostate string: average cf. values.
  std::unordered_map<std::string, double> InfostateAvgValues(
      Player player) const;
  // Return the underlying game for this public state;
  std::shared_ptr<const Game> game() const;
};

void DebugPrintPublicFeatures(const std::vector<PublicState>& states);
void PrintPublicStatesStats(const std::vector<PublicState>& public_leaves);
bool DoStatesProduceEqualPublicObservations(const Game& game,
                                            std::shared_ptr<Observer> public_observer,
                                            const algorithms::InfostateNode& node,
                                            const Observation& expected_observation);

// -- Many public states -------------------------------------------------------

// TODO: merge with Subgame and introduce public tree structure.
//  This struct is pretty much the same, but subgame stores only
//  the initial / leaf public states.
// Store all public states in the game. This requires building the game tree
// for the entire game. All the public states are marked as initial (even
// the terminal ones).
struct PublicStatesInGame {
  std::vector<std::shared_ptr<algorithms::InfostateTree>> infostate_trees;
  std::vector<PublicState> public_states;
  PublicState* GetPublicState(const Observation& public_observation);
};
std::unique_ptr<PublicStatesInGame> MakeAllPublicStates(const Game& game);

// -- Subgame ------------------------------------------------------------------

// Depth-limited CFR requires storage of perfect-information states both
// in the roots and in the leaves of the infostate trees.
constexpr int kDlCfrInfostateTreeStorage = algorithms::kStoreStatesInRoots
                                         | algorithms::kStoreStatesInLeaves;

// Subgame stores initial and leaf public states, associated
// with depth-limited infostate trees.
struct Subgame {
  std::shared_ptr<const Game> game;
  std::shared_ptr<Observer> public_observer;
  std::vector<std::shared_ptr<algorithms::InfostateTree>> trees;
  std::vector<PublicState> public_states;

  Subgame(std::shared_ptr<const Game> game, int max_moves);
  Subgame(std::shared_ptr<const Game> game,
          std::shared_ptr<Observer> public_observer,
          std::vector<std::shared_ptr<algorithms::InfostateTree>> trees);

  PublicState& initial_state() {
    SPIEL_CHECK_FALSE(public_states.empty());
    SPIEL_CHECK_TRUE(public_states[0].IsInitial());
    return public_states[0];
  }

  PublicState* PickRandomLeaf(std::mt19937& engine);
 private:
  void MakePublicStates();
  void MakeBeliefsAndValues();
  PublicState* GetPublicState(const Observation& public_observation,
                              PublicStateType state_type);
  PublicState* GetPublicState(Observation& public_observation,
                              PublicStateType state_type,
                              algorithms::InfostateNode* node);
};

std::unique_ptr<Subgame> MakeSubgame(
    const PublicState& state,
    std::shared_ptr<const Game> game = nullptr,
    std::shared_ptr<Observer> public_observer = nullptr,
    int custom_move_ahead_limit = algorithms::kNoMoveAheadLimit);


}  // namespace papers_with_code
}  // namespace open_spiel

#endif  // OPEN_SPIEL_PAPERS_WITH_CODE_VALUE_FUNCTIONS_SUBGAME_
