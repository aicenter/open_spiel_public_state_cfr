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


#include <utility>
#include "bot.h"

namespace open_spiel {
namespace papers_with_code {

SherlockBot::SherlockBot(std::shared_ptr<SubgameFactory> subgame_factory,
                         std::shared_ptr<SolverFactory> solver_factory,
                         Player player_id, int seed)
    : subgame_factory_(std::move(subgame_factory)),
      solver_factory_(std::move(solver_factory)),
      player_id_(player_id),
      rnd_gen_(seed) {
  subgame_ = subgame_factory_->MakeTrunk(1);
  first_step_ = true;
}

SherlockBot::SherlockBot(const SherlockBot& bot)
    : subgame_factory_(bot.subgame_factory_),
      solver_factory_(bot.solver_factory_),
      player_id_(bot.player_id_),
      rnd_gen_(bot.rnd_gen_),
      subgame_(std::make_shared<Subgame>(*bot.subgame_)),  // Copy the subgame.
      past_policy_(bot.past_policy_),
      first_step_(bot.first_step_) {
  // Make sure we really made a copy (different memory pointers).
  SPIEL_CHECK_NE(subgame_.get(), bot.subgame_.get());
}

Action SherlockBot::Step(const State& state) {
  return StepWithPolicy(state).second;
}

void SherlockBot::SetSeed(int seed) {
  rnd_gen_.seed(seed);
}

void SherlockBot::Restart() {
  subgame_ = subgame_factory_->MakeTrunk(1);
  first_step_ = true;
  past_policy_.PolicyTable().clear();
}

std::pair<ActionsAndProbs, Action> SherlockBot::StepWithPolicy(const State& state) {
  SPIEL_CHECK_TRUE(subgame_);

  // Infostate observations.
  Observation infostate_observation
      (*subgame_factory_->game, subgame_factory_->infostate_observer);
  infostate_observation.SetFrom(state, player_id_);
  const std::string infostate =
      subgame_factory_->infostate_observer->StringFrom(state, player_id_);

  // Public state observations.
  Observation public_observation
      (*subgame_factory_->game, subgame_factory_->public_observer);
  public_observation.SetFrom(state, kDefaultPlayerId);

  // We should always be able to localize current public state based
  // on previous bot steps.
  PublicState* public_state = nullptr;
  for (PublicState& maybe_current : subgame_->public_states) {
    if (maybe_current.public_tensor == public_observation) {
      public_state = &maybe_current;
      break;
    }
  }
  SPIEL_CHECK_TRUE(public_state);
  SPIEL_CHECK_TRUE( // In the first step, the public state comes from the trunk,
                    // and is the initial public state.
                    (first_step_ && public_state->IsInitial())
                    // All subsequent steps are leaves from lookahead subgames.
                    || public_state->IsLeaf());

  // Select particles to keep from the previous step along with their beliefs.
  // Currently, this works only for one-step lookahead trees.
  std::unique_ptr<ParticleSet> set =
      GetParticles(public_state, infostate_observation, public_observation);
  SPIEL_CHECK_FALSE(set->particles.empty());

  // We will do the gadget if we are resolving
  if (!first_step_ && solver_factory_->safe_resolving) {
    subgame_ = subgame_factory_->MakeSubgameSafeResolving(
        *set,  // Includes reach probs.
        player_id_,
        /*opponent_CFVs=*/public_state->InfostateAvgValues(1 - player_id_));
  } else {
    subgame_ = subgame_factory_->MakeSubgame(*set);
  }

  // We used up this variable in previous step, so set it to false now.
  first_step_ = false;

  std::unique_ptr<SubgameSolver> solver = solver_factory_->MakeSolver(subgame_);

  solver->RunSimultaneousIterations(solver_factory_->cfr_iterations);
  solver->SetAverageBeliefsInLeaves();
  if (state.IsPlayerActing(player_id_)) {
    auto policy = solver->AveragePolicy();
    ActionsAndProbs actions_and_probs = policy->GetStatePolicy(infostate);
    SPIEL_CHECK_FALSE(actions_and_probs.empty());
    for (int pl = 0; pl < 2; ++pl) StorePastPolicy(subgame_->trees[pl], *policy);

    double p = std::uniform_real_distribution<>(0., 1.)(rnd_gen_);
    std::pair<Action, double> outcome = SampleAction(actions_and_probs, p);
    return {actions_and_probs, outcome.first};
  } else {
    // We're not acting: return empty actions and probs and invalid action.
    return {{}, kInvalidAction};
  }
}

std::unique_ptr<Bot> SherlockBot::Clone() const {
  return std::make_unique<SherlockBot>(*this);
}

void SherlockBot::StorePastPolicy(
    const std::shared_ptr<algorithms::InfostateTree> tree,
    const Policy& policy) {
  for(algorithms::InfostateNode* node : tree->AllDecisionInfostates()) {
    past_policy_.SetStatePolicy(node->infostate_string(),
                                policy.GetStatePolicy(node->infostate_string()));
  }
}

std::unique_ptr<ParticleSet> SherlockBot::GetParticles(
    PublicState* for_state,
    const Observation& infostate_observation,
    const Observation& public_observation) {
  std::unique_ptr<ParticleSetPartition> partition =
      MakeParticleSetPartition(*for_state, subgame_factory_->max_particles - 1,
                               subgame_factory_->particle_epsilon, false, rnd_gen_);
  std::unique_ptr<ParticleSet> set = std::move(partition->primary);
  AssignBeliefs(set.get());

  // We will augment the current particle set using the particle generator.
  ParticleGenerator* particle_generator = subgame_factory_->particle_generator.get();
  if (particle_generator) {  // Implemented currently only for goofspiel.
    // Make sure we always have 1 particle in the current infostate.
    // Using this removes the strong global consistency guarantee,
    // but it makes the algorithm always capable of playing the game.
    particle_generator->SetInfoState(infostate_observation, player_id_);
    std::unique_ptr<ParticleSet> infostate_targeted_set =
        particle_generator->GenerateParticles(1, /*max_rejection_cnt=*/100);
    SPIEL_CHECK_FALSE(infostate_targeted_set->particles.empty());
    AssignBeliefs(infostate_targeted_set.get());
    set->ImportSet(*infostate_targeted_set);

    // Generate even more particles if we starved too many of them between steps.
    if (set->size() < subgame_factory_->max_particles) {
      int num_new = subgame_factory_->max_particles - set->particles.size();
      particle_generator->SetPublicState(public_observation);
      std::unique_ptr<ParticleSet> pubstate_targeted_set =
          particle_generator->GenerateParticles(num_new, 10*num_new);
      AssignBeliefs(pubstate_targeted_set.get());
      set->ImportSet(*infostate_targeted_set);
    }
  }
  return set;
}

void SherlockBot::AssignBeliefs(ParticleSet* set) const {
  Observer* obs = subgame_factory_->infostate_observer.get();

  for (Particle& p : set->particles) {
    p.chance_reach = 1.;  // TODO: chance reach for other games (GS has no chance).
    p.player_reach[0] = 1.;
    p.player_reach[1] = 1.;

    std::unique_ptr<State> s = subgame_factory_->game->NewInitialState();
    for (int i = 0; i < p.history.size(); ++i) {
      if (s->IsChanceNode()) {
        ActionsAndProbs policy = s->ChanceOutcomes();
        Action a = p.history[i];
        double prob = GetProb(policy, a);
        p.chance_reach *= prob;
        s->ApplyAction(a);
      } else if (s->IsSimultaneousNode()) {
        for (int pl = 0; pl < 2; ++pl) {
          ActionsAndProbs policy = past_policy_.GetStatePolicy(obs->StringFrom(*s, pl));
          if (policy.empty()) {  // Such infostate was not visited in the past.
            policy = UniformStatePolicy(*s, pl);
          }
          Action a = p.history[i+pl];
          double prob = GetProb(policy, a);
          p.player_reach[pl] *= prob;
        }

        s->ApplyActions({p.history[i], p.history.at(++i)});
      } else {
        SPIEL_CHECK_FALSE(s->IsTerminal());
        Player pl = s->CurrentPlayer();
        ActionsAndProbs policy = past_policy_.GetStatePolicy(obs->StringFrom(*s, pl));
        if (policy.empty()) {  // Such infostate was not visited in the past.
          policy = UniformStatePolicy(*s, pl);
        }
        Action a = p.history[i];
        double prob = GetProb(policy, a);
        p.player_reach[pl] *= prob;
        s->ApplyAction(a);
      }
    }
  }
}

std::unique_ptr<Bot> MakeSherlockBot(
    std::shared_ptr<SubgameFactory> subgame_factory,
    std::shared_ptr<SolverFactory> solver_factory,
    Player player_id, int seed) {
  return std::make_unique<SherlockBot>(std::move(subgame_factory),
                                       std::move(solver_factory),
                                       player_id, seed);
}

}  // namespace papers_with_code
}  // namespace open_spiel

