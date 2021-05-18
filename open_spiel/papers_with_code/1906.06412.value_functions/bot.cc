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


  SherlockBot::SherlockBot(std::unique_ptr<SubgameFactory> subgame_factory,
              std::unique_ptr<SolverFactory> solver_factory,
              Player player_id, int seed)
    : subgame_factory_(std::move(subgame_factory)),
      solver_factory_(std::move(solver_factory)),
      player_id_(player_id),
      rnd_gen_(seed){
      subgame_ = subgame_factory_->MakeTrunk(1);

  }

    SherlockBot::SherlockBot(SherlockBot const &bot)
            : subgame_factory_(bot.subgame_factory_),
              solver_factory_(bot.solver_factory_),
              player_id_(bot.player_id_),
              rnd_gen_(bot.rnd_gen_),
              subgame_(std::make_shared<Subgame>(*bot.subgame_)){
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
  }

  std::pair<ActionsAndProbs, Action> SherlockBot::StepWithPolicy(const State& state) {

      SPIEL_CHECK_TRUE(subgame_);
    // First check if we will play
        // These should be provided by the referee at some point,
        // not accessed from the perfect-information State.
//        std::cout << "# Make observations\n";
        // infostate observations
        Observation infostate_observation(*subgame_factory_->game,subgame_factory_->infostate_observer);
        infostate_observation.SetFrom(state, player_id_);
        const std::string infostate =
                subgame_factory_->infostate_observer->StringFrom(state, player_id_);

        // public state observations
        Observation public_observation(*subgame_factory_->game,subgame_factory_->public_observer);
        public_observation.SetFrom(state, 0);
        PublicState *publicState = nullptr;
        for (PublicState& pubState : subgame_->public_states) {
            if (pubState.public_tensor == public_observation) {
                publicState = &pubState;
                break;
            }
        }
        SPIEL_CHECK_TRUE(publicState);

        // TODO: Tabularization of any Bot to compute offline TabularPolicy.
        //       Bot base class will need to add a Clone() method.

        // TODO: keep particles from previous step along with beliefs.
        //       Currently can work only for one-step lookahead trees.
//        std::cout << "# Generate particles for current public state\n";
        std::unique_ptr<ParticleSetPartition> partition = MakeParticleSetPartition(*publicState, pow(10,7), pow(10,-9),false,rnd_gen_);
        std::unique_ptr<ParticleSet> set = std::make_unique<ParticleSet>(partition->primary);
        //    std::unique_ptr<ParticleSet> set = GenerateParticles(
        //        infostate_observation,
        //        player_id_,
        //        subgame_factory_->max_particles,
        //        subgame_factory_->max_particles,
        //        // Make sure we always have 1 particle in the current infostate.
        //        // Using this removes the strong global consistency guarantee,
        //        // but it makes the algorithm always capable of playing the game.
        //        /*infostate_particles=*/1,
        //        rnd_gen_);
        SPIEL_CHECK_FALSE(set->particles.empty());

        //subgame_factory_->game->NewInitialState();

        // TODO: proper management of beliefs between steps. This is just
        //       a dummy initialization. (Not needed when I initialize from public state.)

        //    for (auto& particle: set->particles) {
        //      particle.chance_reach = 1.;
        //      particle.player_reach[0] = 1.;
        //      particle.player_reach[1] = 1.;
        //    }

        // We will do the gadget if we are resolving
//        std::cout << "# Making subgame\n";
        if (state.MoveNumber() > 0) {
            subgame_ = subgame_factory_->MakeSubgameSafeResolving(*set, player_id_, publicState->GetCFVs(1-player_id_));
        } else {
            subgame_ = subgame_factory_->MakeSubgame(*set);
        }

        // TODO: implement continual resolving.
        //  Update subgame's infostate trees: subgame->trees[1-player_id_]
        //  such that they begin with the choice for the opponent
        //  to follow or not into this subgame. This could be done by careful
        //  manipulation with the (already constructed) infostate tree,
        //  or with changing how the trees are constructed. Plumb this through
        //  MakeSubgame to affect infostate tree construction.

//        std::cout << "# Making solver\n";
        std::unique_ptr<SubgameSolver> solver = solver_factory_->MakeSolver(subgame_, nullptr, "", true);

//    // Code for opponent fixation:
//    TabularPolicy opponent_policy;  // Needs to be provided.
//    int opponent = 1 - player_id_;
//    algorithms::BanditVector& opponent_bandits = solver->bandits()[opponent];
//    for (algorithms::DecisionId id : opponent_bandits.range()) {
//      algorithms::InfostateNode* node = subgame->trees[opponent]->decision_infostate(id);
//      ActionsAndProbs infostate_policy = opponent_policy.GetStatePolicy(node->infostate_string());
//      std::vector<double> probs = GetProbs(infostate_policy);
//      auto fixable_bandit = std::make_unique<algorithms::bandits::FixableStrategy>(probs);
//      opponent_bandits[id] = std::move(fixable_bandit);
//    }

//        std::cout << "# Solving!\n";
      solver->RunSimultaneousIterations(solver_factory_->cfr_iterations);
      if(state.IsPlayerActing(player_id_)) {
          auto policy = std::make_shared<algorithms::BanditsAveragePolicy>(
                  subgame_->trees, solver->bandits());
          ActionsAndProbs actions_and_probs = policy->GetStatePolicy(infostate);
          SPIEL_CHECK_FALSE(actions_and_probs.empty());

          double p = std::uniform_real_distribution<>(0., 1.)(rnd_gen_);
          std::pair<Action, double> outcome = SampleAction(actions_and_probs, p);
          return {actions_and_probs, outcome.first};
      } else {
        // And we return empty actions and probs and -1 action
        ActionsAndProbs actions_and_probs;
        return {actions_and_probs, Action(-1)};
    }
  }

std::unique_ptr<Bot> MakeSherlockBot(
    std::unique_ptr<SubgameFactory> subgame_factory,
    std::unique_ptr<SolverFactory> solver_factory,
    Player player_id, int seed) {
  return std::make_unique<SherlockBot>(std::move(subgame_factory),
                                       std::move(solver_factory),
                                       player_id, seed);
}

std::unique_ptr<SherlockBot> MakeSherlockBot(
        std::unique_ptr<SubgameFactory> subgame_factory,
        std::unique_ptr<SolverFactory> solver_factory,
        Player player_id, int seed, bool sherlock_type) {
    return std::make_unique<SherlockBot>(std::move(subgame_factory),
            std::move(solver_factory),
            player_id, seed);
}

}  // namespace papers_with_code
}  // namespace open_spiel

