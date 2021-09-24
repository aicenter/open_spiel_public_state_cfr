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

#include "open_spiel/spiel_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/particle.h"

namespace open_spiel {
namespace papers_with_code {

std::unique_ptr<State> Particle::MakeState(const Game& game) const {
  std::unique_ptr<State> s = game.NewInitialState();
  for (int i = 0; i < history.size(); ++i) {
    if (s->IsSimultaneousNode()) {
      s->ApplyActions({history[i], history.at(++i)});
    } else {
      s->ApplyAction(history[i]);
    }
  }
  return  s;
}


void ParticleSet::AssignBeliefsTo(PublicState* state) const {
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(state->beliefs[pl].size(), state->nodes[pl].size());
    for (int i = 0; i < state->nodes[pl].size(); ++i) {
      // Assign beliefs based on a single particle.
      SPIEL_CHECK_FALSE(state->nodes[pl][i]->corresponding_states().empty());
      State* a_state = state->nodes[pl][i]->corresponding_states()[0].get();
      const Particle& particle = at(a_state->History());
      state->beliefs[pl][i] = particle.player_reach[pl];

      SPIEL_DCHECK({
         // All particles should have identical player beliefs
         // for the same infostates.
         for (const std::unique_ptr <State>& s:
              state->nodes[pl][i]->corresponding_states()) {
           const Particle& particle = at(s->History());
           SPIEL_CHECK_EQ(state->beliefs[pl][i], particle.player_reach[pl]);
         }
       });
    }
  }
}

void ParticleSet::ComputeBeliefs(const Game& game, const TabularPolicy& policy,
                                 const Observer& infostate_observer) {
  for (Particle& p: particles) {
    p.chance_reach = 1.;
    p.player_reach[0] = 1.;
    p.player_reach[1] = 1.;

    std::unique_ptr<State> s = game.NewInitialState();
    for (int i = 0; i < p.history.size(); ++i) {
      if (s->IsChanceNode()) {
        ActionsAndProbs infostate_policy = s->ChanceOutcomes();
        Action a = p.history[i];
        double prob = GetProb(infostate_policy, a);
        p.chance_reach *= prob;
        s->ApplyAction(a);
      } else if (s->IsSimultaneousNode()) {
        for (int pl = 0; pl < 2; ++pl) {
          ActionsAndProbs infostate_policy =
              policy.GetStatePolicy(infostate_observer.StringFrom(*s, pl));
          if (infostate_policy.empty()) {  // Such infostate was not visited in the past.
            infostate_policy = UniformStatePolicy(*s, pl);
          }
          Action a = p.history[i + pl];
          double prob = GetProb(infostate_policy, a);
          p.player_reach[pl] *= prob;
        }

        s->ApplyActions({p.history[i], p.history.at(++i)});
      } else {
        SPIEL_CHECK_FALSE(s->IsTerminal());
        Player pl = s->CurrentPlayer();
        ActionsAndProbs infostate_policy =
            policy.GetStatePolicy(infostate_observer.StringFrom(*s, pl));
        if (infostate_policy.empty()) {  // Such infostate was not visited in the past.
          infostate_policy = UniformStatePolicy(*s, pl);
        }
        Action a = p.history[i];
        double prob = GetProb(infostate_policy, a);
        p.player_reach[pl] *= prob;
        s->ApplyAction(a);
      }
    }
  }
}

Particle& ParticleSet::at(const std::vector<Action>& history) {
  for (Particle& particle : particles) {
    // Maybe should be SPIEL_CHECK_EQ in most games.
    if (particle.history.size() != history.size()) continue;
    // Compare in reverse, as the endings are where they will not be equal.
    if (std::equal(particle.history.rbegin(), particle.history.rend(),
                   history.rbegin())) return particle;
  }
  SpielFatalError("Particle not found");
}

int ParticleSet::index_of(const std::vector<Action>& history) const {
  for (int i = 0; i < particles.size(); ++i) {
    auto& particle = particles[i];
    // Maybe should be SPIEL_CHECK_EQ in most games.
    if (particle.history.size() != history.size()) continue;
    // Compare in reverse, as the endings are where they will not be equal.
    if (std::equal(particle.history.rbegin(), particle.history.rend(),
                   history.rbegin())) return i;
  }
  return -1;
}

const Particle& ParticleSet::at(const std::vector<Action>& history) const {
  for (const Particle& particle : particles) {
    // Maybe should be SPIEL_CHECK_EQ in most games.
    if (particle.history.size() != history.size()) continue;
    // Compare in reverse, as the endings are where they will not be equal.
    if (std::equal(particle.history.rbegin(), particle.history.rend(),
                   history.rbegin())) return particle;
  }
  SpielFatalError("Particle not found");
}
Particle& ParticleSet::add(const std::vector<Action>& history) {
  particles.emplace_back(history);
  return particles.back();
}

bool ParticleSet::has(const std::vector<Action>& history) const {
  for (const Particle& particle : particles) {
    // Maybe should be SPIEL_CHECK_EQ in most games.
    if (particle.history.size() != history.size()) continue;
    // Compare in reverse, as the endings are where they will not be equal.
    if (std::equal(particle.history.rbegin(), particle.history.rend(),
                   history.rbegin())) return true;
  }
  return false;
}

void CheckObservation(const Observation& actual, const Observation& expected) {
  SPIEL_CHECK_EQ(actual.tensor_info(), expected.tensor_info());
  SPIEL_CHECK_EQ(actual.Tensor(), expected.Tensor());
}

void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> public_observer,
                                 std::shared_ptr<Observer> infostate_observer,
                                 const ParticleSet& set) {
  SPIEL_CHECK_FALSE(set.particles.empty());
//  SPIEL_CHECK_FALSE(set.partition.empty());

  std::vector<std::unique_ptr<State>> histories;
  for (const Particle& particle : set.particles) {
    histories.push_back(particle.MakeState(game));
  }

  // Check public observations
  Observation expected_public(game, public_observer);
  Observation actual_public = expected_public;
  expected_public.SetFrom(*histories[0], kDefaultPlayerId);
  // Check reach probs consistency
  Observation infostate(game, infostate_observer);
  std::array<std::unordered_map<Observation, double>, 2> infostate_reach;

  for (int i = 0; i < set.particles.size(); ++i) {
    const std::unique_ptr<State>& history = histories[i];
    const Particle& particle = set.particles[i];
    actual_public.SetFrom(*history, kDefaultPlayerId);
    CheckObservation(actual_public, expected_public);

    for (int pl = 0; pl < 2; ++pl) {
      infostate.SetFrom(*history, pl);
      if (infostate_reach[pl].find(infostate) == infostate_reach[pl].end()) {
        infostate_reach[pl][infostate] = particle.player_reach[pl];
      } else {
        SPIEL_CHECK_EQ(infostate_reach[pl][infostate],
                       particle.player_reach[pl]);
      }
    }
  }
}

std::unique_ptr<ParticleSet> PickParticlesBasedOnReach(const PublicState& state,
                                                       int max_particles) {
  auto set = std::make_unique<ParticleSet>();
  set->particles.reserve(max_particles);

  // 1. Add all particles from the public state.
  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < state.nodes[pl].size(); ++i) {
      for (int k = 0; k < state.nodes[pl][i]->corresponding_states_size(); ++k) {
        const std::unique_ptr<State>& s = state.nodes[pl][i]->corresponding_states()[k];
        const double chn = state.nodes[pl][i]->corresponding_chance_reach_probs()[k];
        // While technically possible, no game makes zero chance transitions.
        SPIEL_CHECK_GT(chn, 0.);

        Particle& particle = pl == 0 ? set->add(s->History())
                                     : set->at(s->History());
        particle.player_reach[pl] = state.beliefs[pl][i];
        particle.chance_reach = chn;
      }
    }
  }

  if (set->particles.size() <= max_particles) {
    return set;
  }

  // 2. Sort the particles based on reach.
  std::sort(set->particles.begin(), set->particles.end(),
            [](const Particle& a, const Particle& b) {
              return a.reach() > b.reach();
            });

  // 3. Erase the particles that go over the limit.
  set->particles.erase(set->particles.begin() + max_particles,
                       set->particles.end());
  SPIEL_CHECK_EQ(set->size(), max_particles);
  return set;
}

void AddParticlesBelow(
    const algorithms::InfostateNode* node,
    const std::vector<const algorithms::InfostateNode*>& target_leaf_nodes,
    ParticleSet* set,
    int max_particles) {

  if (node->is_leaf_node()) {
    // Check if we arrived to the public state.
    if (std::find(target_leaf_nodes.begin(), target_leaf_nodes.end(), node)
        == target_leaf_nodes.end()) {
      return;
    }

    for (int i = 0; i < node->corresponding_states_size()
        && set->particles.size() < max_particles; i++) {
      auto h = node->corresponding_states()[i]->History();
      if (!set->has(h)) {
        set->add(h);
      }
    }
    return;
  }

  for (const algorithms::InfostateNode* child : node->child_iterator()) {
    if (set->particles.size() >= max_particles) return;
    AddParticlesBelow(child, target_leaf_nodes, set, max_particles);
  }
}

std::unique_ptr<ParticleSet> PickParticlesBasedOnQvalues(const PublicState& state,
                                                         int max_particles) {
  auto set = std::make_unique<ParticleSet>();

  // 1. Implementation for other games
  //    (mainly to make tests on games like Kuhn work).
  GameType g =
      state.nodes[0][0]->corresponding_states()[0]->GetGame()->GetType();
  if (g.short_name != "goofspiel") {
    // TODO: proper last Q-nodes tree slice implementation.
    std::cerr << "Particle selection is implemented only for goofspiel due "
                 "to game structure. Selecting all particles instead! \n";
    for (int j = 0; j < state.nodes[0].size(); j++) {
      for (const std::unique_ptr<State>& s
          : state.nodes[0][j]->corresponding_states()) {
        set->add(s->History());
      }
    }
    return set;
  }

  // 2. Actual particle selection on Goofspiel:
  //    Select particles as nodes with the highest Q-values within resolving.
  using Record = std::pair<double, const algorithms::InfostateNode*>;
  std::vector<Record> node_q_values;
  for (int pl = 0; pl < 2; ++pl) {
    for (algorithms::InfostateNode* dec_node
        : state.trees[pl]->AllDecisionInfostates()) {
      // Skip resolving nodes.
      if (absl::StartsWith(dec_node->infostate_string(),
                           algorithms::kFtInfostatePrefix))
        continue;

      for (algorithms::InfostateNode* q_node: dec_node->children()) {
        SPIEL_CHECK_EQ(q_node->type(), algorithms::kObservationInfostateNode);
        node_q_values.push_back({q_node->cumul_value, q_node});
      }
    }
  }

  std::sort(node_q_values.begin(), node_q_values.end(),
            [](const Record& a, const Record& b) { return a.first > b.first; });

  // 3. Keep adding particles until the limit is reached.
  for (const auto&[q_value, node]: node_q_values) {
    if (set->particles.size() < max_particles) {
      AddParticlesBelow(node, state.nodes[node->tree().acting_player()],
                        set.get(), max_particles);
    }
  }
  return set;
}

void GetStatesInSupport(const State* s, const Policy& policy,
                        std::vector<std::unique_ptr<State>>& out) {
  if (s->IsChanceNode()) {
    for (Action a:  s->LegalActions()) {
      std::unique_ptr<State> c = s->Child(a);
      GetStatesInSupport(c.get(), policy, out);
      out.push_back(std::move(c));
    }
  } else if (s->IsSimultaneousNode()) {
    std::array<std::vector<Action>, 2> play_actions;
    for (int pl = 0; pl < 2; ++pl) {
      ActionsAndProbs local_policy =
          policy.GetStatePolicy(s->InformationStateString(pl));
      std::vector<Action> actions =
          s->LegalActions(pl);
      SPIEL_CHECK_TRUE(local_policy.empty()
                       || actions.size() == local_policy.size());
      for (int i = 0; i < actions.size(); ++i) {
        if (local_policy.empty() || local_policy[i].second > 0.) {
          play_actions[pl].push_back(actions[i]);
        }
      }
    }
    SPIEL_CHECK_FALSE(play_actions[0].empty());
    SPIEL_CHECK_FALSE(play_actions[1].empty());
    for (int i = 0; i < play_actions[0].size(); ++i) {
      for (int j = 0; j < play_actions[1].size(); ++j) {
        std::unique_ptr<State> c = s->Clone();
        c->ApplyActions({play_actions[0][i], play_actions[1][j]});
        GetStatesInSupport(c.get(), policy, out);
        out.push_back(std::move(c));
      }
    }
  } else if (s->IsPlayerNode()) {
    Player pl = s->CurrentPlayer();
    ActionsAndProbs local_policy =
        policy.GetStatePolicy(s->InformationStateString(pl));
    std::vector<Action> actions =
        s->LegalActions(pl);
    SPIEL_CHECK_TRUE(local_policy.empty()
                     || actions.size() == local_policy.size());
    for (int i = 0; i < actions.size(); ++i) {
      if (local_policy.empty() || local_policy[i].second > 0.) {
        std::unique_ptr<State> c = s->Child(actions[i]);
        GetStatesInSupport(c.get(), policy, out);
        out.push_back(std::move(c));
      }
    }
  } else {
    SPIEL_CHECK_TRUE(s->IsTerminal());
  }
}

std::vector<std::unique_ptr<State>> GetStatesInSupport(const Game& game,
                                                       const Policy& policy) {
  std::vector<std::unique_ptr<State>> out;
  std::unique_ptr<State> s = game.NewInitialState();
  GetStatesInSupport(s.get(), policy, out);
  return out;
}


}  // papers_with_code

std::ostream& operator<<(std::ostream& os,
                         const papers_with_code::Particle& particle) {
  const std::vector<Action>& h = particle.history;
  for (Action a : h) os << a << ' ';
  return os << " with reach p=" << particle.reach();
}

}  // open_spiel
