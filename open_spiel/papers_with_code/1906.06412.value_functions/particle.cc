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


void ParticleSet::AssignBeliefs(PublicState& state) const {
  for (int pl = 0; pl < 2; ++pl) {
    SPIEL_CHECK_EQ(state.beliefs[pl].size(), state.nodes[pl].size());
    for (int i = 0; i < state.nodes[pl].size(); ++i) {
      // Assign beliefs based on a single particle.
      SPIEL_CHECK_FALSE(state.nodes[pl][i]->corresponding_states().empty());
      State* a_state = state.nodes[pl][i]->corresponding_states()[0].get();
      const Particle& particle = at(a_state->History());
      state.beliefs[pl][i] = particle.player_reach[pl];

      SPIEL_DCHECK({
         // All particles should have identical player beliefs
         // for the same infostates.
         for (const std::unique_ptr <State>& s:
              state.nodes[pl][i]->corresponding_states()) {
           const Particle& particle = at(s->History());
           SPIEL_CHECK_EQ(state.beliefs[pl][i], particle.player_reach[pl]);
         }
       });
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

void CheckParticleSetConsistency(const Game& game,
                                 std::shared_ptr<Observer> infostate_observer,
                                 std::vector<std::vector<algorithms::InfostateNode*>> infostate_nodes,
                                 const ParticleSet& set) {
  SPIEL_CHECK_FALSE(set.particles.empty());
//  SPIEL_CHECK_FALSE(set.partition.empty());

  std::vector<std::unique_ptr<State>> histories;
  for (const Particle& particle : set.particles) {
    histories.push_back(particle.MakeState(game));
  }

//  for (int pl = 0; pl < 2; ++pl) {
//    SPIEL_CHECK_FALSE(set.partition[pl].empty());
//    SPIEL_CHECK_FALSE(infostate_nodes[pl].empty());
//    SPIEL_CHECK_EQ(set.partition[pl].size(), infostate_nodes[pl].size());
//
//    for (int i = 0; i < set.partition[pl].size(); ++i) {
//      SPIEL_CHECK_FALSE(set.partition[pl][i].empty());
//      for (int particle_idx : set.partition[pl][i]) {
//        const algorithms::InfostateNode* node = infostate_nodes[pl][i];
//        const State& h = *histories[particle_idx];
//        SPIEL_CHECK_TRUE(node);
//        SPIEL_CHECK_EQ(infostate_observer->StringFrom(h, pl),
//                       node->infostate_string());
//      }
//    }
//  }

}
std::unique_ptr<ParticleSetPartition> MakeParticleSetPartition(
    const PublicState& state,
    int primary_max_particles, double epsilon,
    bool save_secondary, std::mt19937& rnd_gen) {
  // TODO: make work well for case "primary_max_particles == 0"
  ParticleSet all;
  all.particles.reserve(primary_max_particles);

  for (int pl = 0; pl < 2; ++pl) {
    for (int i = 0; i < state.nodes[pl].size(); ++i) {
      for (int k = 0; k < state.nodes[pl][i]->corresponding_states_size(); ++k) {
        const std::unique_ptr<State>& s = state.nodes[pl][i]->corresponding_states()[k];
        const double chn = state.nodes[pl][i]->corresponding_chance_reach_probs()[k];
        // While technically possible, no game makes zero chance transitions.
        SPIEL_CHECK_GT(chn, 0.);

        Particle& particle = pl == 0 ? all.add(s->History())
                                     : all.at(s->History());
        particle.player_reach[pl] = state.beliefs[pl][i];
        particle.chance_reach = chn;
      }
    }
  }

  auto partition = std::make_unique<ParticleSetPartition>();
  if (all.particles.size() <= primary_max_particles) {  // Fast track return.
    partition->primary = std::move(all);
    return partition;  // Secondary is empty.
  }

  // We have more particles (N) than needed (K):
  // N choose (at most) K without repetition, using a discrete probability
  // distribution D(U,R).
  // The distribution D is epsilon-convex combination of a) uniform U and
  // b) the normalized reach probs distribution R.

  // Compute normalization factor of reach probs distribution R.
  double norm_r = 0.;
  for (const Particle& p : all.particles) norm_r += p.reach();
  SPIEL_CHECK_GT(norm_r, 0.);

  // Prepare CDF of distribution D.
  std::map</*cumul=*/double, /*particle_index=*/int> cdf;
  double cumul = 0.;
  int n = all.particles.size();
  int zero_entries = 0;
  for (int i = 0; i < n; ++i) {
    // D = U + R
    double p = epsilon / n
             + (1-epsilon) * all.particles[i].reach() / norm_r;
    if (cumul + p == cumul) {
      // Do not add this to the cdf, as it cannot be sampled.
      ++zero_entries;
    } else {
      cumul += p;
      cdf[cumul] = i;
    }
  }
  SPIEL_CHECK_FLOAT_NEAR(cumul, 1., 1e-6);

  // Pick K particles.
  int k = primary_max_particles;

  // However, some particles could have had (near) zero probability:
  // we omit those choices.
  if (k >= n - zero_entries) {  // Fast track return.
    SPIEL_CHECK_GT(n, zero_entries);
    // Keep only the non-zero reach entries.
    for(auto& [cumul, particle_index] : cdf) {
      partition->primary.particles.push_back(all.particles[particle_index]);
    }
    return partition;  // Secondary is empty.
  }

  // Finally, pick the indices iteratively based on the CDF.
  // We remove the selected choices from the CDF (otherwise it may take a long
  // time for the while loop to terminate).
  std::set<int> pick;
  std::uniform_real_distribution<double> unif(0., 1.);  // Interval [0,1)
  while (pick.size() != k) {
    double p = unif(rnd_gen);
    auto it = cdf.upper_bound(p);
    if (it == cdf.end()) {
      // This can happen due to iterative removal from cdf.
      SPIEL_CHECK_FALSE(cdf.empty());
      --it;
    }
    int particle_index = it->second;
    pick.insert(particle_index);
    cdf.erase(it);  // TODO: think about more if this is indeed correct.
  }

  // Construct primary set based on the pick.
  partition->primary.particles.reserve(primary_max_particles);
  for(int particle_index : pick) {
    partition->primary.particles.push_back(all.particles[particle_index]);
  }
  // Construct secondary set as all the other particles.
  if (save_secondary) {
    for (int i = 0; i < n; ++i) {
      if (pick.count(i) != 0) continue;
      partition->secondary.particles.push_back(all.particles[i]);
    }
  }

  return partition;
}

}  // papers_with_code

std::ostream& operator<<(std::ostream& os,
                         const papers_with_code::Particle& particle) {
  const std::vector<Action>& h = particle.history;
  for (Action a : h) os << a << ' ';
  return os << " with reach p=" << particle.reach();
}

}  // open_spiel
