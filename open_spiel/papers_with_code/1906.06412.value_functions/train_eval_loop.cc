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


// -- FLAGS --------------------------------------------------------------------

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

// -- General --
ABSL_FLAG(int, seed, 0, "Seed.");
ABSL_FLAG(std::string, device, "cpu", "Device used for neural networks.");
ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");

// -- Solver --
ABSL_FLAG(std::string, use_bandits_for_cfr, "RegretMatchingPlus",
          "Which bandit should be used in the trunk.");
ABSL_FLAG(bool, safe_resolving, true,
          "Should the online bot use safe resolving? "
          "(used for br metric, not training)");
ABSL_FLAG(int, cfr_iterations, 100,
          "Number of iterations with value function (used in bootstrapping).");
ABSL_FLAG(int, max_move_ahead_limit, 1, "Size of the lookahead tree.");
ABSL_FLAG(bool, beliefs_for_average, false,
          "Use average beliefs for leaf evaluation instead of current beliefs.");
ABSL_FLAG(double, noisy_values, 0.,
          "Additive noise for the value function, sigma of the normal "
          "distribution centered at zero");
ABSL_FLAG(double, opponent_beliefs_eps, 0.,
          "How much the opponent's beliefs should be mixed with 1. beliefs per "
          "infostate when constructing gadget game for safe resolving.");

// -- Data generation --
ABSL_FLAG(int, depth, 3, "Depth of the trunk.");
ABSL_FLAG(double, prob_pure_strat, 0.1, "Params for random generation.");
ABSL_FLAG(double, prob_fully_mixed, 0.05, "Params for random generation.");
ABSL_FLAG(double, prob_benford_dist, 0.0, "Params for random generation.");
ABSL_FLAG(int, replay_size, 100,
          "Size of experience replay in terms of public states.");
ABSL_FLAG(int, cfr_oracle_iterations, 100, "Number of oracle iterations.");
ABSL_FLAG(double, sparse_epsilon, 0.,
          "How uniformly should the particles be sampled? [0,1] interval");

ABSL_FLAG(std::string, exp_init, "trunk_random",
          "Init experience. See options in code.");
ABSL_FLAG(int, exp_init_size, -1, "How many experiences should be initialized."
                                  " -1 fills replay buffer.");
ABSL_FLAG(std::string, exp_loop, "nothing",
          "How the experience replay should be updated.");
ABSL_FLAG(int, exp_loop_new, 128,
          "Update experience replay every n steps of the train-eval loop.");
ABSL_FLAG(int, exp_update_size, 128, "How many experiences should be updated. "
                                     "-1 for the whole replay buffer");
ABSL_FLAG(bool, exp_reset_nn, false,
          "Should the neural net be reset and trained from scratch, "
          "when new experience is made?");
ABSL_FLAG(bool, bootstrap_reset_nn, false,
          "Should the neural net be reset and trained from scratch, "
          "when final bootstrap is made?");
ABSL_FLAG(int, bootstrap_from_move, 1,
          "From what depth should the bootstrapping happen?");
ABSL_FLAG(std::string, save_values_policy, "average",
          "What cf. values should be saved after solving the subgame: "
          " one of {current,average}.");
ABSL_FLAG(int, ismcts_num_matches, 1000,
          "Number of matches for IS-MCTS playthroughs.");
ABSL_FLAG(int, ismcts_max_simulations, 100,
          "Number of simulations for IS-MCTS playthroughs.");

// -- Training --
ABSL_FLAG(int, train_batches, 32,
          "Number of training batches before the evalution is run.");
ABSL_FLAG(int, batch_size, 32,
          "Batch size per train step. If <1, then full replay buffer is used.");
ABSL_FLAG(int, num_loops, 5000, "Number of train-eval loops.");
ABSL_FLAG(std::string, optimizer, "adam", "Optimizer. One of adam/sgd");
ABSL_FLAG(double, learning_rate, 1e-3, "Optimizer learning rate.");
ABSL_FLAG(double, lr_decay, 1., "Learning rate decay after each loop.");
ABSL_FLAG(int, max_particles, -1,
          "Max particles to use. Set -1 to find an upper bound automatically.");

// -- Network --
ABSL_FLAG(std::string, arch, "particle_vf",
          "Which architecture of the value function should be used.");
ABSL_FLAG(int, num_layers, 3, "Number of hidden layers.");
ABSL_FLAG(int, num_width, 3, "Multiplicative constant of the number of neurons "
                             "per layer compared to the input size.");
ABSL_FLAG(std::string, num_inputs_regression, "128",
          "Size of the regression input for particle VF. "
          "'max' means it will be the same as max_parviews."
          "'2x' means it will be twice the number of particle features.");
ABSL_FLAG(bool, zero_sum_regression, false,
          "Make the regressed values automatically zero-sum through special "
          "layer (does not introduce any new weights)");
ABSL_FLAG(bool, normalize_beliefs, false,
          "Normalize the per-player beliefs and values accordingly.");
ABSL_FLAG(std::string, set_pooling, "sum",
          "Pooling operation for deep sets (sum/mean).");

// -- Model checkpoints (snapshots) --
ABSL_FLAG(int, snapshot_loop, -1,
          "When should NN weights be saved to snapshot/ dir? -1 for never.");
ABSL_FLAG(std::string, snapshot_dir, "snapshots/",
          "Directory to store snapshots of NN weights.");
ABSL_FLAG(std::string, load_snapshot, "",
          "Absolute path to the snapshot that should be loaded. "
          "A special keyword 'automatic' will load the latest snapshot "
          " found in the snapshot_dir.");

// -- Bot --
ABSL_FLAG(int, bot_particles, -1,
          "Number of particles the bot should use when stepping to the next "
          "public state. -1 to set the limit same as max_particles.");
ABSL_FLAG(bool, bot_use_oracle, false,
          "Use oracle instead of net as leaf evaluator.");

// -- Metrics --
// Validation loss
ABSL_FLAG(int, val_experiences, 0,
          "Number of separate experiences that should be used for computation "
          "of validation loss.");
ABSL_FLAG(std::string, val_init, "nothing",
          "How the data for validation loss computation should be generated.");

// FullTrunkExplMetric
ABSL_FLAG(std::string, trunk_expl_iterations, "",
          "Evaluate trunk exploitability for each trunk iteration.");
// ReplayVisitsMetric
ABSL_FLAG(int, replay_visits_window, -1,
          "Track the average visit count over a past window "
          "behind the head in experience replay");
ABSL_FLAG(bool, track_time, false, "Track time between loops");
ABSL_FLAG(bool, track_lr, false, "Track time between loops");
// IigsBrMetric
ABSL_FLAG(bool, iigs_br_metric, false,
          "Compute domain-specific BR for IIGS(N,K)");
ABSL_FLAG(bool, iigs_approx_response, true,
          "Make an approximate version of the response, faster to compute.");
// BrMetric
ABSL_FLAG(bool, br_metric, false,
          "Compute domain-agnostic BR (can be very slow)");


// -----------------------------------------------------------------------------

#include <fenv.h>
#include "absl/random/random.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/include_libs_ordered.h"

#include "open_spiel/game_transforms/turn_based_simultaneous_game.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/metrics.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/ismcts_playthroughs.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/bot.h"

namespace open_spiel {
namespace papers_with_code {


std::vector<int> ItersFromString(const std::string& s) {
  std::vector<std::string> xs = absl::StrSplit(s, ',');
  std::vector<int> out;
  for (auto& x : xs) {
    if (!x.empty()) out.push_back(std::stoi(x));
  }
  return out;
}

double TrainNetwork(ValueNet* model, torch::Device* device,
                    torch::optim::Optimizer* optimizer,
                    BatchData* batch) {
  SPIEL_DCHECK_TRUE(model->is_training());
  torch::Tensor data = batch->data.to(*device);
  torch::Tensor target = model->PrepareTarget(batch).to(*device);
  SPIEL_DCHECK_TRUE(torch::isfinite(data).all().item<bool>());
  SPIEL_DCHECK_TRUE(torch::isfinite(target).all().item<bool>());
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  SPIEL_DCHECK_TRUE(torch::isfinite(output).all().item<bool>());
  torch::Tensor loss = torch::mse_loss(output, target);
  SPIEL_CHECK_TRUE(std::isfinite(loss.item<float>()));
  loss.backward();
  optimizer->step();
  SPIEL_DCHECK_TRUE(model->isfinite());
  return loss.item().to<double>();
}

void DecayLearningRate(torch::optim::Optimizer* optimizer, double lr_decay) {
  for (auto &group : optimizer->param_groups()) {
    if(group.has_options()) {
      auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
      options.lr(options.lr() * lr_decay);
    }
  }
}

void ResetLearningRate(torch::optim::Optimizer* optimizer, double lr) {
  for (auto &group : optimizer->param_groups()) {
    if(group.has_options()) {
      auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
      options.lr(lr);
    }
  }
}

int LargestPublicState(const std::vector<PublicState>& states) {
  int max_size = 0;
  for (const PublicState& state : states) {
    int state_size = 0;
    for (const algorithms::InfostateNode* node : state.nodes[0]) {
      state_size += node->corresponding_states_size();
    }
    max_size = std::max(max_size, state_size);
  }
  SPIEL_CHECK_GT(max_size, 0);
  return max_size;
}

std::shared_ptr<torch::optim::Optimizer> MakeOptimizer(ValueNet* model) {
  std::string choice = absl::GetFlag(FLAGS_optimizer);
  if (choice == "sgd") {
    SpielFatalError("Not supported");
//    return std::make_unique<torch::optim::SGD>(
//        model->parameters(),
//        torch::optim::SGDOptions(absl::GetFlag(FLAGS_learning_rate)));
  }
  if (choice == "adam") {
    return std::make_shared<torch::optim::Adam>(
        model->parameters(),
        torch::optim::AdamOptions(absl::GetFlag(FLAGS_learning_rate)));
  }
  SpielFatalError("Unknown optimizer!");
}

double size_mb(double replay_size, double input_size, double output_size) {
  return replay_size * (input_size + output_size) / 256. / 1024.;
}

void TrainEvalLoop() {
  // Create all the data structures needed for the training-evaluation loop.
  // There is quite a lot of them.

  // Replicable experiments FTW!
  std::cout << "# Seeding experiment ..." << std::endl;
  const int seed = absl::GetFlag(FLAGS_seed);
  torch::manual_seed(seed);
  // All source of randomness need to plumb this through.
  std::shared_ptr<std::mt19937> rnd_gen = std::make_shared<std::mt19937>(seed);
  //
  std::cout << "# Making strategy randomizer ..." << std::endl;
  StrategyRandomizer randomizer;
  randomizer.rnd_gen = rnd_gen;
  randomizer.prob_pure_strat = absl::GetFlag(FLAGS_prob_pure_strat);
  randomizer.prob_fully_mixed = absl::GetFlag(FLAGS_prob_fully_mixed);
  randomizer.prob_benford_dist = absl::GetFlag(FLAGS_prob_benford_dist);
  //
  std::cout << "# Making subgame subgame_factory ..." << std::endl;
  auto subgame_factory = std::make_shared<SubgameFactory>();
  auto game = subgame_factory->game     = LoadGame(absl::GetFlag(FLAGS_game_name));
  subgame_factory->infostate_observer   = game->MakeObserver(kInfoStateObsType, {});
  subgame_factory->public_observer      = game->MakeObserver(kPublicStateObsType, {});
  subgame_factory->hand_observer        = game->MakeObserver(kHandHistoryObsType, {});
  subgame_factory->max_move_ahead_limit = absl::GetFlag(FLAGS_max_move_ahead_limit);
  subgame_factory->max_trunk_depth      = absl::GetFlag(FLAGS_depth);
  subgame_factory->max_particles        = absl::GetFlag(FLAGS_max_particles);
  if (game->GetType().short_name == "goofspiel") {
    auto goof_game =
        std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game);
    subgame_factory->particle_generator =
        std::make_shared<ParticleGenerator>(goof_game, rnd_gen);
  }
  //
  std::cout << "# Making oracle evaluator ..." << std::endl;
  const PolicySelection save_values_policy =
      GetSaveValuesPolicy(absl::GetFlag(FLAGS_save_values_policy));
  auto terminal_evaluator = std::make_shared<TerminalEvaluator>();
  auto oracle = std::make_shared<CFREvaluator>(
      subgame_factory->game, algorithms::kNoMoveAheadLimit,
      /*no_leaf_evaluator=*/nullptr, terminal_evaluator,
      subgame_factory->public_observer, subgame_factory->infostate_observer);
  oracle->bandit_name = absl::GetFlag(FLAGS_use_bandits_for_cfr);
  oracle->num_cfr_iterations = absl::GetFlag(FLAGS_cfr_oracle_iterations);
  oracle->save_values_policy = save_values_policy;
  //
  std::cout << "# Init empty reusable structures ..." << std::endl;
  ReusableStructures reuse(subgame_factory.get(),
                           /*solver_factory=*/nullptr, // Supplied later.
                           oracle);
  //
  std::cout << "# Setting IS-MCTS config ..." << std::endl;
  reuse.playthroughs = std::make_unique<IsmctsPlaythroughs>();
  reuse.playthroughs->num_matches     = absl::GetFlag(FLAGS_ismcts_num_matches);
  reuse.playthroughs->max_simulations = absl::GetFlag(FLAGS_ismcts_max_simulations);
  //
  const NetArchitecture arch = GetArchitecture(absl::GetFlag(FLAGS_arch));
  std::cout << "# Deducing dimensions ..." << std::endl;
  std::shared_ptr<BasicDims> dims = DeduceBasicDims(
      arch, *game, subgame_factory->public_observer, subgame_factory->hand_observer);
  std::cout << "# Public features: " << dims->public_features_size << std::endl;
  std::cout << "# Hand features: " << dims->hand_features_size << std::endl;
  //
  std::cout << "# Deducing dimensions for specific VFs ..." << std::endl;
  if (subgame_factory->max_particles >= 1) {  // Static setting.
    auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
    particle_dims->max_parviews = subgame_factory->max_particles * 2;
    std::cout << "# Particle VF (set statically): "
                 "max_particles = " << subgame_factory->max_particles << ' ' <<
              "max_parviews = " << particle_dims->max_parviews << std::endl;
  }
  std::shared_ptr<HandInfo> hand_info;
  if (arch == NetArchitecture::kPositional || subgame_factory->max_particles == -1) {
    std::cout << "# Finding all hands in the game (may take a while) ..."
              << std::endl;
    PublicStatesInGame* all_states = reuse.GetAllPublicStates();
    const std::vector<PublicState>& public_states = all_states->public_states;
    std::cout << "# Number of public states: "
              << all_states->public_states.size() << std::endl;
    hand_info = MakeHandInfo(*game, subgame_factory->hand_observer, public_states);
    // Hand info can be computed only based on the trunk, or can be provided
    // in a domain-dependent manner in games where it is possible thanks to
    // the structure of the game (Poker / Liar's dice)
//    Subgame* trunk_states = reuse.GetFixableTrunkWithOracle();
//    const std::vector<PublicState>& public_states =
//        trunk_states->public_states();
//    std::cout << "# Number of public states: "
//              << trunk_states->public_states().size() << std::endl;
//    hand_info = MakeHandInfo(*game, subgame_factory->hand_observer,
//                             trunk_states->public_states());

    if (arch == NetArchitecture::kPositional) {
      auto pos_dims = open_spiel::down_cast<PositionalDims*>(dims.get());
      for (int pl = 0; pl < 2; ++pl) {
        pos_dims->net_ranges_size[pl] = hand_info->tables[pl].private_hands.size();
      }
      std::cout << "# Positional VF: net_ranges_size = "
                << pos_dims->net_ranges_size << std::endl;
    }
    if (arch == NetArchitecture::kParticle) {
      auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
      subgame_factory->max_particles = LargestPublicState(public_states);
      particle_dims->max_parviews = hand_info->num_hands();
      std::cout << "# Particle VF (derived automatically): "
                   "max_particles = " << subgame_factory->max_particles << ' ' <<
                "max_parviews = " << particle_dims->max_parviews << "\n";
      std::cout << "# Full features for a particle: "
                << particle_dims->full_features_size() << "\n";
    }
  }
  std::cout << "# Point input size: " << dims->point_input_size() << "\n";
  std::cout << "# Point output size: " << dims->point_output_size() << "\n";
  //
  std::cout << "# Using device: " << absl::GetFlag(FLAGS_device) << "\n";
  torch::Device device(absl::GetFlag(FLAGS_device));
  //
  int num_inputs_regression = -1;
  if (arch == NetArchitecture::kParticle) {
    auto num_inputs_regression_str = absl::GetFlag(FLAGS_num_inputs_regression);
    if(num_inputs_regression_str == "max") {
      auto particle_dims = open_spiel::down_cast <ParticleDims*>(dims.get());
      num_inputs_regression = particle_dims->max_parviews;
    } else if(absl::EndsWith(num_inputs_regression_str, "x")) {
      auto particle_dims = open_spiel::down_cast <ParticleDims*>(dims.get());
      int multiplicative_factor = std::stoi(num_inputs_regression_str);
      std::cout << "# Using multiplicative factor for num regression inputs: "
                << multiplicative_factor << std::endl;
      num_inputs_regression = particle_dims->full_features_size()
                            * multiplicative_factor;
    } else {
      num_inputs_regression = std::stoi(num_inputs_regression_str);
    }
    std::cout << "# Size of the input for regression component: "
              << num_inputs_regression << "\n";
    SPIEL_CHECK_GE(num_inputs_regression, 1);  // Must be provided!
  }
  //
  std::cout << "# Making eval data ..." << std::endl;
  auto eval_batch = std::make_shared<BatchData>(1,
                                                dims->point_input_size(),
                                                dims->point_output_size());
  //
  std::cout << "# Creating model ..." << std::endl;
  std::shared_ptr<ValueNet> model = MakeModel(
      arch, dims,
      absl::GetFlag(FLAGS_num_layers), absl::GetFlag(FLAGS_num_width),
      num_inputs_regression, absl::GetFlag(FLAGS_zero_sum_regression),
      absl::GetFlag(FLAGS_normalize_beliefs),
      GetPoolingOp(absl::GetFlag(FLAGS_set_pooling)));
  std::cout << "# Model has " << model->num_parameters()
            << " trainable params" << std::endl;
  //
  std::string load_snapshot = absl::GetFlag(FLAGS_load_snapshot);
  const std::string snapshot_dir = absl::GetFlag(FLAGS_snapshot_dir);
  if (load_snapshot == kLoadAutomaticSnapshot) {
    std::cout << "# Finding most recent snapshot automatically ..." << std::endl;
    load_snapshot = FindSnapshot(snapshot_dir);
    if (!load_snapshot.empty())
      std::cout << "# A snapshot was found." << std::endl;
  }
  if (!load_snapshot.empty()) {
    std::cout << "# Loading model from snapshot: " << load_snapshot
              << " ..." << std::endl;
    LoadNetSnapshot(model, load_snapshot);
  } else {
    std::cout << "# No snapshot was specified, training from scratch."
              << std::endl;
  }
  model->to(device);
  //
  std::cout << "# Creating optimizer ..." << std::endl;
  std::shared_ptr<torch::optim::Optimizer> optimizer =
      MakeOptimizer(model.get());
  const double lr_decay = absl::GetFlag(FLAGS_lr_decay);
  const double learning_rate = absl::GetFlag(FLAGS_learning_rate);
  //
  const int replay_size = absl::GetFlag(FLAGS_replay_size);
  std::cout << "# Allocating experience replay ("
            << size_mb(replay_size,
                       dims->point_input_size(),
                       dims->point_output_size())
            << " MB) ..." << std::endl;
  ExperienceReplay experience_replay(replay_size, dims->point_input_size(),
                                     dims->point_output_size());
  //
  std::cout << "# Making batch train data ..." << std::endl;
  const int batch_size = absl::GetFlag(FLAGS_batch_size) > 0
                       ? std::min(absl::GetFlag(FLAGS_batch_size),
                                  experience_replay.size())
                       : experience_replay.size();
  BatchData train_batch(batch_size,
                        dims->point_input_size(),
                        dims->point_output_size());
  //
  auto solver_factory = std::make_shared<SolverFactory>();
  std::cout << "# Making net evaluator ..." << std::endl;
  solver_factory->cfr_iterations = absl::GetFlag(FLAGS_cfr_iterations);
  solver_factory->use_bandits_for_cfr  = absl::GetFlag(FLAGS_use_bandits_for_cfr);
  solver_factory->safe_resolving       = false;  // No resolving for training.
  solver_factory->beliefs_for_average  = absl::GetFlag(FLAGS_beliefs_for_average);
  solver_factory->opponent_beliefs_eps = absl::GetFlag(FLAGS_opponent_beliefs_eps);
  solver_factory->save_values_policy   = save_values_policy;
  solver_factory->terminal_evaluator   = terminal_evaluator;
  solver_factory->leaf_evaluator = MakeNetEvaluator(
      dims, model, eval_batch, device, rnd_gen,
      hand_info,  // May be nullptr for particle VF.
      subgame_factory->hand_observer);
  solver_factory->rnd_gen = rnd_gen;
  reuse.solver_factory = solver_factory.get();
  //
  std::cout << "# Making replay filler ..." << std::endl;
  ReplayFiller filler;
  filler.replay     = &experience_replay;
  filler.subgame_factory = subgame_factory.get();
  filler.solver_factory  = solver_factory.get();
  filler.dims       = dims.get();
  filler.randomizer = &randomizer;
  filler.reuse      = &reuse;
  filler.arch       = arch;
  filler.normalize_beliefs    = absl::GetFlag(FLAGS_normalize_beliefs);
  filler.sparse_epsilon       = absl::GetFlag(FLAGS_sparse_epsilon);
  filler.eval_iters =
      ItersFromString(absl::GetFlag(FLAGS_trunk_expl_iterations));
  //
  std::cout << "# Making bot for online play ..." << std::endl;
  // Make a separate solver/subgame copies for the bot,
  // so they can have custom settings.
  auto bot_subgame_factory = std::make_shared<SubgameFactory>(*subgame_factory);
  int bot_particles = absl::GetFlag(FLAGS_bot_particles);
  if (bot_particles > -1) {
    bot_subgame_factory->max_particles = bot_particles;
  }
  auto bot_solver_factory = std::make_shared<SolverFactory>(*solver_factory);
  // Make sure that bot has a different rnd_gen, but seeded in the same way.
  // This makes sure that different bot settings will have no effect on the
  // stochasticity of the training procedure.
  bot_solver_factory->rnd_gen = std::make_shared<std::mt19937>(seed);
  bot_solver_factory->safe_resolving = absl::GetFlag(FLAGS_safe_resolving);
  bot_solver_factory->noisy_values   = absl::GetFlag(FLAGS_noisy_values);
  if (absl::GetFlag(FLAGS_bot_use_oracle)) {
    bot_solver_factory->leaf_evaluator = oracle;
  }
  // Make the factories accessible for reuse.
  reuse.bot_subgame_factory = bot_subgame_factory;
  reuse.bot_solver_factory = bot_solver_factory;
  //
  std::cout << "# Making evaluation metrics ..." << std::endl;
  std::vector<std::unique_ptr<Metric>> metrics;
  {
    std::vector<int> trunk_expl_iterations =
        ItersFromString(absl::GetFlag(FLAGS_trunk_expl_iterations));
    if (!trunk_expl_iterations.empty()) {
      std::cout << "# Making full trunk exploitability metric ..." << std::endl;
      metrics.push_back(MakeFullTrunkExplMetric(
          trunk_expl_iterations, reuse.GetTrunkWithNet(), reuse.GetSfLp()));
    }
  }
  {
    int window = absl::GetFlag(FLAGS_replay_visits_window);
    if (window > 0) {
      std::cout << "# Making replay visits metric ..." << std::endl;
      SPIEL_CHECK_LE(window, replay_size);
      metrics.push_back(MakeReplayVisitsMetric(&experience_replay, window));
    }
  }
  if (absl::GetFlag(FLAGS_track_lr)) {
    std::cout << "# Making tracking learning rate ..." << std::endl;
    metrics.push_back(MakeTrackLearningRate(optimizer.get()));
  }
  if (absl::GetFlag(FLAGS_iigs_br_metric)) {
    std::cout << "# Making IIGS BR metric ..." << std::endl;
    auto goof_game =
        std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game);
    bool approx_response = absl::GetFlag(FLAGS_iigs_approx_response);
    metrics.push_back(MakeIigsBrMetric(
        MakeSherlockBot(bot_subgame_factory, bot_solver_factory),
        goof_game, approx_response));
  }
  if (absl::GetFlag(FLAGS_br_metric)) {
    std::cout << "# Making BR metric ..." << std::endl;
    metrics.push_back(MakeBrMetric(
        MakeSherlockBot(bot_subgame_factory, bot_solver_factory), game));
  }
  if (absl::GetFlag(FLAGS_val_experiences) > 0) {
    std::cout << "# Making validation loss metric ..." << std::endl;
    metrics.push_back(MakeValidationLossMetric(
        filler, model, &device,
        GetReplayFillerPolicy(absl::GetFlag(FLAGS_val_init)),
        absl::GetFlag(FLAGS_val_experiences)));
  }
  if (absl::GetFlag(FLAGS_track_time)) {
    std::cout << "# Making tracking time metric ..." << std::endl;
    metrics.push_back(MakeTrackTimeMetric());
  }
  //
  ReplayFillerPolicy exp_init =
      GetReplayFillerPolicy(absl::GetFlag(FLAGS_exp_init));
  SPIEL_CHECK_NE(exp_init, kNothing);
  int exp_init_size = absl::GetFlag(FLAGS_exp_init_size);
  //
  ReplayFillerPolicy exp_loop =
      GetReplayFillerPolicy(absl::GetFlag(FLAGS_exp_loop));
  int exp_loop_new = absl::GetFlag(FLAGS_exp_loop_new);
  // Must have some non-zero value, so we can also init experience.
  // Use exp_loop="nothing" to skip experience regeneration.
  SPIEL_CHECK_GT(exp_loop_new, 0);
  int exp_update_size = absl::GetFlag(FLAGS_exp_update_size);
  if (exp_update_size == -1) exp_update_size = replay_size;
  bool exp_reset_nn = absl::GetFlag(FLAGS_exp_reset_nn);
  //
  const int num_loops = absl::GetFlag(FLAGS_num_loops);
  const int train_batches = absl::GetFlag(FLAGS_train_batches);
  //
  if (exp_loop == kBootstrap || exp_loop == kIsmctsBootstrap) {
    SPIEL_CHECK_TRUE(exp_init == kBootstrap || exp_init == kIsmctsBootstrap);
    // Number of times the experience should be regenerated.
    const int num_regenerations = num_loops / exp_loop_new;
    // Bootstrap replay must hold all created experiences,
    // i.e. from initialization as well as from regeneration.
    const int bootstrap_size = exp_update_size * num_regenerations;

    std::cout << "# Number of regenerations: " << num_regenerations << "\n";
    std::cout << "# Will be bootstrapping using replay of size: "
              << bootstrap_size << "\n"
              << "# Allocating bootstrap replay ("
              << size_mb(bootstrap_size,
                         dims->point_input_size(),
                         dims->point_output_size())
              << " MB) ..." << std::endl;
    filler.bootstrap = std::make_unique<ExperienceReplay>(
        bootstrap_size, dims->point_input_size(), dims->point_output_size());
    // This is decremented before creating experience, so make + 1.
    filler.bootstrap_move_number = absl::GetFlag(FLAGS_bootstrap_from_move) + 1;
  }
  //
  if (exp_loop == kIsmctsBootstrap) {
    SPIEL_CHECK_TRUE(exp_init == kIsmctsBootstrap);
    reuse.playthroughs->MakeBot((*rnd_gen)());
    std::shared_ptr<const Game> turn_based = game;
    if (game->GetType().dynamics == GameType::Dynamics::kSimultaneous) {
      turn_based = ConvertToTurnBased(*game);
    }
    reuse.playthroughs->GenerateNodes(*turn_based, rnd_gen.get());
  }
  //
  const int snapshot_loop = absl::GetFlag(FLAGS_snapshot_loop);

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------

  std::cout << "# Ready to run the train/eval loop!" << std::endl;
  for (int i = 0; i < 80; ++i) std::cout << '#';
  std::cout << std::endl;
  std::cout << "loop,avg_loss";
  PrintHeaders(metrics);
  std::cout << std::endl;
  //
  for (int loop = 0; loop < num_loops; ++loop) {
    if (snapshot_loop > 0 && (loop+1) % snapshot_loop == 0) {
      std::string save_as = absl::StrCat(snapshot_dir, "/", loop, kModelExt);
      std::cout << "# Saving snapshot of the neural net to "
                << save_as << std::endl;
      SaveNetSnapshot(model, save_as);
    }
    //
    if (loop % exp_loop_new == 0) {
      if (loop == 0) {
        std::cout << "# Initializing experience replay." << std::endl;
        filler.CreateExperiences(exp_init, exp_init_size);
      } else {
        filler.CreateExperiences(exp_loop, exp_update_size);
      }
      // Reset must come after experience creation!
      if (exp_reset_nn) {
        std::cout << "# Resetting net weights with random init." << std::endl;
        model->apply(InitWeights);
      }
      std::cout << "# Resetting optimizer." << std::endl;
      optimizer->state().clear();
      ResetLearningRate(optimizer.get(), learning_rate);
    }

    std::cout << "# Training  ";
    model->train();  // Train mode.
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      experience_replay.SampleBatch(&train_batch, rnd_gen.get());
      cumul_loss += TrainNetwork(model.get(), &device,
                                 optimizer.get(), &train_batch);
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    std::cout << "# Evaluating " << std::endl;
    model->eval();  // Eval mode.
    ComputeMetrics(metrics);
    std::cout << loop;
    // Always print avg loss.
    std::cout << ',' << (train_batches > 0 ? cumul_loss / train_batches : 0.);
    PrintMetrics(metrics);
    std::cout << std::endl;
    DecayLearningRate(optimizer.get(), lr_decay);
  }
  // ---------------------------------------------------------------------------
  if (filler.bootstrap) {
    if (!filler.bootstrap->IsAtBeginning()) {
      // Don't put a CHECK here, because this may happen after hours of training
      // :'-)
      std::cout << "# WARN: bootstrap replay may not be properly filled !!\n";
    }
    std::cout << "# Finished bootstrapped training.\n";
    std::cout << "# Retraining the final network.\n" << std::endl;

    bool bootstrap_reset_nn = absl::GetFlag(FLAGS_bootstrap_reset_nn);
    if (bootstrap_reset_nn) {
      std::cout << "# Resetting net weights with random init." << std::endl;
      model->apply(InitWeights);
    }
    std::cout << "# Resetting optimizer." << std:: endl;
    optimizer->state().clear();
    ResetLearningRate(optimizer.get(), learning_rate);

    for (int loop = num_loops; loop < 2*num_loops; ++loop) {
      if (snapshot_loop > 0 && (loop+1) % snapshot_loop == 0) {
        std::string save_as = absl::StrCat(snapshot_dir, "/", loop, ".model");
        std::cout << "# Saving snapshot of the neural net to "
                  << save_as << std::endl;
        SaveNetSnapshot(model, save_as);
      }
      // Do not make any new data this time.
      std::cout << "# Training  ";
      model->train();  // Train mode.
      double cumul_loss = 0.;
      for (int i = 0; i < train_batches; ++i) {
        filler.bootstrap->SampleBatch(&train_batch, rnd_gen.get());
        cumul_loss += TrainNetwork(model.get(), &device,
                                   optimizer.get(), &train_batch);
        std::cout << '.' << std::flush;
      }
      std::cout << std::endl;

      std::cout << "# Evaluating " << std::endl;
      model->eval();  // Eval mode.
      ComputeMetrics(metrics);
      std::cout << loop;
      // Always print avg loss.
      std::cout << ',' << (train_batches > 0 ? cumul_loss / train_batches : 0.);
      PrintMetrics(metrics);
      std::cout << std::endl;
      DecayLearningRate(optimizer.get(), lr_decay);
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  INIT_EXPERIMENT();
  // Enable float error signals.
  feenableexcept(FE_INVALID);
  open_spiel::papers_with_code::TrainEvalLoop();
}
