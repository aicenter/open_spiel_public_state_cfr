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
ABSL_FLAG(std::string, use_bandits_for_cfr, "RegretMatchingPlus",
          "Which bandit should be used in the trunk.");

// -- Data generation --
ABSL_FLAG(int, depth, 3, "Depth of the trunk.");
ABSL_FLAG(double, prob_pure_strat, 0.1, "Params for random generation.");
ABSL_FLAG(double, prob_fully_mixed, 0.05, "Params for random generation.");
ABSL_FLAG(bool, shuffle_input_output, false,
          "Should parview inputs/outputs be shuffled?");
ABSL_FLAG(int, replay_size, 100,
          "Size of experience replay in terms of public states.");
ABSL_FLAG(int, cfr_oracle_iterations, 100, "Number of oracle iterations.");
ABSL_FLAG(int, sparse_particles, 1000,
          "Number of particles to use for sparse pbs generation");
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
ABSL_FLAG(int, cfr_iterations, 100,
          "Number of iterations with value function (used in bootstrapping).");
ABSL_FLAG(int, max_move_ahead_limit, 1, "Size of the lookahead tree.");
ABSL_FLAG(std::string, save_values_policy, "average",
          "What cf. values should be saved after solving the subgame: "
          " one of {current,average}.");

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
ABSL_FLAG(int, snapshot_loop, -1,
          "When should NN weights be saved to snapshot/ dir? -1 for never.");
ABSL_FLAG(std::string, snapshot_dir, "snapshots/",
          "Directory to store snapshots of NN weights.");

// -- Network --
ABSL_FLAG(std::string, arch, "particle_vf",
          "Which architecture of the value function should be used.");
ABSL_FLAG(int, num_layers, 3, "Number of hidden layers.");
ABSL_FLAG(int, num_width, 3, "Multiplicative constant of the number of neurons "
                             "per layer compared to the input size.");
ABSL_FLAG(int, num_inputs_regression, 128,
          "Size of the regression input for particle VF. -1 means it will be"
          "the same as max_parviews");

// -- Metrics --
// FullTrunkExplMetric
ABSL_FLAG(std::string, trunk_expl_iterations, "",
          "Evaluate trunk exploitability for each trunk iteration.");
// ReplayVisitsMetric
ABSL_FLAG(int, replay_visits_window, -1,
          "Track the average visit count over a past window "
          "behind the head in experience replay");

// -----------------------------------------------------------------------------

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/metrics.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/snapshot.h"


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
  optimizer->zero_grad();
  torch::Tensor output = model->forward(data);
  torch::Tensor loss = torch::mse_loss(output, target);
  SPIEL_CHECK_FALSE(std::isnan(loss.template item<float>()));
  loss.backward();
  optimizer->step();
  return loss.item().to<double>();
}

void DecayLearningRate(torch::optim::Optimizer* optimizer, double lr_decay) {
  for (auto &group : optimizer->param_groups()) {
    if(group.has_options()) {
      auto &options = static_cast<torch::optim::SGDOptions &>(group.options());
      options.lr(options.lr() * lr_decay);
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

std::unique_ptr<torch::optim::Optimizer> MakeOptimizer(ValueNet* model) {
  std::string choice = absl::GetFlag(FLAGS_optimizer);
  if (choice == "sgd") {
    return std::make_unique<torch::optim::SGD>(
        model->parameters(),
        torch::optim::SGDOptions(absl::GetFlag(FLAGS_learning_rate)));
  }
  if (choice == "adam") {
    return std::make_unique<torch::optim::Adam>(
        model->parameters(),
        torch::optim::AdamOptions(absl::GetFlag(FLAGS_learning_rate)));
  }
  SpielFatalError("Unknown optimizer!");
}

float size_mb(int replay_size, int input_size, int output_size) {
  return replay_size * (input_size + output_size) * 4 / 1024. / 1024.;
}

void TrainEvalLoop() {
  // Create all the data structures needed for the training-evaluation loop.
  // There is quite a lot of them.

  // Replicable experiments FTW!
  std::cout << "# Seeding experiment ..." << std::endl;
  const int seed = absl::GetFlag(FLAGS_seed);
  torch::manual_seed(seed);
  // All source of randomness need to plumb this through.
  std::mt19937 rnd_gen(seed);
  //
  std::cout << "# Making strategy randomizer ..." << std::endl;
  StrategyRandomizer randomizer;
  randomizer.rnd_gen = &rnd_gen;
  randomizer.prob_pure_strat = absl::GetFlag(FLAGS_prob_pure_strat);
  randomizer.prob_fully_mixed = absl::GetFlag(FLAGS_prob_fully_mixed);
  //
  std::cout << "# Making subgame subgame_factory ..." << std::endl;
  SubgameFactory subgame_factory;
  auto game = subgame_factory.game     = LoadGame(absl::GetFlag(FLAGS_game_name));
  subgame_factory.infostate_observer   = game->MakeObserver(kInfoStateObsType, {});
  subgame_factory.public_observer      = game->MakeObserver(kPublicStateObsType, {});
  subgame_factory.hand_observer        = game->MakeObserver(kHandHistoryObsType, {});
  subgame_factory.max_move_ahead_limit = absl::GetFlag(FLAGS_max_move_ahead_limit);
  subgame_factory.max_trunk_depth      = absl::GetFlag(FLAGS_depth);
  subgame_factory.max_particles        = absl::GetFlag(FLAGS_max_particles);
  //
  std::cout << "# Making oracle evaluator ..." << std::endl;
  const SaveValuesPolicy save_values_policy =
      GetSaveValuesPolicy(absl::GetFlag(FLAGS_save_values_policy));
  auto terminal_evaluator = std::make_shared<TerminalEvaluator>();
  auto oracle = std::make_shared<CFREvaluator>(
      subgame_factory.game, /*full_subgame_depth=*/1000,
      /*no_leaf_evaluator=*/nullptr, terminal_evaluator,
      subgame_factory.public_observer, subgame_factory.infostate_observer);
  oracle->bandit_name = kDefaultDlCfrBandit;
  oracle->num_cfr_iterations = absl::GetFlag(FLAGS_cfr_oracle_iterations);
  oracle->save_values_policy = save_values_policy;
  //
  std::cout << "# Init empty reusable structures ..." << std::endl;
  ReusableStructures reuse(&subgame_factory,
                           /*solver_factory=*/nullptr, // Supplied later.
                           oracle);
  //
  const NetArchitecture arch = GetArchitecture(absl::GetFlag(FLAGS_arch));
  std::cout << "# Deducing dimensions ..." << std::endl;
  std::shared_ptr<BasicDims> dims = DeduceBasicDims(
      arch, *game, subgame_factory.public_observer, subgame_factory.hand_observer);
  std::cout << "# Public features: " << dims->public_features_size << std::endl;
  std::cout << "# Hand features: " << dims->hand_features_size << std::endl;
  //
  std::cout << "# Deducing dimensions for specific VFs ..." << std::endl;
  if (subgame_factory.max_particles >= 1) {  // Static setting.
    auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims.get());
    particle_dims->max_parviews = subgame_factory.max_particles * 2;
    std::cout << "# Particle VF (set statically): "
                 "max_particles = " << subgame_factory.max_particles << ' ' <<
              "max_parviews = " << particle_dims->max_parviews << std::endl;
  }
  std::shared_ptr<HandInfo> hand_info;
  if (arch == NetArchitecture::kPositional || subgame_factory.max_particles == -1) {
    std::cout << "# Finding all hands in the game (may take a while) ..."
              << std::endl;
    PublicStatesInGame* all_states = reuse.GetAllPublicStates();
    const std::vector<PublicState>& public_states = all_states->public_states;
    std::cout << "# Number of public states: "
              << all_states->public_states.size() << std::endl;
    hand_info = MakeHandInfo(*game, subgame_factory.hand_observer, public_states);
    // Hand info can be computed only based on the trunk, or can be provided
    // in a domain-dependent manner in games where it is possible thanks to
    // the structure of the game (Poker / Liar's dice)
//    Subgame* trunk_states = reuse.GetFixableTrunkWithOracle();
//    const std::vector<PublicState>& public_states =
//        trunk_states->public_states();
//    std::cout << "# Number of public states: "
//              << trunk_states->public_states().size() << std::endl;
//    hand_info = MakeHandInfo(*game, subgame_factory.hand_observer,
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
      subgame_factory.max_particles = LargestPublicState(public_states);
      particle_dims->max_parviews = hand_info->num_hands();
      std::cout << "# Particle VF (derived automatically): "
                   "max_particles = " << subgame_factory.max_particles << ' ' <<
                "max_parviews = " << particle_dims->max_parviews << "\n";
    }
  }
  std::cout << "# Point input size: " << dims->point_input_size() << "\n";
  std::cout << "# Point output size: " << dims->point_output_size() << "\n";
  //
  std::cout << "# Using device: " << absl::GetFlag(FLAGS_device) << "\n";
  torch::Device device(absl::GetFlag(FLAGS_device));
  //
  int num_inputs_regression = absl::GetFlag(FLAGS_num_inputs_regression);;
  if (arch == NetArchitecture::kParticle) {
    if(num_inputs_regression == -1) {
      auto particle_dims = open_spiel::down_cast <ParticleDims*>(dims.get());
      num_inputs_regression = particle_dims->max_parviews;
    }
    std::cout << "# Size of the input for regression component: "
              << num_inputs_regression << "\n";
  }
  //
  std::cout << "# Creating model ..." << std::endl;
  std::shared_ptr<ValueNet> model = MakeModel(arch, dims.get(),
                                              absl::GetFlag(FLAGS_num_layers),
                                              absl::GetFlag(FLAGS_num_width),
                                              num_inputs_regression);
  model->to(device);
  //
  std::cout << "# Creating optimizer ..." << std::endl;
  std::unique_ptr<torch::optim::Optimizer> optimizer =
      MakeOptimizer(model.get());
  const double lr_decay = absl::GetFlag(FLAGS_lr_decay);
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
  std::cout << "# Making batch train/eval data ..." << std::endl;
  const int batch_size = absl::GetFlag(FLAGS_batch_size) > 0
                       ? std::min(absl::GetFlag(FLAGS_batch_size),
                                  experience_replay.size())
                       : experience_replay.size();
  BatchData train_batch(batch_size,
                        dims->point_input_size(),
                        dims->point_output_size());
  auto eval_batch = std::make_shared<BatchData>(1,
                                                dims->point_input_size(),
                                                dims->point_output_size());
  //
  SolverFactory solver_factory;
  std::cout << "# Making net evaluator ..." << std::endl;
  solver_factory.cfr_iterations = absl::GetFlag(FLAGS_cfr_iterations);
  solver_factory.use_bandits_for_cfr  = absl::GetFlag(FLAGS_use_bandits_for_cfr);
  solver_factory.save_values_policy   = save_values_policy;
  solver_factory.terminal_evaluator   = terminal_evaluator;
  solver_factory.leaf_evaluator = MakeNetEvaluator(
      dims, model, eval_batch, device,
      hand_info,  // May be nullptr for particle VF.
      subgame_factory.hand_observer);
  reuse.solver_factory = &solver_factory;
  //
  std::cout << "# Making replay filler ..." << std::endl;
  ReplayFiller filler;
  filler.replay     = &experience_replay;
  filler.subgame_factory = &subgame_factory;
  filler.solver_factory  = &solver_factory;
  filler.dims       = dims.get();
  filler.randomizer = &randomizer;
  filler.reuse      = &reuse;
  filler.arch       = arch;
  filler.shuffle_input_output = absl::GetFlag(FLAGS_shuffle_input_output);
  filler.sparse_particles     = absl::GetFlag(FLAGS_sparse_particles);
  filler.sparse_epsilon       = absl::GetFlag(FLAGS_sparse_epsilon);
  filler.eval_iters =
      ItersFromString(absl::GetFlag(FLAGS_trunk_expl_iterations));
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
  if (exp_loop == kBootstrap) {
    SPIEL_CHECK_EQ(exp_init, kBootstrap);
    // First loop is not "regenerated" -- it is initialized.
    const int num_regenerations = num_loops / exp_loop_new;
    // Skips root (move_number=0) and terminals (move_number=max):
    // - computing root is unnecessary for bootstrapping.
    // - terminals can be initialized using a custom exp_init.
    const int bootstrap_depths = subgame_factory.game->MaxMoveNumber() - 2;
    std::cout << "# Num regeneration steps: " << num_regenerations << "\n";
    std::cout << "# Bootstrap depths: " << bootstrap_depths << "\n";
    SPIEL_CHECK_EQ(bootstrap_depths, num_regenerations);

    // Bootstrap replay must hold all created experiences,
    // i.e. from initialization as well as from regeneration.
    const int bootstrap_size = exp_update_size * num_regenerations;

    std::cout << "# Bootstrap size: " << bootstrap_size << "\n";
    std::cout << "# Will be bootstrapping using replay of size: "
              << bootstrap_size << "\n"
              << "# Allocating bootstrap replay ("
              << size_mb(bootstrap_size,
                         dims->point_input_size(),
                         dims->point_output_size())
              << " MB) ..." << std::endl;
    filler.bootstrap = std::make_unique<ExperienceReplay>(
        bootstrap_size, dims->point_input_size(), dims->point_output_size());
    // This is decremented before creating experience, so is not at terminals,
    // but just above them.
    filler.bootstrap_move_number = subgame_factory.game->MaxMoveNumber();
  }
  //
  const int snapshot_loop = absl::GetFlag(FLAGS_snapshot_loop);
  const std::string snapshot_dir = absl::GetFlag(FLAGS_snapshot_dir);
  // ---------------------------------------------------------------------------
  std::cout << "# Ready to run the train/eval loop!" << std::endl;
  for (int i = 0; i < 80; ++i) std::cout << '#';
  std::cout << std::endl;
  std::cout << "loop,avg_loss";
  PrintHeaders(metrics);
  std::cout << std::endl;
  //
  for (int loop = 0; loop < num_loops; ++loop) {
    if (loop % exp_loop_new == 0) {
      if (loop == 0) {
        std::cout << "# Initializing experience replay." << std::endl;
        filler.CreateExperiences(exp_init, exp_init_size);
      } else {
        filler.CreateExperiences(exp_loop, exp_update_size);
      }

      if (exp_reset_nn) {
        std::cout << "# Resetting net weights with random init." << std::endl;
        model->apply(InitWeights);
      }
    }
    if (snapshot_loop > 0 && loop % snapshot_loop == 0) {
      std::string save_as = absl::StrCat(snapshot_dir, loop, ".model");
      std::cout << "# Saving snapshot of the neural net to "
                << save_as << std::endl;
      SaveNetSnapshot(model, save_as);
    }

    std::cout << "# Training  ";
    model->train();  // Train mode.
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      experience_replay.SampleBatch(&train_batch, rnd_gen);
      cumul_loss += TrainNetwork(model.get(), &device,
                                 optimizer.get(), &train_batch);
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    std::cout << "# Evaluating " << std::endl;
    model->eval();  // Eval mode.
    ComputeMetrics(metrics);
    // Always print avg loss.
    std::cout << loop << ',' << cumul_loss / train_batches;
    PrintMetrics(metrics);
    std::cout << std::endl;
    //    DecayLearningRate(optimizer.get(), lr_decay);
  }
  // ---------------------------------------------------------------------------
  if (filler.bootstrap) {
    if (!filler.bootstrap->IsAtBeginning()) {
      // Don't put a CHECK here, because this may happen after hours of training
      // :'-)
      std::cout << "# WARN: bootstrap replay may not be properly filled.\n";
    }
    if (filler.bootstrap_move_number != 1) {
      std::cout << "# WARN: bootstrap depth != 1 -- bootstrapping may have "
                   "skipped over some depths. (was " << filler.bootstrap_move_number
                << ")\n";
    }
    std::cout << "# Finished bootstrapped training.\n";
    std::cout << "# Retraining the final network.\n" << std::endl;

    bool bootstrap_reset_nn = absl::GetFlag(FLAGS_bootstrap_reset_nn);
    if (bootstrap_reset_nn) {
      std::cout << "# Resetting net weights with random init." << std::endl;
      model->apply(InitWeights);
    }

    for (int loop = num_loops; loop < 2*num_loops; ++loop) {
      // Do not make any new data this time.
      std::cout << "# Training  ";
      model->train();  // Train mode.
      double cumul_loss = 0.;
      for (int i = 0; i < train_batches; ++i) {
        filler.bootstrap->SampleBatch(&train_batch, rnd_gen);
        cumul_loss += TrainNetwork(model.get(), &device,
                                   optimizer.get(), &train_batch);
        std::cout << '.' << std::flush;
      }
      std::cout << std::endl;

      std::cout << "# Evaluating " << std::endl;
      model->eval();  // Eval mode.
      ComputeMetrics(metrics);
      // Always print avg loss.
      std::cout << loop << ',' << cumul_loss / train_batches;
      PrintMetrics(metrics);
      std::cout << std::endl;
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  INIT_EXPERIMENT();
  open_spiel::papers_with_code::TrainEvalLoop();
}
