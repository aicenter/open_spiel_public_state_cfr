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

ABSL_FLAG(std::string, game_name, "kuhn_poker", "Game to run.");
ABSL_FLAG(int, depth, 3, "Depth of the trunk.");
ABSL_FLAG(std::string, arch, "particle_vf",
          "Which architecture of the value function should be used.");
ABSL_FLAG(int, train_batches, 32,
          "Number of training batches before the evalution is run.");
ABSL_FLAG(int, batch_size, 32,
          "Batch size per train step. If <1, then full replay buffer is used.");
ABSL_FLAG(int, num_loops, 5000, "Number of train-eval loops.");
ABSL_FLAG(int, cfr_oracle_iterations, 100, "Number of oracle iterations.");
ABSL_FLAG(std::string, trunk_eval_iterations, "1,2,5,10,20,50,100,200,500,1000",
          "List of trunk eval iterations.");
ABSL_FLAG(int, num_layers, 3, "Number of hidden layers.");
ABSL_FLAG(int, num_width, 3, "Multiplicative constant of the number "
                             "of neurons per layer.");
ABSL_FLAG(int, num_trunks, 100, "Size of experience replay in terms of trunks");
ABSL_FLAG(int, seed, 0, "Seed.");
ABSL_FLAG(std::string, use_bandits_for_cfr, "RegretMatchingPlus",
          "Which bandit should be used in the trunk.");
ABSL_FLAG(std::string, data_generation, "random", "One of random,dl_cfr");
ABSL_FLAG(double, prob_pure_strat, 0.1, "Params for random generation.");
ABSL_FLAG(double, prob_fully_mixed, 0.05, "Params for random generation.");
ABSL_FLAG(bool, shuffle_input, false,
          "Should experience replay particle data input be shuffled?");
ABSL_FLAG(bool, shuffle_output, false,
          "Should experience replay particle data output be shuffled?");
ABSL_FLAG(int, limit_particle_count, -1,
          "How many particles should be used at most in neural network training?"
          " -1 for all.");
ABSL_FLAG(int, sparse_roots_depth, 0,
          "The depth at which sparse roots should be found."
          " -1 will automatically use some reasonable depth based on the game.");
ABSL_FLAG(double, support_threshold, 1e-5,
          "Pruning threshold for not playing actions from equilibrium, "
          "used for trunk sparsification.");
ABSL_FLAG(bool, prune_chance_histories, false,
          "If true, do not start at chance histories.");

// -----------------------------------------------------------------------------

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/evaluators.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/generate_data.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/net_dl_evaluator.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/sparse_trunk.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/torch_utils.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/trunk.h"


namespace open_spiel {
namespace papers_with_code {

using namespace algorithms;

enum ExpReplayInitPolicy {
  kGenerateDlcfrIterations,
  kGenerateRandomRangesAndSubgameValues,
};

ExpReplayInitPolicy GetInitPolicy(const std::string& s) {
  if (s == "dl_cfr") return kGenerateDlcfrIterations;
  if (s == "random") return kGenerateRandomRangesAndSubgameValues;
  SpielFatalError("Exhausted pattern match: data_generation");
}

void FillExperienceReplay(ExpReplayInitPolicy init,
                          const BasicDims& dims,
                          NetArchitecture arch,
                          ExperienceReplay* experience_replay,
                          Trunk* trunk,
                          const std::vector<NetContext*>& net_contexts,
                          ortools::SequenceFormLpSpecification* whole_game,
                          const std::vector<int>& eval_iters,
                          std::mt19937& rnd_gen) {

  std::cout << "# Filling experience replay buffer." << std::endl;

  switch (init) {
    case kGenerateDlcfrIterations: {
      std::cout << "# Computing reference expls for given trunk iterations.\n";
      std::cout << "# <ref_expl>\n";
      std::cout << "# trunk_iter,expl\n";
      GenerateDataDLCfrIterations(
        trunk, net_contexts, dims, arch, experience_replay,
        absl::GetFlag(FLAGS_num_trunks),
        /*monitor_fn*/[&](int trunk_iter) {
          bool should_evaluate =
              std::find(eval_iters.begin(), eval_iters.end(), trunk_iter)
                  != eval_iters.end();

          if (should_evaluate) {
            double expl = ortools::TrunkExploitability(
                whole_game,
                *trunk->iterable_trunk_with_oracle->AveragePolicy());
            std::cout << "# " << trunk_iter << "," << expl << std::endl;
          }
        },
        rnd_gen,
        absl::GetFlag(FLAGS_shuffle_input),
        absl::GetFlag(FLAGS_shuffle_output));
      std::cout << "# </ref_expl>\n";
      break;
    }

    case kGenerateRandomRangesAndSubgameValues: {
      std::cout << "# Generating random trunks to fill experience replay.\n# ";
      for (int i = 0; i < absl::GetFlag(FLAGS_num_trunks); ++i) {
        if (i % 10 == 0) std::cout << '.' << std::flush;
        GenerateDataRandomRanges(trunk, net_contexts, dims, arch, experience_replay,
                                 absl::GetFlag(FLAGS_prob_pure_strat),
                                 absl::GetFlag(FLAGS_prob_fully_mixed),
                                 rnd_gen,
                                 absl::GetFlag(FLAGS_shuffle_input),
                                 absl::GetFlag(FLAGS_shuffle_output));
      }
      std::cout << std::endl;
      break;
    }

    default:
      SpielFatalError("Exhausted pattern match: data_generation");
  }
}

std::vector<int> ItersFromString(const std::string& s) {
  std::vector<std::string> xs = absl::StrSplit(s, ',');
  std::vector<int> out;
  for (auto& x : xs) {
    if (!x.empty()) out.push_back(std::stoi(x));
  }
  return out;
}

int GetSparseRootsDepth(const Game& game) {
  int cmd_flag = absl::GetFlag(FLAGS_sparse_roots_depth);
  if (cmd_flag >= 0) return cmd_flag;

  const std::string& name = game.GetType().short_name;
  if (name == "kuhn_poker") {
    return 2;
  } else if (name == "leduc_poker") {
    return 2;
  } else if (name == "goospiel") {
    return 1;
  } else {
    SpielFatalError("Exhausted pattern match! No default value "
                    "for sparse roots depth available.");
  }
}

NetArchitecture GetArchitecture(const std::string& arch) {
  if (arch == "particle_vf") {
    return NetArchitecture::kParticle;
  } else  if (arch == "positional_vf") {
    return NetArchitecture::kPositional;
  } else {
    SpielFatalError("Exhausted pattern match! Architecture not recognized.");
  }
}


std::unique_ptr<ValueNet> MakeModel(NetArchitecture arch, BasicDims* dims) {
  switch(arch) {
    case NetArchitecture::kParticle: {
      auto particle_dims = open_spiel::down_cast<ParticleDims*>(dims);
      auto model = std::make_unique<ParticleValueNet>(
          particle_dims,
          absl::GetFlag(FLAGS_num_layers),
          absl::GetFlag(FLAGS_num_width),
          ActivationFunction::kRelu);
      model->limit_particle_count = absl::GetFlag(FLAGS_limit_particle_count);
      return model;
    }
    case NetArchitecture::kPositional: {
      auto positional_dims = open_spiel::down_cast<PositionalDims*>(dims);
      return std::make_unique<PositionalValueNet>(
          positional_dims->point_input_size(),
          positional_dims->point_output_size(),
          positional_dims->point_input_size() * absl::GetFlag(FLAGS_num_width),
          absl::GetFlag(FLAGS_num_layers),
          ActivationFunction::kRelu);
    }
  }
}

std::shared_ptr<NetEvaluator> MakeEvaluator(
    BasicDims* dims, HandInfo* hand_info, ValueNet* model,
    BatchData* eval_batch, torch::Device* device) {
  switch(model->architecture()) {
    case NetArchitecture::kParticle: {
      auto particle_model =
          open_spiel::down_cast<ParticleValueNet*>(model);
      auto particle_dims =
          open_spiel::down_cast<ParticleDims*>(dims);
      return std::make_shared<ParticleNetEvaluator>(
          hand_info, particle_model, particle_dims, eval_batch, device);
    }
    case NetArchitecture::kPositional: {
      auto positional_model =
          open_spiel::down_cast<PositionalValueNet*>(model);
      auto positional_dims =
          open_spiel::down_cast<PositionalDims*>(dims);
      return std::make_shared<PositionalNetEvaluator>(
          hand_info, positional_model, positional_dims, eval_batch, device);
    }
  }
}

void PrintHeaders(const std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  std::cout << "loop,avg_loss";
  for (const std::unique_ptr<Evaluator>& evaluator : evaluators) {
    std::cout << ',';
    evaluator->PrintHeader(std::cout);
  }
  std::cout << std::endl;
}

void PrintEvaluations(int loop, double avg_loss,
                      const std::vector<std::unique_ptr<Evaluator>>& evaluators) {
  std::cout << loop << ',' << avg_loss;
  for (const std::unique_ptr<Evaluator>& evaluator : evaluators) {
    std::cout << ',';
    evaluator->PrintEval(std::cout);
  }
  std::cout << std::endl;
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

void TrainEvalLoop(std::unique_ptr<Trunk> t, int train_batches, int num_loops,
                   int cfr_oracle_iterations, std::string use_bandits_for_cfr,
                   int seed) {
  // Replicable experiments!
  torch::manual_seed(seed);
  std::mt19937 rnd_gen(seed);

  const ExpReplayInitPolicy init_policy =
      GetInitPolicy(absl::GetFlag(FLAGS_data_generation));
  const std::vector<int> eval_iters =
      ItersFromString(absl::GetFlag(FLAGS_trunk_eval_iterations));
  const int num_trunks = absl::GetFlag(FLAGS_num_trunks);
  const int experience_replay_buffer_size =
      t->num_non_terminal_leaves * num_trunks;
  const int batch_size = absl::GetFlag(FLAGS_batch_size) > 0
      ? std::min(absl::GetFlag(FLAGS_batch_size), experience_replay_buffer_size)
      : experience_replay_buffer_size;
  const int roots_depth = GetSparseRootsDepth(*t->game);
  const int no_move_limit = 1000;
  const double support_threshold = absl::GetFlag(FLAGS_support_threshold);
  const bool prune_chance_histories = absl::GetFlag(FLAGS_prune_chance_histories);
  const NetArchitecture arch = GetArchitecture(absl::GetFlag(FLAGS_arch));
  t->oracle_evaluator->num_cfr_iterations = cfr_oracle_iterations;

  // General info about the problem.
  std::cout << "# Number of public states: " << t->num_leaves << "\n";
  std::cout << "# Number of non-terminal public states: "
            << t->num_non_terminal_leaves << "\n";
  std::cout << "# Public states stats: \n";
  dlcfr::PrintPublicStatesStats(t->fixable_trunk_with_oracle->public_leaves());
  SPIEL_CHECK_GT(t->num_non_terminal_leaves, 0);  // The trunk is too deep?

  const std::unique_ptr<BasicDims> dims = DeduceDims(*t, arch);
  std::cout << "# Public features: " << dims->public_features_size << "\n";
  std::cout << "# Hand features: " << dims->hand_features_size << "\n";
  std::cout << "# Ranges size: " << dims->net_ranges_size << "\n";
  std::cout << "# Point input size: " << dims->point_input_size() << "\n";
  std::cout << "# Point output size: " << dims->point_output_size() << "\n";
//  std::cout << "# Max particles: " << t->dims->max_particles << "\n";


  // 1. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*t->game, "CLP");

  // 2. Create network and optimizer.
  torch::Device device = FindDevice();
  std::unique_ptr<ValueNet> model = MakeModel(arch, dims.get());
  model->to(device);
  torch::optim::Adam optimizer(model->parameters());

  // 3. Create a value function associated to the trunk.
  std::cout << "# Batch size: " << batch_size << "\n";
  // Train on multiple public states at once.
  BatchData train_batch(batch_size,
                        dims->point_input_size(), dims->point_output_size());
  // Evaluate a single public state.
  // TODO: Maybe extend this to parallel evaluation?
  BatchData eval_batch(1, dims->point_input_size(), dims->point_output_size());
  // Use eval batch only for the net evaluator.
  std::shared_ptr<NetEvaluator> net_evaluator =  // TODO: rename to VF
      MakeEvaluator(dims.get(), t->hand_info.get(), model.get(),
                    &eval_batch, &device);
  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      t->game, t->trunk_trees, net_evaluator, t->terminal_evaluator,
      t->public_observer,
      MakeBanditVectors(t->trunk_trees, use_bandits_for_cfr));
  auto net_contexts = trunk_with_net->contexts_as<NetContext>();

  // 4. Make experience replay buffer.
  std::cout << "# Allocating experience replay buffer: "
            << experience_replay_buffer_size << " sample points ("
            << experience_replay_buffer_size
               * (dims->point_input_size() + dims->point_output_size())
            << " floats)" << std::endl;
  ExperienceReplay experience_replay(experience_replay_buffer_size,
                                     dims->point_input_size(),
                                     dims->point_output_size());
  FillExperienceReplay(init_policy, *dims, arch, &experience_replay, t.get(),
                       net_contexts, &whole_game, eval_iters, rnd_gen);

  // 5. Create training evaluators.
  std::vector<std::unique_ptr<Evaluator>> evaluators;
  if (!eval_iters.empty()) {
    evaluators.push_back(
        MakeFullTrunkEvaluator(eval_iters, trunk_with_net.get(), &whole_game));
  }

  // 6. The train-eval loop.
  PrintHeaders(evaluators);
  for (int loop = 0; loop < num_loops; ++loop) {
    std::cout << "# Training  ";
    model->train();  // Train mode.
    double cumul_loss = 0.;
    for (int i = 0; i < train_batches; ++i) {
      experience_replay.SampleBatch(&train_batch, rnd_gen);
      cumul_loss += TrainNetwork(model.get(), &device,
                                 &optimizer, &train_batch);
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    std::cout << "# Evaluating " << std::endl;
    model->eval();  // Eval mode.
    EvaluateNetwork(evaluators);
    PrintEvaluations(loop, cumul_loss / train_batches, evaluators);
//    PrintTrunkStrategies(trunk_with_net.get());
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel::papers_with_code;
  INIT_EXPERIMENT();

  TrainEvalLoop(
      MakeTrunk(absl::GetFlag(FLAGS_game_name), absl::GetFlag(FLAGS_depth),
                absl::GetFlag(FLAGS_use_bandits_for_cfr)),
      absl::GetFlag(FLAGS_train_batches),
      absl::GetFlag(FLAGS_num_loops),
      absl::GetFlag(FLAGS_cfr_oracle_iterations),
      absl::GetFlag(FLAGS_use_bandits_for_cfr),
      absl::GetFlag(FLAGS_seed));
}
