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
ABSL_FLAG(int, sparse_roots_depth, -1,
          "The depth at which sparse roots should be found."
          " -1 will automatically use some reasonable depth based on the game.");

// -----------------------------------------------------------------------------

#include "absl/random/random.h"
#include "torch/torch.h"

#include "open_spiel/papers_with_code/1906.06412.value_functions/experience_replay.h"
#include "open_spiel/papers_with_code/1906.06412.value_functions/train_eval.h"
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
                          ExperienceReplay* experience_replay,
                          Trunk* trunk,
                          const std::vector<NetContext*>& net_contexts,
                          ortools::SequenceFormLpSpecification* whole_game,
                          const std::vector<int>& eval_iters,
                          std::mt19937& rnd_gen) {

  std::cout << "# Filling experience replay buffer." << std::endl;

  switch (init) {
    case kGenerateDlcfrIterations: {
      int trunk_eval_iterations = *std::max_element(eval_iters.begin(),
                                                    eval_iters.end());

      std::cout << "# Computing reference expls for given trunk iterations.\n";
      GenerateDataDLCfrIterations(
        trunk, net_contexts, experience_replay, trunk_eval_iterations,
        /*monitor_fn*/[&](int trunk_iter) {
          bool should_evaluate =
              std::find(eval_iters.begin(), eval_iters.end(), trunk_iter)
                  != eval_iters.end();

          if (should_evaluate) {
            double expl = ortools::TrunkExploitability(
                whole_game,
                *trunk->iterable_trunk_with_oracle->AveragePolicy());
            std::cout << "# " << trunk_iter << ": "
                      << "expl = " << expl << std::endl;
          }
        },
        rnd_gen,
        absl::GetFlag(FLAGS_shuffle_input),
        absl::GetFlag(FLAGS_shuffle_output));
      break;
    }

    case kGenerateRandomRangesAndSubgameValues: {
      std::cout << "# Generating random trunks to fill experience replay.\n# ";
      for (int i = 0; i < absl::GetFlag(FLAGS_num_trunks); ++i) {
        if (i % 10 == 0) std::cout << '.' << std::flush;
        GenerateDataRandomRanges(trunk, net_contexts, experience_replay,
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
    out.push_back(std::stoi(x));
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
    SpielFatalError("Exhausted pattern match!");
  }
}


void TrainEvalLoop(std::unique_ptr<Trunk> t, int train_batches, int num_loops,
                   int cfr_oracle_iterations, std::string use_bandits_for_cfr,
                   int seed) {

  std::cout << "# Number of public states: " << t->num_leaves << "\n";
  std::cout << "# Number of non-terminal public states: "
            << t->num_non_terminal_leaves << "\n";
  std::cout << "# Public features: " << t->dims->public_features_size << "\n";
  std::cout << "# Hand features: " << t->dims->hand_features_size << "\n";
  std::cout << "# Ranges size: " << t->dims->net_ranges_size << "\n";
  std::cout << "# Point input size: " << t->dims->point_input_size() << "\n";
  std::cout << "# Point output size: " << t->dims->point_output_size() << "\n";
  std::cout << "# Max particles: " << t->dims->max_particles << "\n";
  std::cout << "# Public states stats: \n";
  dlcfr::PrintPublicStatesStats(t->fixable_trunk_with_oracle->public_leaves());
  SPIEL_CHECK_GT(t->num_non_terminal_leaves, 0);  // The trunk is too deep?

  const ExpReplayInitPolicy init_policy =
      GetInitPolicy(absl::GetFlag(FLAGS_data_generation));
  const std::vector<int> eval_iters =
      ItersFromString(absl::GetFlag(FLAGS_trunk_eval_iterations));
  const int num_trunks = init_policy == kGenerateDlcfrIterations
      ? eval_iters.back()
      : absl::GetFlag(FLAGS_num_trunks);
  const int experience_replay_buffer_size =
      t->num_non_terminal_leaves * num_trunks;
  const int batch_size = absl::GetFlag(FLAGS_batch_size) > 0
      ? std::min(absl::GetFlag(FLAGS_batch_size), experience_replay_buffer_size)
      : experience_replay_buffer_size;
  const int roots_depth = GetSparseRootsDepth(*t->game);
  const int no_move_limit = 1000;

  t->oracle_evaluator->num_cfr_iterations = cfr_oracle_iterations;
  torch::manual_seed(seed);
  std::mt19937 rnd_gen(seed);

  // 1. Create the LP spec for the whole game.
  ortools::SequenceFormLpSpecification whole_game(*t->game, "CLP");

  // 2. Create network and optimizer.
  torch::Device device = FindDevice();
  ParticleValueNet model(t->dims.get(), batch_size, ActivationFunction::kRelu);
  model.limit_particle_count = absl::GetFlag(FLAGS_limit_particle_count);
  model.to(device);
  torch::optim::Adam optimizer(model.parameters());

  // 3. Create trunk net evaluator.
  std::cout << "# Batch size: " << batch_size << "\n";
  BatchData train_batch(batch_size,
                        t->dims->point_input_size(),
                        t->dims->point_output_size());
  BatchData eval_batch(1, t->dims->point_input_size(),
                       t->dims->point_output_size());

  auto net_evaluator = std::make_shared<ParticleNetEvaluator>(
      t->hand_info.get(), &model, t->dims.get(), &eval_batch, &device);

  auto [eq_policy, game_value] = ortools::MakeEquilibriumPolicy(&whole_game);
  std::vector<std::unique_ptr<SparseTrunk>> sparse_eq_trunk_with_net;
  sparse_eq_trunk_with_net.push_back(MakeSparseTrunkWithEqSupport(
      eq_policy, t->game, t->infostate_observer, t->public_observer,
      roots_depth, t->trunk_depth,
      net_evaluator, t->terminal_evaluator, use_bandits_for_cfr));
  std::cout << "# Equilibrium sparse trunk:"
            << "\n# - Infostate leaves: "
            << sparse_eq_trunk_with_net.back()->dlcfr->trees()[0]->num_leaves()
            << "\n# - Eval infostates: "
            << sparse_eq_trunk_with_net.back()->eval_infostates.size()
            << "\n# Full trunk infostate leaves: "
            << t->fixable_trunk_with_oracle->trees()[0]->num_leaves() << "\n";

  // The sparse trunk is constructed as replacing the players' equilibrium
  // policies as a chance in the upper game. By constructing the trunk with no
  // move limit, we make an evaluation trunk.
  std::unique_ptr<SparseTrunk> eval_trunk =
      MakeSparseTrunkWithEqSupport(eq_policy, t->game,
                                   t->infostate_observer, t->public_observer,
                                   roots_depth, no_move_limit,
                                   nullptr, t->terminal_evaluator,
                                   use_bandits_for_cfr);
  ortools::SequenceFormLpSpecification eq_fixed_as_chance_lp(
      eval_trunk->dlcfr->trees(), "CLP");

  auto trunk_with_net = std::make_unique<dlcfr::DepthLimitedCFR>(
      t->game, t->trunk_trees, net_evaluator, t->terminal_evaluator,
      t->public_observer,
      MakeBanditVectors(t->trunk_trees, use_bandits_for_cfr));
  auto net_contexts = trunk_with_net->contexts_as<NetContext>();

  // 4. Make experience replay buffer.
  std::cout << "# Allocating experience replay buffer: "
            << experience_replay_buffer_size << " sample points ("
            << experience_replay_buffer_size
               * (t->dims->point_input_size() + t->dims->point_output_size())
            << " floats)" << std::endl;
  ExperienceReplay experience_replay(experience_replay_buffer_size,
                                     t->dims->point_input_size(),
                                     t->dims->point_output_size());
  FillExperienceReplay(init_policy, &experience_replay, t.get(),
                       net_contexts, &whole_game, eval_iters, rnd_gen);

  // 5. The train-eval loop.
  std::cout << "loop,avg_loss,";
  for (int i : eval_iters) {
    std::cout << "expl[" << i << "],";
  }
  std::cout << std::endl;

  for (int loop = 0; loop < num_loops; ++loop) {
    model.train();  // Train mode.
    std::cout << "# Training  ";

    double cumul_loss = 0.;
    torch::Tensor output, loss;
    for (int i = 0; i < train_batches; ++i) {
      // Train using generated data. This data may be shuffled.
      experience_replay.SampleBatch(&train_batch, rnd_gen);
      cumul_loss += TrainNetwork(&model, &device, &optimizer, &train_batch);
      std::cout << '.' << std::flush;
    }
    std::cout << std::endl;

    model.eval();  // Eval mode.
    std::cout << "# Evaluating  " << std::flush;
    std::vector<double> evals_eq = EvaluateNetwork(
        sparse_eq_trunk_with_net, &eq_fixed_as_chance_lp, eval_iters);
    std::cout << std::endl;
//    PrintTrunkStrategies(trunk_with_net.get());

    const double avg_loss = cumul_loss / train_batches;
    std::cout << loop << ',' << avg_loss << ',';
    for (float eval : evals_eq) {
      std::cout << eval << ',';
    }
    std::cout << std::endl;
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
