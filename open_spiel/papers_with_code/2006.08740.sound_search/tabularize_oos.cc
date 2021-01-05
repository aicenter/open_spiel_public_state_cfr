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

#include <string>
#include <utility>
#include <memory>

#include "absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/flags/usage.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/algorithms/oos.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/random.h"

ABSL_FLAG(std::string, game, "coordinated_mp",
          "Game to run. One of: coordinated_mp, kuhn_poker");
ABSL_FLAG(int, total_iterations, 100000,
          "Number of total iterations to make");
ABSL_FLAG(double, biasing, 0.1,
          "Target biasing -- parameter delta");
ABSL_FLAG(double, exploration, 0.6,
          "Exploration -- parameter epsilon");
ABSL_FLAG(int, seed, 0,
          "Seed for the pseudo-RNG");
ABSL_FLAG(bool, csv_header, 0,
          "Print CSV header and terminate program");
ABSL_FLAG(bool, print_stats, false,
          "Print algorithm stats to stderr");

namespace open_spiel {
namespace papers_with_code {
namespace {

using namespace open_spiel::algorithms;

void InitializeInfosets(
    const std::shared_ptr<const Game>& game, OOSInfoStateValuesTable& table) {

  std::function<void(State*)> walk = [&](State* s) {
      auto actions = s->LegalActions();
      if (s->IsPlayerNode()) {
        std::string info_state = s->InformationStateString();
        if (table.find(info_state) == table.end()) {
          table[info_state] = CFRInfoStateValues(actions);
        }
      }

      for (auto action : actions) {
        auto next = s->Child(action);
        walk(next.get());
      }
  };

  auto root_state = game->NewInitialState();
  walk(root_state.get());
}

void InitializeRegrets(
    const std::string& game_name, OOSInfoStateValuesTable& t,
    int target_table, const std::vector<std::string>& target_infosets) {

  // Regret constant from the paper.
  const double mu = 500.;

  if (game_name == "coordinated_mp") {
    if (target_table == 0) {
      t[target_infosets[0]].cumulative_regrets = {mu, mu};
      t[target_infosets[1]].cumulative_regrets = {mu, mu};
    } else if (target_table == 1) {
      t[target_infosets[0]].cumulative_regrets = {0, mu};
      t[target_infosets[1]].cumulative_regrets = {mu, 0};
    } else {
      SpielFatalError("Bad target_table number.");
    }
  } else if (game_name == "kuhn_poker") {
    // From Wikipedia:
    //
    // The first player should expect to lose at a rate of −1/18. per hand.
    //
    // Kuhn demonstrated there are infinitely many equilibrium strategies
    // for the first player, forming a continuum governed by a single
    // parameter. In one possible formulation, player one freely chooses
    // the probability α ∈ [ 0 , 1 / 3 ] with which he will bet when having
    // a Jack (otherwise he checks; if the other player bets, he should always
    // fold).
    //
    // When having a King, he should bet with the probability of 3 α (otherwise
    // he checks; if the other player bets, he should always call).
    //
    // He should always check when having a Queen, and if the other player bets
    // after this check, he should call with the probability of α + 1 / 3.
    //
    // The second player has a single equilibrium strategy:
    // - Always betting or calling when having a King;
    // - when having a Queen, checking if possible, otherwise calling with
    //   the probability of 1/3;
    // - when having a Jack, never calling and betting with
    //   the probability of 1/3.

    double a;  // alpha from the paper
    if (target_table == 0) a = 0;
    else if (target_table == 1) a = 1 / 6.;
    else if (target_table == 2) a = 1 / 3.;
    else SpielFatalError("Bad target_table number.");

    // 0 = Jack, 1 = Queen, 2 = King
    // p = Pass (equilvalent to fold/call), b = Bet
    // First action is Pass, second is Bet.
    // @formatter:off
    // FIXME: this is probably flipped! pass vs bet
    t["0"].cumulative_regrets   = { mu * (a),          mu * (1 - a)};
    t["0pb"].cumulative_regrets = { mu * (0),          mu * (1)};
    t["1"].cumulative_regrets   = { mu * (0),          mu * (1)};
    t["1pb"].cumulative_regrets = { mu * (a + 1 / 3.), mu * (2 / 3. - a)};
    t["2"].cumulative_regrets   = { mu * (3 * a),      mu * (1 - 3 * a)};
    t["2pb"].cumulative_regrets = { mu * (1),          mu * (0)};
    t["0b"].cumulative_regrets  = { mu * (0),          mu * (1)};
    t["0p"].cumulative_regrets  = { mu * (1 / 3.),     mu * (2 / 3.)};
    t["1b"].cumulative_regrets  = { mu * (1 / 3.),     mu * (2 / 3.)};
    t["1p"].cumulative_regrets  = { mu * (0),          mu * (1)};
    t["2b"].cumulative_regrets  = { mu * (1),          mu * (0)};
    t["2p"].cumulative_regrets  = { mu * (1),          mu * (0)};
    // @formatter:on
  } else {
    SpielFatalError("Bad game name.");
  }
}

using InfosetsWithNumActions = std::vector<std::pair<std::string, int>>;

void PrintPolicies(const std::vector<std::unique_ptr<Policy>>& policies,
                   const InfosetsWithNumActions& infosets) {
  for (const auto& policy : policies) {
    for (const auto &[infoset, num_actions] : infosets) {
      ActionsAndProbs probs = policy->GetStatePolicy(infoset);
      for (const auto&[action, prob] : probs) {
        std::cout << "," << prob;
      }
    }
  }
}

InfosetsWithNumActions GetInfosetsWithNumActions(
    const std::shared_ptr<const Game>& game, Player p) {
  InfosetsWithNumActions infosets;

  std::function<void(State*)> walk = [&](State* s) {
      auto actions = s->LegalActions();
      if (s->CurrentPlayer() == p) {
        std::string info_state = s->InformationStateString();
        infosets.push_back({info_state, s->LegalActions().size()});
      }

      for (auto action : actions) {
        auto next = s->Child(action);
        walk(next.get());
      }
  };

  auto root_state = game->NewInitialState();
  walk(root_state.get());

  std::sort(infosets.begin(), infosets.end());
  auto ip = std::unique(infosets.begin(), infosets.end());
  infosets.resize(std::distance(infosets.begin(), ip));
  return infosets;
}

void PrintHeader(const std::string& game_name) {
  const auto game = LoadGame(game_name);
  InfosetsWithNumActions infosets;
  int num_algs;
  if (game_name == "coordinated_mp") {
    infosets = GetInfosetsWithNumActions(game, Player(1));
    num_algs = 3;
  }
  if (game_name == "kuhn_poker") {
    infosets = GetInfosetsWithNumActions(game, Player(1));
    num_algs = 4;
  }

  std::cout << "seed,iter";
  for (int i = 0; i < num_algs; ++i) {
    for (const auto&[infoset, num_actions] : infosets) {
      for (int j = 0; j < num_actions; ++j) {
        std::cout << "," << i << "::" << infoset << "::" << j;
      }
    }
  }
  std::cout << std::endl;
}

void RunExperiment(
    const std::string& game_name, int total_iterations,
    double target_biasing, double exploration, int seed) {
  std::optional<Action> None = std::nullopt; // Imitate Python way of writing.

  // --- 1. Get the game.
  const auto game = LoadGame(game_name);

  // --- 2. Specify the biasing targets for each game. 
  InfosetsWithNumActions infosets;
  std::vector<std::string> target_infosets;
  std::vector<ActionObservationHistory> target_aohs;

  if (game_name == "coordinated_mp") {
    infosets = GetInfosetsWithNumActions(game, Player(1));
    target_infosets = {"T", "B"};
    target_aohs = std::vector<ActionObservationHistory>{
        ActionObservationHistory(1, {{None, "."}, {None, "T"}}),
        ActionObservationHistory(1, {{None, "."}, {None, "B"}})
    };
  }
  if (game_name == "kuhn_poker") {
    infosets = GetInfosetsWithNumActions(game, Player(0));
    target_infosets = {"0", "1", "2"};
    target_aohs = std::vector<ActionObservationHistory>{
        ActionObservationHistory(0, {{None, "011"}, {None, "011"}}),
        ActionObservationHistory(0, {{None, "111"}, {None, "111"}}),
        ActionObservationHistory(0, {{None, "211"}, {None, "211"}}),
    };
  }

  // --- 3. Initialize the (un)biased variants of the OOS algorithm.
  const int num_algs = target_aohs.size() + 1;  // +1 for the non-biased run.
  std::vector<std::unique_ptr<OOSAlgorithm>> algs;
  algs.reserve(num_algs);
  std::vector<std::unique_ptr<Policy>> policies;
  policies.reserve(num_algs);

  for (int i = 0; i < num_algs; ++i) {
    auto table = std::make_unique<OOSInfoStateValuesTable>();
    InitializeInfosets(game, *table);
    if (i < num_algs - 1)
      InitializeRegrets(game_name, *table, i, target_infosets);

    auto explore =
        std::make_unique<ExplorativeSamplingPolicy>(*table, exploration);
    auto target = std::make_unique<TargetedPolicy>(game, *table, exploration);
    auto alg = std::make_unique<OOSAlgorithm>(
        game,
        std::move(table),
        std::make_unique<RandomMT>(/*seed=*/seed),
        std::move(explore),
        std::move(target),
        /*default_policy=*/std::make_shared<UniformPolicy>(),
        i == num_algs - 1 ? 0. : target_biasing);
    auto policy = alg->AveragePolicy();
    algs.push_back(std::move(alg));
    policies.push_back(std::move(policy));
  }
  auto& alg_no_bias = algs[num_algs - 1];

  // --- 4. Run for a specified number of iterations.

  // We use a "logarithmic" spacing of printed outputs.
  const long n = 50;  // (approx) number of prints per factor of 10.
  long counter = -1;
  unsigned long last_print = 0;
  unsigned long i = 0;

  for (; i < total_iterations; ++i) {
    for (int j = 0; j < num_algs - 1; ++j) {
      algs[j]->RunTargetedIterations(target_aohs[j], 1);
    }
    alg_no_bias->RunUnbiasedIterations(1);

    int counter_new = int(log10((double) i) * n);
    if (counter_new > counter) {
      counter = counter_new;
      last_print = i;

      // Skip some initial iterations, they are not interesting
      // and these logs occupy too much space.
      if (i >= 100) {
        std::cout << seed << "," << i;
        PrintPolicies(policies, infosets);
        std::cout << std::endl;
      }
    }
  }

  if (last_print != i) {
    std::cout << seed << "," << i;
    PrintPolicies(policies, infosets);
    std::cout << std::endl;
  }

  // --- 5. Optionally print algorithm stats.
  if (!absl::GetFlag(FLAGS_print_stats))
    return;

  for (int j = 0; j < num_algs; ++j) {
    std::cerr << "Alg " << j << " stats (target ";
    if (j == num_algs - 1) {
      std::cerr << "Nothing";
    } else {
      std::cerr << target_aohs[j];
    }
    std::cerr << "):\n";
    std::cerr << algs[j]->GetStats();
    std::cerr << "----" << std::endl;
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel


int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "Experiment runner for the paper "
      "Sound Search in Imperfect Information Games");
  absl::ParseCommandLine(argc, argv);

  const std::string game = absl::GetFlag(FLAGS_game);
  if (game != "coordinated_mp" && game != "kuhn_poker") {
    open_spiel::SpielFatalError("Unrecognized game.");
  }

  if (absl::GetFlag(FLAGS_csv_header)) {
    open_spiel::papers_with_code::PrintHeader(game);
    return 0;
  }

  open_spiel::papers_with_code::RunExperiment(
      game,
      absl::GetFlag(FLAGS_total_iterations),
      absl::GetFlag(FLAGS_biasing),
      absl::GetFlag(FLAGS_exploration),
      absl::GetFlag(FLAGS_seed));
  return 0;
}
