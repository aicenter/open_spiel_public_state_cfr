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

#include "open_spiel/papers_with_code/1906.06412.value_functions/metrics.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace papers_with_code {

// Approx BR:
//  num_turns = 3 num_cards = 4 br = 0.791667
//  num_turns = 3 num_cards = 5 br = 0.895833
//  num_turns = 3 num_cards = 6 br = 0.921875
//  num_turns = 3 num_cards = 7 br = 0.921875
//  num_turns = 3 num_cards = 8 br = 0.921875
//  num_turns = 3 num_cards = 9 br = 0.921875
//  num_turns = 3 num_cards = 10 br = 0.921875
//
// Full BR:
//  num_turns = 3 num_cards = 4 br = 0.791667
//  num_turns = 3 num_cards = 5 br = 0.883333
//  num_turns = 3 num_cards = 6 br = 0.925
//  num_turns = 3 num_cards = 7 br = 0.947619
//  num_turns = 3 num_cards = 8 br = 0.96131
//  num_turns = 3 num_cards = 9 br = 0.970238
//  num_turns = 3 num_cards = 10 br = 0.976389

void EvaluateUniformStrategy() {
  int total_turns = 3;
  int total_cards = 10;

  std::cout << "Approx BR:\n";
  for (int num_turns = 3; num_turns <= total_turns; ++num_turns) {
    for (int num_cards = 4; num_cards <= total_cards; ++num_cards) {
      auto game  = LoadGame(absl::StrCat(
          "goofspiel(players=2"
                   ",num_turns=", num_turns,
                   ",num_cards=", num_cards,
                   ",imp_info=True"
                   ",points_order=descending)"));
      auto goof_game =
          std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game);

      std::unique_ptr<Bot> bot = MakeUniformRandomBot(Player{0}, /*seed=*/0);
      std::unique_ptr<Metric> br = MakeIigsBrMetric(std::move(bot), goof_game,
                                                    /*approx_response=*/true);
      br->Reset();
      br->Evaluate(std::cout);
      std::cout
      << " num_turns = " << num_turns
      << " num_cards = " << num_cards
      << " br = ";
      br->PrintMetric(std::cout);
      std::cout << std::endl;
    }
  }
    std::cout << "\nFull BR:\n";
  for (int num_turns = 3; num_turns <= total_turns; ++num_turns) {
    for (int num_cards = 4; num_cards <= total_cards; ++num_cards) {
      auto game  = LoadGame(absl::StrCat(
          "goofspiel(players=2"
          ",num_turns=", num_turns,
          ",num_cards=", num_cards,
          ",imp_info=True"
          ",points_order=descending)"));
      auto goof_game =
          std::dynamic_pointer_cast<const goofspiel::GoofspielGame>(game);

      std::unique_ptr<Bot> bot = MakeUniformRandomBot(Player{0}, /*seed=*/0);
      std::unique_ptr<Metric> br = MakeIigsBrMetric(std::move(bot), goof_game,
                                                    /*approx_response=*/false);
      br->Reset();
      br->Evaluate(std::cout);
      std::cout
      << " num_turns = " << num_turns
      << " num_cards = " << num_cards
      << " br = ";
      br->PrintMetric(std::cout);
      std::cout << std::endl;
    }
  }
}

}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::EvaluateUniformStrategy();
}
