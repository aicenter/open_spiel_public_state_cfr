// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
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

#include "open_spiel/papers_with_code/1906.06412.value_functions/pareto_frontier.h"

namespace open_spiel {
namespace papers_with_code {
namespace {

void TestParetoFrontier() {
  {
    std::vector<std::array<double, 2>> points{
        {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}, {5., 5.},
    };
    std::vector<int> expected_points{4, 3, 2, 1, 0};
    auto actual_points = SortByParetoFrontier(points, 5);
    SPIEL_CHECK_EQ(actual_points, expected_points);
  }

  {
    std::vector<std::array<double, 2>> points{
        {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}, {5., 5.},
    };
    std::vector<int> expected_points{4, 3};
    auto actual_points = SortByParetoFrontier(points, 2);
    SPIEL_CHECK_EQ(actual_points, expected_points);
  }

  {
    std::vector<std::array<double, 2>> points{
        {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}, {5., 5.},
    };
    std::vector<int> expected_points{4, 3, 2, 1, 0};
    auto actual_points = SortByParetoFrontier(points, 10);
    SPIEL_CHECK_EQ(actual_points, expected_points);
  }

  {
    std::vector<std::array<double, 2>> points{
        {2., 1.}, {1., 2.}, {1., 1.}
    };
    std::vector<int> expected_points{1, 0};
    auto actual_points = SortByParetoFrontier(points, 2);
    SPIEL_CHECK_EQ(actual_points, expected_points);
  }

  {
    std::vector<std::array<double, 2>> points{
        {2., 1.}, {1., 2.}, {1., 1.}
    };
    std::vector<int> expected_points{1};
    auto actual_points = SortByParetoFrontier(points, 1);
    SPIEL_CHECK_EQ(actual_points, expected_points);
  }
}

}  // namespace
}  // namespace papers_with_code
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::papers_with_code::TestParetoFrontier();
}

