//
// Created by milecdav on 21.12.21.
//

#include "include_libs_ordered.h"

#ifndef OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
#define OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
struct Net : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr}, fc5{nullptr}, fc6{nullptr};

  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(2705, 2048));
    fc2 = register_module("fc2", torch::nn::Linear(2048, 2048));
    fc3 = register_module("fc3", torch::nn::Linear(2048, 2048));
    fc4 = register_module("fc4", torch::nn::Linear(2048, 2048));
    fc5 = register_module("fc5", torch::nn::Linear(2048, 2048));
    fc6 = register_module("fc6", torch::nn::Linear(2048, 2652));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    x = torch::relu(fc4->forward(x));
    x = torch::relu(fc5->forward(x));
    x = fc6->forward(x);
    return x;
  }
};
#endif //OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
