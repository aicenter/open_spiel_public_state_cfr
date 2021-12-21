//
// Created by milecdav on 21.12.21.
//

#include "include_libs_ordered.h"
#include "torch_utils.h"

#ifndef OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
#define OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
struct Net : torch::nn::Module {
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(2704, 2048));
    fc2 = register_module("fc2", torch::nn::Linear(2048, 2048));
    fc3 = register_module("fc3", torch::nn::Linear(2048, 2652));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }
};

struct Batch {
  Batch(torch::Tensor data, torch::Tensor target) : data_(data), target_(target) {}
  torch::Tensor data_;
  torch::Tensor target_;
};

struct Data : Batch {
  Data(torch::Tensor data, torch::Tensor target, int batch_size) : Batch(data, target), batch_size_(batch_size) {}
  int batch_size_;
  int batch_index_ = 0;
  Batch SampleBatch() {
    Batch new_batch = Batch(data_[batch_index_ * batch_size_, (batch_index_ + 1) * batch_size_],
                            target_[batch_index_ * batch_size_, (batch_index_ + 1) * batch_size_]);
    batch_index_++;
    return new_batch;
  }

  void Reset() {
    batch_index_ = 0;
  }
};
#endif //OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
