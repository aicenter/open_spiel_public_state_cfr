//
// Created by milecdav on 21.12.21.
//

#include "include_libs_ordered.h"

#ifndef OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
#define OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
struct Net : torch::nn::Module {
  std::vector<torch::nn::Linear> layers_;

  int input_size_;
  int output_size_;
  int hidden_layers_;
  int deck_size_;
  int possible_hands_;

  Net(int deck_size, int possible_hands, int layer_size, int hidden_layers) {
    possible_hands_ = possible_hands;
    deck_size_ = deck_size;
    input_size_ = deck_size + 2 * possible_hands + 1;
    output_size_ = 2 * possible_hands;
    hidden_layers_ = hidden_layers;
    layers_.reserve(hidden_layers_ + 2);
    layers_.emplace_back(register_module("fc_in", torch::nn::Linear(input_size_, layer_size)));
    for (int i = 0; i < hidden_layers_; i++) {
      layers_.emplace_back(register_module("fc" + std::to_string(i), torch::nn::Linear(layer_size, layer_size)));
    }
    layers_.emplace_back(register_module("fc_out", torch::nn::Linear(layer_size, output_size_)));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor
        beliefs = x.index({torch::indexing::Slice(),
                           torch::indexing::Slice(deck_size_, possible_hands_ * 2 + deck_size_)});
    for (int i = 0; i < layers_.size() - 1; i++) {
      x = torch::leaky_relu(layers_[i]->forward(x));
    }
    x = layers_.back()->forward(x);

    torch::Tensor numer = (x * beliefs).sum(/*dim=*/1);
    torch::Tensor denum = (beliefs * beliefs).sum(/*dim=*/1);
    torch::Tensor proj_error = torch::zeros({x.size(0)});
    proj_error.index_put_({denum != 0}, numer.index({denum != 0}).div(denum.index({denum != 0})));
    torch::Tensor expanded_proj_error = proj_error.expand({possible_hands_ * 2, -1}).permute({1, 0});
    x = x - expanded_proj_error * beliefs;

    SPIEL_DCHECK_TRUE(torch::isfinite(x).all().item<bool>());
    return x;
  }
};
#endif //OPEN_SPIEL_OPEN_SPIEL_PAPERS_WITH_CODE_TURNPOKERNET_H_
