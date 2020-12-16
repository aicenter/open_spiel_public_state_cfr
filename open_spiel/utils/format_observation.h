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

#ifndef OPEN_SPIEL_UTILS_FORMAT_OBSERVATION_H_
#define OPEN_SPIEL_UTILS_FORMAT_OBSERVATION_H_

#include <algorithm>
#include <string>
#include <memory>
#include <iostream>

#include "open_spiel/spiel.h"
#include "open_spiel/observer.h"

namespace open_spiel {

inline void _format_binary_value(std::ostream& os, const float& v) {
  if (v == 0) {
    return os << "◯";
  } else if (v == 1) {
    return os << "◉";
  }
  SpielFatalError("Values must all be 0 or 1");
}

inline void _format_nonbinary_value(std::ostream& os, const float& v) {
  os << std::fixed << std::setprecision(2) << std::setw(5) << v << ' ';
}

inline void _format_value(std::ostream& os, const float& v, bool binary) {
  if (binary) {
    _format_binary_value(os, v);
  } else {
    _format_nonbinary_value(os, v);
  }
}

inline void _format_vec(std::ostream& os, absl::Span<const float> vs,
                        bool binary) {
  for (float v : vs) { _format_value(os, v, binary); }
}

inline void _format_matrix(std::ostream& os, absl::Span<const float> vvs,
                           absl::Span<const int> dims, bool binary) {
  SPIEL_CHECK_EQ(dims.size(), 2);
  bool first = true;
  for (int i = 0; i < dims[0]; ++i) {
    if (!first) os << "\n";
    else first = false;
    _format_vec(os, absl::MakeSpan(&vvs[dims[1] * i], dims[1]), binary);
  }
}

inline void _format_tensor(std::ostream& os, absl::Span<const float> vvs,
                           absl::Span<const int> dims, bool binary) {
  SPIEL_CHECK_EQ(dims.size(), 3);
  size_t matrix_size = dims[1] * dims[2];
  bool first = true;
  for (int i = 0; i < dims[0]; ++i) {
    if (!first) os << "\n";
    else first = false;
    _format_matrix(
        os, absl::MakeSpan(&vvs[matrix_size * i], matrix_size),
        absl::MakeSpan(&dims[1], 2), binary);
  }
}

inline bool IsBinaryTensor(absl::Span<const float> tensor) {
  for (int i = 0; i < tensor.size(); ++i) {
    if (tensor[i] != 0 && tensor[i] != 1) {
      return false;
    }
  }
  return true;
}

inline void _format_tensor(
    std::ostream& os, const std::string& name,
    absl::Span<const float> tensor_span,
    absl::Span<const int> tensor_dims) {

  size_t num_dims = tensor_dims.size();
  bool binary = IsBinaryTensor(tensor_span);

  os << name << ' ' << tensor_dims << ": ";
  if (num_dims > 1) {
    os << '\n';
  }
  switch (num_dims) {
    case 0:
      _format_value(os, tensor_span[0], binary);
      break;
    case 1:
      _format_vec(os, tensor_span, binary);
      break;
    case 2:
      _format_matrix(os, tensor_span, tensor_dims, binary);
      break;
    case 3:
      _format_tensor(os, tensor_span, tensor_dims, binary);
      break;
    default:
      SpielFatalError("Cannot print such high-dimensional tensor.");
  }
}

inline std::ostream& operator<<(
    std::ostream& os, const Observation& observation) {
  SPIEL_CHECK_TRUE(observation.HasTensor());
  bool first = true;
  for (const TensorInfo& tensor : observation.tensor_info()) {
    if (!first) os << "\n";
    else first = false;
    absl::Span<const float> tensor_span = observation.Tensor(tensor.name);
    _format_tensor(os, tensor.name, tensor_span, tensor.shape);
  }
  return os;
}

inline std::string ReplaceString(std::string subject, const std::string& search,
                                 const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
}


inline std::string ObservationToString(const Observation& observation, std::string sep="\n") {
  SPIEL_CHECK_TRUE(observation.HasTensor());
  std::stringstream ss;
  bool first = true;
  for (const TensorInfo& tensor : observation.tensor_info()) {
    if (!first) ss << "\n";
    else first = false;
    absl::Span<const float> tensor_span = observation.Tensor(tensor.name);
    _format_tensor(ss, tensor.name, tensor_span, tensor.shape);
  }
  std::string out = ss.str();
  return ReplaceString(out, "\n", sep);
}

}  // open_spiel

#endif OPEN_SPIEL_UTILS_FORMAT_OBSERVATION_H_
