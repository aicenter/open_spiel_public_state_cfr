#!/bin/bash

set -eux

module add cmake-3.14.5
module add clang-9.0

wget --show-progress -O "libtorch.zip" "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip"
unzip "libtorch.zip" -d "open_spiel/libtorch/"

wget --show-progress -O "ortools.tar.gz" "https://github.com/google/or-tools/releases/download/v8.0/or-tools_ubuntu-18.04_v8.0.8283.tar.gz"

mkdir open_spiel/build
cd open_spiel/build

OPEN_SPIEL_BUILD_WITH_LIBNOP=ON \
OPEN_SPIEL_BUILD_WITH_PAPERS=ON \
OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON \
OPEN_SPIEL_BUILD_WITH_ORTOOLS=ON \
OPEN_SPIEL_BUILD_WITH_PYTHON=OFF
BUILD_SHARED_LIB=ON \
cmake .. \
-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_FLAGS=-w

make -j 8 train_eval_loop
