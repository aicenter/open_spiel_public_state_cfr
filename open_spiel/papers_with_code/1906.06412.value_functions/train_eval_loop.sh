#!/bin/bash

RESULTS_DIR=experiments/train_eval_loop
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/1906.06412.value_functions/train_eval_loop"
GAMES=(
  "kuhn_poker"
  "leduc_poker"
)
DEPTHS="3 4 5 6 7"
DL_ITERS=$(seq 1 16)
DATA_GENERATION="mix random dl_cfr"
ARGS="--cfr_oracle_iterations=100 --num_loops=200 --use_bandits_for_cfr=\"RegretMatchingPlus\""
let "NPROC=`nproc --all` - 1"

parallel -j $NPROC --bar --shuf --header : --results "$RESULTS_DIR" \
         $BINARY $ARGS --game_name={a_game_name} \
                       --depth={b_depth} \
                       --trunk_eval_iterations={c_trunk_eval_iterations} \
                       --data_generation={d_data_generation} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_depth $DEPTHS \
         ::: c_trunk_eval_iterations $DL_ITERS \
         ::: d_data_generation $DATA_GENERATION \
         > /dev/null

