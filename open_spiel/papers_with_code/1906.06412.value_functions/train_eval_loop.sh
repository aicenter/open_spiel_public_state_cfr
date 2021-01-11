#!/bin/bash

RESULTS_DIR=experiments/eval_iters
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/1906.06412.value_functions/train_eval_loop"
GAMES=(
  "kuhn_poker"
  "leduc_poker"
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)"
  "goofspiel(players=2,num_cards=4,imp_info=True)"
)
DEPTHS="1 2 3 4 5 6"
DATA_GENERATION="random dl_cfr"
ARGS="--cfr_oracle_iterations=100 --num_loops=200  --use_bandits_for_cfr=\"RegretMatchingPlus\""
let "NPROC=`nproc --all` - 1"

parallel -j $NPROC -v -v --shuf --header --dry-run : --results "$RESULTS_DIR" \
         $BINARY $ARGS --game_name={a_game_name} \
                       --depth={b_depth} \
                       --data_generation={c_data_generation} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_depth $DEPTHS \
         ::: c_data_generation $DATA_GENERATION

