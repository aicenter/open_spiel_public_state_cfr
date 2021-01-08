#!/bin/bash

RESULTS_DIR=experiments/train_eval_loop
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/1906.06412.value_functions/train_eval_loop"
GAMES=(
  "kuhn_poker"
#  "leduc_poker"
#  "goofspiel(players=2,num_cards=3,imp_info=True)"
#  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)"
#  "goofspiel(players=2,num_cards=4,imp_info=True)"
#  "goofspiel(players=2,num_cards=6,imp_info=True,points_order=ascending)"
)
DEPTHS="3 4"
DL_ITERS=$(seq 1 16)
ARGS="--num_loops=10000 --use_bandits_for_cfr=\"RegretMatching\""
let "NPROC=`nproc --all` - 1"

parallel -j $NPROC --bar --shuf --header : --results "$RESULTS_DIR" \
         $BINARY $ARGS --game_name={a_game_name} \
                       --depth={b_depth} \
                       --trunk_eval_iterations={c_trunk_eval_iterations} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_depth $DEPTHS \
         ::: c_trunk_eval_iterations $DL_ITERS \
         > /dev/null

