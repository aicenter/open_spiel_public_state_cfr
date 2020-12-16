#!/bin/bash

RESULTS_DIR=train_test_loop
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/1906.06412.value_functions/value_functions_train_test_loop"
GAMES=(
  "kuhn_poker"
  "leduc_poker"
#  "goofspiel(players=2,num_cards=3,imp_info=True)"
#  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)"
#  "goofspiel(players=2,num_cards=4,imp_info=True)"
  "goofspiel(players=2,num_cards=6,imp_info=True,points_order=ascending)"
)
#DEPTHS=$(seq 3 4)
DEPTHS=3

let "NPROC=`nproc --all` - 1"

parallel -j $NPROC --bar --shuf --header : --results "$RESULTS_DIR" \
         $BINARY $ARGS --game_name={a_game_name} \
                       --depth={b_depth} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_depth $DEPTHS \
         > /dev/null

