#!/bin/bash

RESULTS_DIR=value_fn_convergence
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/1906.06412.value_functions/value_fn_convergence"
ARGS="--evaluator=cfr"
GAMES=(
  "matrix_mp"
  "matrix_biased_mp"
  "kuhn_poker"
  "leduc_poker"
  "goofspiel(players=2,num_cards=3,imp_info=True)"
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)"
  "goofspiel(players=2,num_cards=4,imp_info=True)"
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)"
)
DEPTHS=$(seq 1 11)
BANDITS=(
  "RegretMatching"
  "RegretMatchingPlus"
  "PredictiveRegretMatchingPlus"
)
SUBGAME_CFR_ITERATIONS="10 31 100 316 1000"

let "NPROC=`nproc --all` - 1"

parallel -j $NPROC --bar --shuf --header : --results "$RESULTS_DIR" \
         $BINARY $ARGS --game_name={a_game_name} \
                       --bandit_name={b_bandit_name} \
                       --depth={c_depth} \
                        --subgame_cfr_iterations={d_subgame_cfr_iterations} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_bandit_name "${BANDITS[@]}" \
         ::: c_depth $DEPTHS \
         ::: d_subgame_cfr_iterations $SUBGAME_CFR_ITERATIONS \
         > /dev/null

