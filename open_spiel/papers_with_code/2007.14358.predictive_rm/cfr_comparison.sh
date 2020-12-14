#!/bin/bash

RESULTS_DIR=cfr_comparison
BINARY="/home/michal/Code/open_spiel/open_spiel/cmake-build-release/papers_with_code/2007.14358.predictive_rm/cfr_comparison"

GAMES=(
  "small_matrix"
  "matrix_biased_mp"
  "kuhn_poker"
  "leduc_poker"
  "goofspiel(players=2,num_cards=3,imp_info=True)"
  "goofspiel(players=2,num_cards=3,imp_info=True,points_order=ascending)"
  "goofspiel(players=2,num_cards=4,imp_info=True)"
  "goofspiel(players=2,num_cards=4,imp_info=True,points_order=ascending)"
  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=ascending)"
  "goofspiel(players=2,num_cards=5,imp_info=True,points_order=ascending)"
)
BANDITS=(
  "RegretMatching"
  "RegretMatchingPlus"
  "PredictiveRegretMatching"
  "PredictiveRegretMatchingPlus"
)

let "NPROC=`nproc --all` - 1"

parallel -j $NPROC --bar --shuf --header : --results "$RESULTS_DIR" \
         $BINARY --game_name={a_game_name} --bandit_name={b_bandit_name} \
         ::: a_game_name "${GAMES[@]}" \
         ::: b_bandit_name "${BANDITS[@]}" \
         > /dev/null
