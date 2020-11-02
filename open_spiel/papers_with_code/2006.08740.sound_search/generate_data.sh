#!/usr/bin/env bash

set -eu
# Approximate running time in the full setting was ~200 hours of CPU time.

# The paper used 30000
NUM_SEEDS=1
# The paper used 1000000
TOTAL_ITERATIONS=10000

BINARY="$(pwd)/../../../build/papers_with_code/2006.08740.sound_search/tabularize_oos"
[[ ! -f "$BINARY" ]] && echo "Binary not found: $BINARY" && exit

function get_data() {
  # Print header.
  $BINARY --game=$1 --csv_header

  # Run sampling in parallel.
  parallel --progress \
    $BINARY --game=$1 --total_iterations=$TOTAL_ITERATIONS \
    --seed ::: `seq 1 $NUM_SEEDS`
}
function echo_bold() {  echo -e "\033[1;33m$1\033[m"; }

for GAME in coordinated_mp kuhn_poker; do
  echo_bold "Running experiment for $GAME";
  time get_data $GAME > $GAME.csv
  echo "---"
done
