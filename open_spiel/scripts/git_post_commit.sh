#!/bin/bash

# This is a git commit hook to automatically test each commit to the repository.
# All versions are automatically built under $CHECKOUT_DIR
# Last build can be found under $BUILD_HEAD symlink.
#
# Instructions for installation:
#
# - Copy to .git/hooks/post-commit
# - Update settings for file paths, create corresponding directories.
# - Test the script runs properly with just `run` (uncomment, comment closing outputs)
# - Undo the comment switch.

# Settings
GIT_COMMIT=$(git rev-parse --short=8 HEAD)
OPENSPIEL_HOME="${HOME}/Code/open_spiel"
CHECKOUT_DIR="${OPENSPIEL_HOME}/build_git"
CI_LOG_FILE="${CHECKOUT_DIR}/log"
COMMIT_DIR="${CHECKOUT_DIR}/${GIT_COMMIT}"
BUILD_DIR="${COMMIT_DIR}/build"
PYTHON_ENV="${HOME}/.python_envs/os/bin/activate"
BUILD_HEAD="${OPENSPIEL_HOME}/build_HEAD"

# ------------------------------------------------------------------------------

# Prepare dirs
rm -rf $COMMIT_DIR $BUILD_DIR;
mkdir $COMMIT_DIR $BUILD_DIR;
unlink $BUILD_HEAD
ln -s $BUILD_DIR $BUILD_HEAD

function run() {

# Checkout commit
git archive $GIT_COMMIT | tar -x -C $COMMIT_DIR

# Reuse download cache through symlink
rm -rf "$COMMIT_DIR/download_cache"
ln -s "$OPENSPIEL_HOME/download_cache" "$COMMIT_DIR/download_cache"

# Prepare python
source "$PYTHON_ENV"

# Flags
export BUILD_WITH_PAPERS=ON
export BUILD_WITH_LIBTORCH=ON
export BUILD_WITH_ORTOOLS=ON
export BUILD_WITH_PYTHON=ON
export BUILD_WITH_ACPC=ON

# Go to the commit dir.
cd $COMMIT_DIR

SECONDS=0
./install.sh
install_duration=$SECONDS

# Run tests!
SECONDS=0
./open_spiel/scripts/build_and_run_tests.sh \
  --build_dir="${BUILD_DIR}" \
  --num_threads=8
ST=$?
build_test_duration=$SECONDS

if [ $ST -eq 0 ]; then
  notify-send -u low "OpenSpiel: commit $GIT_COMMIT successful." -i face-smile
  echo "$GIT_COMMIT successful (took ${install_duration}+${build_test_duration} secs)." >> $CI_LOG_FILE
else
  COMMIT_MSG=$(git show --name-only)
  notify-send -u critical "OpenSpiel: commit $GIT_COMMIT failed!" -i dialog-error "$COMMIT_MSG"
  echo "$GIT_COMMIT failed! (took ${install_duration}+${build_test_duration} secs)." >> $CI_LOG_FILE
fi

}

# With output -- useful for testing the hook:
# run

# Close out/err so we can put proc in bg and not wait for commit to finish.
run > "${BUILD_DIR}/log" 2>&1 &
