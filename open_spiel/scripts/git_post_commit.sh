!/bin/bash

# This is a git commit hook to automatically test each commit to the repository.
# All versions are automatically built under $CHECKOUT_DIR
# Last build can be found under the $BUILD_HEAD symlink.
#
# Usage instructions:
#
# 1. Copy to .git/hooks/post-commit
# 2. Update settings for file paths, create corresponding directories.
OPENSPIEL_HOME="${HOME}/Code/open_spiel"
PYTHON_ENV="${HOME}/.python_envs/os"
# 3. Test the script runs properly with just `run` -- edit the end of this file
#   (uncomment run, comment closing outputs)
#    Run the script directly using
#    $ ./git/hooks/post-commit
# 4. If everything works properly undo the comment switch.
#    Now every time you commit, there will be a new automatic build that will
#    be tested. All past successes/failures are tracked in $CI_LOG_FILE
# 5. From time to time, you may want to clean the $CHECKOUT_DIR as you will
#    have a number of old builds taking up space on your hard drive.

# Optional path settings (with reasonable defaults).
GIT_COMMIT=$(git rev-parse --short=8 HEAD)
CHECKOUT_DIR="${OPENSPIEL_HOME}/build_git"
CI_LOG_FILE="${CHECKOUT_DIR}/log"
COMMIT_DIR="${CHECKOUT_DIR}/${GIT_COMMIT}"
BUILD_DIR="${COMMIT_DIR}/build"
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

# Add symlink to python env
ln -s "$PYTHON_ENV" "$COMMIT_DIR/venv"

# Reuse download cache through symlink
rm -rf "$COMMIT_DIR/download_cache"
ln -s "$OPENSPIEL_HOME/download_cache" "$COMMIT_DIR/download_cache"

# Prepare python
source "$PYTHON_ENV/bin/activate"

# Flags
export OPEN_SPIEL_BUILD_WITH_PAPERS=ON
export OPEN_SPIEL_BUILD_WITH_LIBTORCH=ON
export OPEN_SPIEL_BUILD_WITH_ORTOOLS=ON
export OPEN_SPIEL_BUILD_WITH_PYTHON=ON
export OPEN_SPIEL_BUILD_WITH_ACPC=ON
export OPEN_SPIEL_BUILD_WITH_LIBNOP=ON
# Turn off bunch of python tests.
export OPEN_SPIEL_ENABLE_JAX=OFF
export OPEN_SPIEL_ENABLE_PYTORCH=OFF
export OPEN_SPIEL_ENABLE_TENSORFLOW=OFF

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
