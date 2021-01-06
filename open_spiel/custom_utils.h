// Custom utils, not to be shared in the main repo.

#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <chrono>


#define LOG_VAR(x) std::cout << "#x" << x << "\n";

#define INIT_EXPERIMENT() _InitiliazeExperimentRunner(__FILE__, argc, argv)

inline std::string CommandOutput(const std::string& command) {
  FILE *lsofFile_p = popen(command.c_str(), "r");
  if (!lsofFile_p) { return ""; }
  char buffer[1024];
  char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p);
  pclose(lsofFile_p);
  return buffer;
}

inline void _InitiliazeExperimentRunner(std::string origin_file, int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string git_version = CommandOutput("git rev-parse HEAD");
  std::string git_status = CommandOutput("git status --porcelain");
  bool is_clean = git_status == git_version;

  std::cout << "# Running " << origin_file << "\n";
  std::cout << "# Git version " << git_version << "\n";
  if (!is_clean) {
    std::cout << "# Warning! The working tree is not clean!\n"
                 "# This may result in non-reproducible experiment.\n"
                 "# Artificially waiting to notice this " << std::flush;
    for (int i = 0; i < 3; ++i) {
      std::this_thread::sleep_for(static_cast<std::chrono::seconds>(1));
      std::cout << '.' << std::flush;
    }
    std::cout << "\n" << std::endl;
  }

  for (int i = 0; i < argc; ++i) {
    std::cout << "# " << argv[i] << " \\ \n";
  }
  std::cout << "\n" << std::endl;
}
