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
  if (!line_p) { return ""; }
  pclose(lsofFile_p);
  return buffer;
}

inline void _InitiliazeExperimentRunner(std::string origin_file,
                                        int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string git_version = CommandOutput("git rev-parse HEAD");
  std::string git_status = CommandOutput("git status --porcelain");
  bool is_clean = git_status == git_version;

  std::cout << "# Running " << origin_file << "\n";
  std::cout << "# Git version " << git_version << "\n";
  std::cout << "# Working tree clean? " << (is_clean ? "true" : "false") << "\n";
  std::cout << "# Launch command:\n";

  for (int i = 0; i < argc; ++i) {
    std::cout << "# " << argv[i];
    if (i + 1 < argc) std::cout << " \\ \n";
  }
  std::cout << "\n" << std::endl;
}
