/**
 * @file heihgm.cc
 * @author Henrik Reinstädtler (reinstaedtler@stud.uni-heidelberg.de)
 * @brief  Main program for b-matching in hypergraphs.
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <google/protobuf/text_format.h>

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "app/algorithms/algorithm_impl.h"
#include "app/app_io.pb.h"
#include "google/protobuf/stubs/common.h"
#include "utils/random.h"

// TODO(reinstaedtler) add non textproto command flags to build text protos on
// command
ABSL_FLAG(std::string, command_textproto, "help", "The command to run in textproto format.");
ABSL_FLAG(int, seed, 0, "The command to run in textproto format.");
#ifdef LOGGING_ENABLED_T
ABSL_FLAG(int, log_level, 0, "The level of logging.");
#endif
namespace {
using heihgm::app::app_io::Result;
}  // namespace
   // needed for libtcmalloc and profiler
char* program_invocation_name = "hypergraph_b_matching";
int main(int argc, char** argv) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  absl::ParseCommandLine(argc, argv);
#ifdef LOGGING_ENABLED_T
  logging_level = absl::GetFlag(FLAGS_log_level);
#endif
  int seed = absl::GetFlag(FLAGS_seed);
  if (seed == 0) {
    seed = (int)time(NULL);
  }
  heihgm::utils::Randomize::instance().setSeed(seed);
  srand(seed);
  heihgm::app::app_io::Command command;
  if (google::protobuf::TextFormat::ParseFromString(absl::GetFlag(FLAGS_command_textproto),
                                                    &command)) {
    if (command.command() == "run") {
      auto& run_config = command.config();
      if (run_config.seeds_size() != 0) {
        for (size_t s = 0; s < run_config.seeds_size(); s++) {
          auto curr_seed = run_config.seeds()[s];
          std::srand(curr_seed);
          auto result =
              heihgm::app::algorithms::Run(command.hypergraph(), command.config(), true);

          result->set_seed(curr_seed);
          result->PrintDebugString();
        }
      } else {
        auto result = heihgm::app::algorithms::Run(command.hypergraph(), command.config(), true);
        if (!result.ok()) {
          std::cout << result.status() << std::endl;
          return 1;
        }
        result->PrintDebugString();
      }
    }
  } else {
    command.set_command("help");
    std::string command_s;
    command.PrintDebugString();
    std::cerr << "Parsing failed." << std::endl;
    return 1;
  }
  return 0;
}