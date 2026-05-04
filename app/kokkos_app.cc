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

#include <Kokkos_Core.hpp>
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
  Kokkos::initialize(argc, argv);
  int num_devices = Kokkos::num_devices();
  std::cout << "Number of Kokkos devices: " << num_devices << std::endl;
  {
    // Get the number of available NUMA nodes
    int numa_count = Kokkos::hwloc::get_available_numa_count();
    // Print the NUMA count
    std::cout << "Number of NUMA nodes: " << numa_count << std::endl;
  }
  for (int dev = 0; dev < num_devices; ++dev) {
    std::cout << "Device " << dev << ":" << std::endl;

    // Get the device name (requires HWLOC)
    // std::cout << "  Name: " << Kokkos::hwloc::get_device_name(dev) << std::endl;

    // Get the execution space type
    // Kokkos::ExecutionSpace space = Kokkos::ExecutionSpace::device_type(dev);
    // std::cout << "  Execution Space: " << space.name() << std::endl;

    // // You can further query execution space properties:
    // if (space == Kokkos::Cuda) {
    //   // CUDA-specific properties
    //   std::cout << "  CUDA Device ID: " << space.cuda_device_id() << std::endl;
    //   // ... other CUDA properties ...
    // } else if (space == Kokkos::OpenMP) {
    //   // OpenMP-specific properties
    //   // ...
    // }
    // ... and so on for other execution spaces ...
  }
  /*{
    // Define a view on the GPU
    Kokkos::View<double*> a("a", 10);

    // Initialize the view with some values
    Kokkos::parallel_for("init", 10, KOKKOS_LAMBDA(const int i) { a(i) = i; });

    // Print the values from the GPU
    Kokkos::parallel_for(
        "print", 10, KOKKOS_LAMBDA(const int i) { printf("a[%d] = %f\n", i, a(i)); });
  }
  Kokkos::finalize();*/
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
  Kokkos::finalize();
  return 0;
}