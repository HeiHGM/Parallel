#pragma once

#include "app/algorithms/algorithm_impl.h"
#include "matching/parallel/local_max.h"

namespace heihgm::algorithms::matching::local_max {
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
template <typename PMatching, typename ComputationModel,
          template <typename, typename, typename> typename RandomGenerator>
class Local_Max_Matching_Parallel : public heihgm::app::algorithms::AlgorithmImpl<PMatching> {
 public:
 protected:
  absl::StatusOr<std::unique_ptr<PMatching>> Execute(const AlgorithmConfig& config,
                                                     std::unique_ptr<PMatching> problem,
                                                     bool& is_exact,
                                                     DebugInformation& mutable_debug) override {
    TIMED_FUNC(fd);
    auto& bm = *problem;
    auto bool_params = config.bool_params();
    auto int_params = config.int64_params();
    auto double_params = config.double_params();
    //   nvtxRangePushA("MATCHING");  // Start marker
    {
      TIMED_SCOPE(bla, "INNER");
      ::heihgm::matching::parallel::local_max<ComputationModel, RandomGenerator, PMatching,
                                                 DebugInformation>(bm, mutable_debug, rand());
    }
    // nvtxRangePop();
    return problem;
  }
  absl::StatusOr<std::unique_ptr<PMatching>> Load(const RunConfig& run_config,
                                                  const Hypergraph& hypergraph) override {
    return absl::UnimplementedError("This function should not be used.");
  }
  absl::Status ValidateConfig(const AlgorithmConfig& config) override { return absl::OkStatus(); }
};
}  // namespace heihgm::algorithms::matching::local_max