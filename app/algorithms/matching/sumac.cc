#include "app/algorithms/matching/sumac.h"

#include <nvtx3/nvToolsExt.h>

#include "app/algorithms/algorithm_impl.h"

namespace heihgm::algorithms::matching::sumac {
namespace {
using SMatching = external::sumac::Matching;
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
}  // namespace
class SLPDPU : public heihgm::app::algorithms::AlgorithmImpl<SMatching> {
 public:
  static constexpr absl::string_view AlgorithmName = "sumac";

 protected:
  absl::StatusOr<std::unique_ptr<SMatching>> Execute(const AlgorithmConfig& config,
                                                     std::unique_ptr<SMatching> problem,
                                                     bool& is_exact,
                                                     DebugInformation& mutable_debug) override {
    //  nvtxRangePushA("SUMAC");  // Start marker
    problem->solve();
    // nvtxRangePop();
    return problem;
  }
  absl::Status ValidateConfig(const AlgorithmConfig& config) override { return absl::OkStatus(); }
  absl::StatusOr<std::unique_ptr<SMatching>> Load(const RunConfig& run_config,
                                                  const Hypergraph& hypergraph) override {
    return absl::UnimplementedError("This function should not be used.");
  }
};
REGISTER_IMPL(SLPDPU);
}  // namespace heihgm::algorithms::matching::sumac