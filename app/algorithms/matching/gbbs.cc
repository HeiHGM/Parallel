
#include "app/algorithms/matching/gbbs.h"

#include "app/algorithms/algorithm_impl.h"

namespace heihgm::algorithms::matching::gbbs {
namespace {
using GMatching = external::gbbs::Matching;
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
}  // namespace
class GBBSIMPL : public heihgm::app::algorithms::AlgorithmImpl<GMatching> {
 public:
  static constexpr absl::string_view AlgorithmName = "gbbs";

 protected:
  absl::StatusOr<std::unique_ptr<GMatching>> Execute(const AlgorithmConfig& config,
                                                     std::unique_ptr<GMatching> problem,
                                                     bool& is_exact,
                                                     DebugInformation& mutable_debug) override {
    problem->solve();
    return problem;
  }
  absl::Status ValidateConfig(const AlgorithmConfig& config) override { return absl::OkStatus(); }
  absl::StatusOr<std::unique_ptr<GMatching>> Load(const RunConfig& run_config,
                                                  const Hypergraph& hypergraph) override {
    return absl::UnimplementedError("This function should not be used.");
  }
};
REGISTER_IMPL(GBBSIMPL);
}  // namespace heihgm::algorithms::matching::gbbs