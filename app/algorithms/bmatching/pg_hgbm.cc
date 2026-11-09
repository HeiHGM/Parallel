#include "bmatching/parallel/pg_hgbm.h"

#include "app/algorithms/algorithm_impl.h"
#include "bmatching/parallel/cuda/CRCW.h"
#include "ds/empty_problem.h"
#include "ds/modifiable_hypergraph.h"
#include "ds/parallel/bmatching.h"
#include "ds/parallel/hypergraph_cuda.h"
namespace heihgm::app::algorithms::bmatching {
struct EmptyHypergraph : public ds::EmptyProblem<ds::StandardIntegerHypergraph> {
  static constexpr absl::string_view ds_name = "empty_hypergraph_b";
  EmptyHypergraph(std::unique_ptr<ds::StandardIntegerHypergraph> g)
      : ds::EmptyProblem<ds::StandardIntegerHypergraph>(std::move(g)) {}
  ~EmptyHypergraph() {}
};
template <class Matching>
struct EmptyToC : public heihgm::app::algorithms::AlgorithmImpl<EmptyHypergraph, Matching> {
 protected:
  absl::StatusOr<std::unique_ptr<EmptyHypergraph>> Load(const RunConfig& run_config,
                                                        const Hypergraph& hypergraph) override {
    TIMED_FUNC(timer);
    if (hypergraph.format() == "hgr" || hypergraph.format() == "tns") {
      auto intermediate =
          heihgm::app::algorithms::loadFile<ds::StandardIntegerHypergraph>(hypergraph);
      if (!intermediate.ok()) {
        return intermediate.status();
      }
      auto gr = std::move(intermediate.value());
      if (run_config.capacity() != -1) {
        for (auto n : gr->nodes()) {
          gr->setNodeWeight(n, std::min((int)run_config.capacity(), (int)gr->nodeDegree(n)));
        }
      }
      auto bool_params = run_config.bool_params();
      auto int_params = run_config.int64_params();
      if (bool_params["set_edge_id_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, e);
        }
      }
      if (bool_params["set_negative_edge_size_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, -gr->edgeSize(e));
        }
      }
      if (int randomness = int_params["randomness"]; randomness > 0) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, gr->edgeWeight(e) - rand() % randomness);
        }
      }
      gr->finish();
      return std::make_unique<EmptyHypergraph>(std::move(gr));
    }

    if (hypergraph.format() == "graph" || hypergraph.format() == "binary_uu") {
      auto intermediate =
          heihgm::app::algorithms::loadFile<heihgm::ds::StandardIntegerHypergraph, false>(
              hypergraph, true);
      if (!intermediate.ok()) {
        return intermediate.status();
      }
      auto gr = std::move(intermediate.value());
      if (run_config.capacity() != -1) {
        for (auto n : gr->nodes()) {
          gr->setNodeWeight(n, run_config.capacity());
        }
      }
      auto bool_params = run_config.bool_params();
      auto int_params = run_config.int64_params();
      if (bool_params["set_edge_id_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, e);
        }
      }
      if (bool_params["set_negative_edge_size_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, -gr->edgeSize(e));
        }
      }
      if (int randomness = int_params["randomness"]; randomness > 0) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, gr->edgeWeight(e) - rand() % randomness);
        }
      }
      gr->finish();
      return std::make_unique<EmptyHypergraph>(std::move(gr));
    }
    return absl::UnimplementedError(" Only hgr is supported");
  }
  absl::StatusOr<std::unique_ptr<Matching>> Execute(const AlgorithmConfig& config,
                                                    std::unique_ptr<EmptyHypergraph> empty,
                                                    bool& exact) override {
    TIMED_FUNC(timer);
    exact = true;
    auto result = std::make_unique<typename Matching::HypergraphType>(*empty->_g);

    return std::make_unique<Matching>(std::move(result));
  }
};

using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
template <typename PMatching, template <bool> typename ComputationModel, bool onDevice,
          bool single_vote>
class PGHM_Matching_Parallel : public heihgm::app::algorithms::AlgorithmImpl<PMatching> {
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
    bool weights = bool_params["weights"];
    {
      TIMED_SCOPE(bla, "INNER");
      ::heihgm::bmatching::parallel::pg_hgbm<ComputationModel, PMatching, DebugInformation,
                                             onDevice, single_vote>(bm, mutable_debug, rand(),
                                                                    weights);
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
#ifdef CUDA_IS_ENABLED
using PBMatching = ds::parallel::CudaBMatching<ds::parallel::CudaHypergraph<int, int, int, int>>;
using EmptyToParallelBMatchingCuda = EmptyToC<PBMatching>;
template <bool X>
using CM = ::heihgm::bmatching::parallel::cuda::CRCWModel<PBMatching, X>;
using CRCW_PGHMCPu = PGHM_Matching_Parallel<PBMatching, CM, false, false>;

using CRCW_PGHM = PGHM_Matching_Parallel<PBMatching, CM, true, false>;
using CRCW_PGHMS = PGHM_Matching_Parallel<PBMatching, CM, true, true>;

REGISTER_IMPL_NAMED(EmptyToParallelBMatchingCuda, "to_cuda_b");
REGISTER_IMPL_NAMED(CRCW_PGHM, "pghm_crcw");
REGISTER_IMPL_NAMED(CRCW_PGHMS, "pghm_crcw_single");

REGISTER_IMPL_NAMED(CRCW_PGHMCPu, "pghm_crcw_cpu");

#endif
}  // namespace heihgm::app::algorithms::bmatching