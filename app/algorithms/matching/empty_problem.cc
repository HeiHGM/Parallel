#include "ds/empty_problem.h"

#include <vector>

#include "app/algorithms/algorithm_impl.h"
#include "app/algorithms/matching/gbbs.h"
#include "ds/bmatching.h"
#include "ds/parallel/hypergraph.h"
#include "ds/parallel/matching.h"
#include "ds/undirected_graph.h"

#ifdef CUDA_IS_ENABLED
#include "app/algorithms/matching/sumac.h"
#include "ds/parallel/hypergraph_cuda.h"

#endif

namespace heihgm::algorithms::matching {
namespace {
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::AlgorithmRunInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::Result;
using heihgm::app::app_io::RunConfig;
}  // namespace
struct EmptyHypergraph : public ds::EmptyProblem<ds::StandardIntegerHypergraph> {
  static constexpr absl::string_view ds_name = "empty_hypergraph";
  EmptyHypergraph(std::unique_ptr<ds::StandardIntegerHypergraph> g)
      : ds::EmptyProblem<ds::StandardIntegerHypergraph>(std::move(g)) {}
  ~EmptyHypergraph() {}
};
template <class Matching>
class EmptyHypergraphTo : public heihgm::app::algorithms::AlgorithmImpl<EmptyHypergraph, Matching> {
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
      for (auto n : gr->nodes()) {
        gr->setNodeWeight(n, 1);
      }
      return std::make_unique<EmptyHypergraph>(std::move(gr));
    }

    if (hypergraph.format() == "graph" || hypergraph.format() == "binary_uu") {
      auto intermediate = heihgm::app::algorithms::loadFile<heihgm::ds::StandardIntegerHypergraph>(
          hypergraph, true);
      if (!intermediate.ok()) {
        return intermediate.status();
      }
      auto gr = std::move(intermediate.value());
      for (auto n : gr->nodes()) {
        gr->setNodeWeight(n, 1);
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
    auto ptr = std::move(empty->_g);
    return std::make_unique<Matching>(std::move(ptr));
  }

 public:
  virtual ~EmptyHypergraphTo() {}
};

using EmptyToParallelMatching =
    EmptyHypergraphTo<ds::parallel::Matching<ds::StandardIntegerHypergraph>>;
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
      for (auto n : gr->nodes()) {
        gr->setNodeWeight(n, 1);
      }
      auto bool_params = run_config.bool_params();
      if (bool_params["set_edge_id_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, e);
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
      for (auto n : gr->nodes()) {
        gr->setNodeWeight(n, 1);
      }
      auto bool_params = run_config.bool_params();

      if (bool_params["set_edge_id_as_weight"]) {
        for (auto e : gr->edges()) {
          gr->setEdgeWeight(e, e);
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

template <class Matching>
struct RandomGenerationMatching : public heihgm::app::algorithms::AlgorithmImpl<Matching> {
 protected:
  absl::StatusOr<std::unique_ptr<Matching>> Execute(const AlgorithmConfig& config,
                                                    std::unique_ptr<Matching> matching,
                                                    bool& exact) override {
    TIMED_FUNC(timer);
    exact = true;

    auto int_params = config.int64_params();
    auto double_params = config.double_params();

    if (int_params.find("random_gen") == int_params.end()) {
      int_params["random_gen"] = 0;
      // std::cout << "Warning: missing random generator specification, initialized instead with
      // default value: " << int_params["random_gen"] << std::endl;
    }
    if (double_params.find("b_left") == double_params.end()) {
      double_params["b_left"] = 0;
      // std::cout << "Warning: missing b_left value, initialized instead with default value: " <<
      // double_params["b_left"] << std::endl;
    }
    if (double_params.find("b_right") == double_params.end()) {
      double_params["b_right"] = 0.00001;
      // std::cout << "Warning: missing b_right value, initialized instead with default value: " <<
      // double_params["b_right"] << std::endl;
    }
    if (int_params["random_gen"] > 2) {
      matching->initialize_noise_weights(double_params["b_left"], double_params["b_right"],
                                         int_params["random_gen"],
                                         12345);  // Generate random numbers pre-algorithm
    }
    return matching;
  }

  absl::StatusOr<std::unique_ptr<Matching>> Load(const RunConfig& run_config,
                                                 const Hypergraph& hypergraph) override {
    return absl::UnimplementedError("Not supported here - should be preprocessed.");
  }
};

using EmptyToParallelMatchingKokkos =
    EmptyToC<ds::parallel::KokkosMatching<ds::parallel::KokkosHypergraph<int, int, int, int>,
                                          Kokkos::View, ds::parallel::EdgeStatus*>>;
using RandomGenerationForMatchingKokkos = RandomGenerationMatching<
    ds::parallel::KokkosMatching<ds::parallel::KokkosHypergraph<size_t, size_t, size_t, int>,
                                 Kokkos::View, ds::parallel::EdgeStatus*>>;
#ifdef CUDA_IS_ENABLED
using EmptyToParallelMatchingCuda =
    EmptyToC<ds::parallel::CudaMatching<ds::parallel::CudaHypergraph<int, int, int, int>>>;
using EmptyToSumacParallelMatching = EmptyToC<external::sumac::Matching>;
using RandomGenerationForMatchingCuda = RandomGenerationMatching<
    ds::parallel::CudaMatching<ds::parallel::CudaHypergraph<int, int, int, int>>>;
#endif
using EmptyToBMatching = EmptyHypergraphTo<heihgm::ds::StandardIntegerHypergraphBMatching>;

REGISTER_IMPL_NAMED(EmptyToParallelMatching, "to_parallel");
REGISTER_IMPL_NAMED(EmptyToParallelMatchingKokkos, "to_kokkos");
REGISTER_IMPL_NAMED(RandomGenerationForMatchingKokkos, "kokkos_random");
#ifdef CUDA_IS_ENABLED
REGISTER_IMPL_NAMED(EmptyToParallelMatchingCuda, "to_cuda");
REGISTER_IMPL_NAMED(RandomGenerationForMatchingCuda, "cuda_random");
REGISTER_IMPL_NAMED(EmptyToSumacParallelMatching, "to_sumac");
#endif
using EmptyToGBBS = EmptyToC<external::gbbs::Matching>;
REGISTER_IMPL_NAMED(EmptyToGBBS, "to_gbbs");
REGISTER_IMPL_NAMED(EmptyToBMatching, "to_bmatching");

}  // namespace heihgm::algorithms::matching