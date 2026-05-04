#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "app/algorithms/algorithm_impl.h"
#include "app/algorithms/streaming_algorithm_impl.h"
#include "app/app_io.pb.h"
#include "ds/bmatching.h"
#include "ds/modifiable_hypergraph.h"
#include "ds/streaming/hypergraph.h"
#include "ds/streaming/matching.h"

namespace heihgm::app::algorithms::streaming::matching {
namespace {
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::Result;
using heihgm::app::app_io::RunConfig;
using heihgm::ds::streaming::StandardFromDiskStreamableHypergraph;
using heihgm::ds::streaming::StandardFromMemoryStreamableHypergraph;
template <class HT>
using MatchingResult = ds::streaming::MatchingResult<HT>;
}  // namespace
template <class StreamableHypergraph, typename DT = double>
struct HypergraphStack {
  std::vector<DT> phi;
  std::vector<typename StreamableHypergraph::StreamMember> stack;
  std::size_t no_edges;
  explicit HypergraphStack(std::size_t no_vertices, std::size_t n_edges)
      : phi(no_vertices), stack(0), no_edges(n_edges) {
    std::fill(phi.begin(), phi.end(), DT{});
  }
  size_t stack_size() const { return stack.size(); }
};
template <class StreamableHypergraph, typename DT = double>
struct HypergraphListStack {
  std::vector<typename StreamableHypergraph::StreamMember*> short_cut;
  // std::list<typename StreamableHypergraph::StreamMember> stack;
  std::size_t no_edges;
  std::size_t curr_no_edges = 0;
  explicit HypergraphListStack(std::size_t no_vertices, std::size_t n_edges)
      : no_edges(n_edges), short_cut(no_vertices, nullptr) {}
  size_t stack_size() const { return curr_no_edges; }
};
template <class ResultTypeOperator, typename DT = double, bool TP = false>
class BasicGreedyMatchingStreaming
    : public StreamingAlgorithmImpl<StandardFromMemoryStreamableHypergraph,
                                    HypergraphStack<StandardFromMemoryStreamableHypergraph, DT>,
                                    typename ResultTypeOperator::ResultType, TP> {
  absl::StatusOr<std::unique_ptr<StandardFromMemoryStreamableHypergraph>> Load(
      const RunConfig& run_config, const Hypergraph& hypergraph) override {
    auto hgr = loadFile<StandardFromMemoryStreamableHypergraph>(hypergraph);
    TIMED_FUNC(timer2);
    if (!hgr.ok()) {
      return hgr.status();
    }
    auto gr = std::move(hgr.value());
    auto bool_params = run_config.bool_params();
    if (!bool_params["no_shuffle"]) {
      gr->shuffle();
    }
    // TODO(reinstaedtler): set capacity
    return gr;
  }
  std::unique_ptr<HypergraphStack<StandardFromMemoryStreamableHypergraph, DT>> InitStorage(
      const AlgorithmConfig& config,
      const StandardFromMemoryStreamableHypergraph& stream) override {
    return std::make_unique<HypergraphStack<StandardFromMemoryStreamableHypergraph, DT>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }

  // Virtual method called once to Transform storage into the result
  absl::StatusOr<std::unique_ptr<typename ResultTypeOperator::ResultType>> Transform(
      HypergraphStack<StandardFromMemoryStreamableHypergraph, DT>& storage) override {
    // std::cout << "Stack: " << storage.stack.size() << std::endl;
    ResultTypeOperator operat;
    return operat.Transform(storage);
  }

 public:
  using StreamableHypergraph = StandardFromMemoryStreamableHypergraph;
};
template <class ResultTypeOperator, typename DT = double, bool TP = false, bool PP = false>
class BasicGreedyMatchingListStreaming
    : public StreamingAlgorithmImpl<StandardFromMemoryStreamableHypergraph,
                                    HypergraphListStack<StandardFromMemoryStreamableHypergraph, DT>,
                                    typename ResultTypeOperator::ResultType, TP, PP> {
  absl::StatusOr<std::unique_ptr<StandardFromMemoryStreamableHypergraph>> Load(
      const RunConfig& run_config, const Hypergraph& hypergraph) override {
    auto hgr = loadFile<StandardFromMemoryStreamableHypergraph>(hypergraph);
    if (!hgr.ok()) {
      return hgr.status();
    }
    auto gr = std::move(hgr.value());
    gr->shuffle();
    // TODO(reinstaedtler): set capacity
    return gr;
  }
  std::unique_ptr<HypergraphListStack<StandardFromMemoryStreamableHypergraph, DT>> InitStorage(
      const AlgorithmConfig& config,
      const StandardFromMemoryStreamableHypergraph& stream) override {
    return std::make_unique<HypergraphListStack<StandardFromMemoryStreamableHypergraph, DT>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }

  // Virtual method called once to Transform storage into the result
  absl::StatusOr<std::unique_ptr<typename ResultTypeOperator::ResultType>> Transform(
      HypergraphListStack<StandardFromMemoryStreamableHypergraph, DT>& storage) override {
    // std::cout << "Stack: " << storage.stack.size() << std::endl;
    ResultTypeOperator operat;
    return operat.Transform(storage);
  }

 public:
  using StreamableHypergraph = StandardFromMemoryStreamableHypergraph;
};
template <class ResultTypeOperator, typename DT = double, bool TP = false, bool PP = false>
class BasicGreedyMatchingListStreamingDisk
    : public StreamingAlgorithmImpl<StandardFromDiskStreamableHypergraph,
                                    HypergraphListStack<StandardFromDiskStreamableHypergraph, DT>,
                                    typename ResultTypeOperator::ResultType, TP, PP> {
  absl::StatusOr<std::unique_ptr<StandardFromDiskStreamableHypergraph>> Load(
      const RunConfig& run_config, const Hypergraph& hypergraph) override {
    const std::vector<std::string> hgr_filenames = absl::StrSplit(hypergraph.file_path(), "?");
    std::string weight_f = "";
    if (hgr_filenames.size() > 1) {
      std::filesystem::path path(hgr_filenames[0]);
      weight_f =
          path.parent_path().string() + (path.has_parent_path() ? "/" : "") + hgr_filenames[1];
    }
    return std::make_unique<StandardFromDiskStreamableHypergraph>(hgr_filenames[0],
                                                                  hypergraph.format(), weight_f);
  }
  std::unique_ptr<HypergraphListStack<StandardFromDiskStreamableHypergraph, DT>> InitStorage(
      const AlgorithmConfig& config, const StandardFromDiskStreamableHypergraph& stream) override {
    return std::make_unique<HypergraphListStack<StandardFromDiskStreamableHypergraph, DT>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }

  // Virtual method called once to Transform storage into the result
  absl::StatusOr<std::unique_ptr<typename ResultTypeOperator::ResultType>> Transform(
      HypergraphListStack<StandardFromDiskStreamableHypergraph, DT>& storage) override {
    // std::cout << "Stack: " << storage.stack.size() << std::endl;
    ResultTypeOperator operat;
    return operat.Transform(storage);
  }

 public:
  using StreamableHypergraph = StandardFromDiskStreamableHypergraph;
};
template <class ResultTypeOperator, typename DT = double, bool TP = false>
class BasicGreedyMatchingStreamingDisk
    : public StreamingAlgorithmImpl<StandardFromDiskStreamableHypergraph,
                                    HypergraphStack<StandardFromDiskStreamableHypergraph, DT>,
                                    typename ResultTypeOperator::ResultType, TP> {
  absl::StatusOr<std::unique_ptr<StandardFromDiskStreamableHypergraph>> Load(
      const RunConfig& run_config, const Hypergraph& hypergraph) override {
    const std::vector<std::string> hgr_filenames = absl::StrSplit(hypergraph.file_path(), "?");
    std::string weight_f = "";
    if (hgr_filenames.size() > 1) {
      std::filesystem::path path(hgr_filenames[0]);
      weight_f =
          path.parent_path().string() + (path.has_parent_path() ? "/" : "") + hgr_filenames[1];
    }
    return std::make_unique<StandardFromDiskStreamableHypergraph>(hgr_filenames[0],
                                                                  hypergraph.format(), weight_f);
  }
  std::unique_ptr<HypergraphStack<StandardFromDiskStreamableHypergraph, DT>> InitStorage(
      const AlgorithmConfig& config, const StandardFromDiskStreamableHypergraph& stream) override {
    return std::make_unique<HypergraphStack<StandardFromDiskStreamableHypergraph, DT>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
  ResultTypeOperator operat;
  // Virtual method called once to Transform storage into the result
  absl::StatusOr<std::unique_ptr<typename ResultTypeOperator::ResultType>> Transform(
      HypergraphStack<StandardFromDiskStreamableHypergraph, DT>& storage) override {
    // std::cout << "Stack: " << storage.stack.size() << std::endl;
    return operat.Transform(storage);
  }

 public:
  using StreamableHypergraph = StandardFromDiskStreamableHypergraph;
};
template <template <typename, typename> typename HS, typename StreamableHypergraph,
          typename DT = double>
class MatchingResultOperatorBase {
 public:
  using ResultType = MatchingResult<StreamableHypergraph>;
  auto Transform(HS<StreamableHypergraph, DT>& storage) {
    auto matchingResult = std::make_unique<MatchingResult<StreamableHypergraph>>(
        storage.phi.size(), storage.stack_size());
    for (auto ix = storage.stack.rbegin(); ix != storage.stack.rend(); ix++) {
      if (matchingResult->matchable(*ix)) {
        matchingResult->addToMatching(*ix);
      }
    }
    return matchingResult;
  }
  auto Transform2(const typename StreamableHypergraph::StreamMember& next,
                  ResultType& prev_result) {
    if (prev_result.matchable(next)) {
      prev_result.addToMatching(next);
    }
  }
};
template <typename StreambleHypergraph, typename DT = double>
using MatchingResultOperator = MatchingResultOperatorBase<HypergraphStack, StreambleHypergraph, DT>;
template <typename StreamableHypergraph, typename DT = double>
class ListMatchingResultOperator {
 public:
  using ResultType = MatchingResult<StreamableHypergraph>;
  auto Transform(HypergraphListStack<StreamableHypergraph, DT>& storage) {
    auto matchingResult = std::make_unique<MatchingResult<StreamableHypergraph>>(
        storage.short_cut.size(), storage.stack_size());
    for (auto ix = storage.short_cut.rbegin(); ix != storage.short_cut.rend(); ix++) {
      if (*ix == nullptr) {
        continue;
      }

      matchingResult->addToMatching(**ix);
      for (auto p : (*ix)->pins) {
        storage.short_cut[p] = nullptr;
      }
      delete *ix;
    }
    return matchingResult;
  }
  void Transform2(const typename StreamableHypergraph::StreamMember& next,
                  ResultType& prev_result) {
    if (prev_result.matchable(next)) {
      prev_result.addToMatching(next);
    }
  }
};
template <typename DT = double>
class MatchingResultOperatorOR {
 public:
  using ResultType = MatchingResult<StandardFromMemoryStreamableHypergraph>;
  auto Transform(HypergraphStack<StandardFromMemoryStreamableHypergraph, DT>& storage) {
    auto matchingResult = std::make_unique<ResultType>(storage.phi.size(), storage.stack.size());
    for (auto ix = storage.stack.rbegin(); ix != storage.stack.rend(); ix++) {
      if (matchingResult->matchable(*ix)) {
        matchingResult->addToMatching(*ix);
      }
    }
    return matchingResult;
  }
  auto Transform2(const typename StandardFromMemoryStreamableHypergraph::StreamMember& next,
                  ResultType& prev_result) {
    if (prev_result.matchable_or(next)) {
      prev_result.addToMatching(next);
    }
  }
};
template <typename StreamableHypergraph, typename DT = double>
class FullBMatchingOperator {
 public:
  using ResultType = ds::StandardIntegerHypergraphBMatching;
  auto Transform(HypergraphStack<StreamableHypergraph, DT>& storage) {
    auto from_storage = std::make_unique<ds::StandardIntegerHypergraph>(
        storage.stack.size(), storage.phi.size(), true, true);
    for (size_t i = 0; i < storage.phi.size(); i++) {
      from_storage->setNodeWeight(i, 1);
    }
    for (const auto& edge : storage.stack) {
      from_storage->addEdge(edge.pins, edge.w);
    }
    return std::make_unique<ResultType>(std::move(from_storage));
  }
  void Transform2(const StreamableHypergraph::StreamMember& next, ResultType& prev_result) {
    bool matchable = true;
    for (auto n : next.pins) {
      if (prev_result.matchesAtNode(n) < prev_result.capacity(n)) {
      } else {
        matchable = false;
        break;
      }
    }
    if (matchable) {
      auto e = prev_result.addEdge(next.pins, next.w);
      assert(prev_result.isMatchable(e));
      prev_result.addToMatching(e);
    }
  }
};

template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TS = false>
class GreedyMatchingStreaming : public BGM<ResultTypeOperator, double, TS> {
  double eps = 0.0;
  // Stream into storage
  void Stream(const typename BGM<ResultTypeOperator, double,
                                 TS>::StreamableHypergraph::StreamMember& member,
              HypergraphStack<typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph>&
                  storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (member.w > weight * (1.0 + eps)) {
      auto offset = member.w - weight;
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] += offset;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename BGM<ResultTypeOperator, double,
                                       TS>::StreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    operat.Transform2(stream, prev_result);
  }
  std::unique_ptr<
      HypergraphStack<typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>
  InitStorage(
      const AlgorithmConfig& config,
      const typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph& stream) override {
    auto double_params = config.double_params();
    eps = double_params["eps"];
    return std::make_unique<HypergraphStack<
        typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
};
template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TS = false>
class GreedyMatchingListStreaming : public BGM<ResultTypeOperator, double, TS> {
  double eps = 0.0;
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember&
          member,
      HypergraphListStack<typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph>&
          storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      if (storage.short_cut[p] != nullptr) {
        weight += storage.short_cut[p]->w;
      }
    }
    if (member.w > weight * (1.0 + eps)) {
      auto offset = member.w - weight;
      if (weight > 0.0) {
        for (auto p : member.pins) {
          if (storage.short_cut[p] != nullptr) {
            typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember* s =
                storage.short_cut[p];

            for (auto p3 : s->pins) {
              storage.short_cut[p3] = nullptr;
            }
            delete s;
            storage.curr_no_edges--;
          }
        }
      }
      typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember* pointer =
          new
          typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember(member);
      storage.curr_no_edges++;
      for (auto p : member.pins) {
        storage.short_cut[p] = pointer;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename BGM<ResultTypeOperator, double,
                                       TS>::StreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    operat.Transform2(stream, prev_result);
  }
  std::unique_ptr<HypergraphListStack<
      typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>
  InitStorage(
      const AlgorithmConfig& config,
      const typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph& stream) override {
    auto double_params = config.double_params();
    eps = double_params["eps"];
    return std::make_unique<HypergraphListStack<
        typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
};

template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TS = false>
class GreedyMatchingSetStreaming : public BGM<ResultTypeOperator, double, TS> {
  double eps = 0.0;
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember&
          member,
      HypergraphListStack<typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph>&
          storage) override {
    double weight = 0;
    std::set<typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember*>
        seen;
    for (auto p : member.pins) {
      if (storage.short_cut[p] != nullptr && !seen.contains(storage.short_cut[p])) {
        weight += storage.short_cut[p]->w;
        seen.insert(storage.short_cut[p]);
      }
    }
    if (member.w > weight * (1.0 + eps)) {
      auto offset = member.w - weight;
      if (weight > 0.0) {
        for (auto p : member.pins) {
          if (storage.short_cut[p] != nullptr) {
            typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember* s =
                storage.short_cut[p];

            for (auto p3 : s->pins) {
              storage.short_cut[p3] = nullptr;
            }
            delete s;
            storage.curr_no_edges--;
          }
        }
      }
      typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember* pointer =
          new
          typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph::StreamMember(member);
      storage.curr_no_edges++;
      for (auto p : member.pins) {
        storage.short_cut[p] = pointer;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename BGM<ResultTypeOperator, double,
                                       TS>::StreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    operat.Transform2(stream, prev_result);
  }
  std::unique_ptr<HypergraphListStack<
      typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>
  InitStorage(
      const AlgorithmConfig& config,
      const typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph& stream) override {
    auto double_params = config.double_params();
    eps = double_params["eps"];
    return std::make_unique<HypergraphListStack<
        typename BGM<ResultTypeOperator, double, TS>::StreamableHypergraph, double>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
};

template <template <class, typename, bool, bool> class BGM, class ResultTypeOperator,
          bool TS = false>
class GreedyMatchingSetStreamingBestApprox : public BGM<ResultTypeOperator, double, TS, true> {
  double eps = 0.0;
  void PreStream(const typename BGM<ResultTypeOperator, double, TS,
                                    true>::StreamableHypergraph::StreamMember& member) {
    eps = std::max(eps, (double)member.pins.size());
  }
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph::StreamMember&
          member,
      HypergraphListStack<typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph>&
          storage) override {
    double weight = 0;
    std::set<
        typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph::StreamMember*>
        seen;
    for (auto p : member.pins) {
      if (storage.short_cut[p] != nullptr && !seen.contains(storage.short_cut[p])) {
        weight += storage.short_cut[p]->w;
        seen.insert(storage.short_cut[p]);
      }
    }
    if (member.w > weight * (1.0 + eps)) {
      auto offset = member.w - weight;
      if (weight > 0.0) {
        for (auto p : member.pins) {
          if (storage.short_cut[p] != nullptr) {
            typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph::StreamMember*
                s = storage.short_cut[p];

            for (auto p3 : s->pins) {
              storage.short_cut[p3] = nullptr;
            }
            delete s;
            storage.curr_no_edges--;
          }
        }
      }
      typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph::StreamMember*
          pointer = new
          typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph::StreamMember(
              member);
      storage.curr_no_edges++;
      for (auto p : member.pins) {
        storage.short_cut[p] = pointer;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename BGM<ResultTypeOperator, double, TS,
                                       true>::StreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    operat.Transform2(stream, prev_result);
  }
  std::unique_ptr<HypergraphListStack<
      typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph, double>>
  InitStorage(
      const AlgorithmConfig& config,
      const typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph& stream,
      app_io::DebugInformation& debug) override {
    // auto double_params = config.double_params();
    eps = std::sqrt((eps - 1.0) / eps);
    debug.mutable_double_info()->insert({"eps", eps});
    return std::make_unique<HypergraphListStack<
        typename BGM<ResultTypeOperator, double, TS, true>::StreamableHypergraph, double>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
};
template <template <class> class BGM, class ResultTypeOperator>
class GreedyLogMatchingStreaming : public BGM<ResultTypeOperator> {
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator>::StreamableHypergraph::StreamMember& member,
      HypergraphStack<typename BGM<ResultTypeOperator>::StreamableHypergraph>& storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (log(member.w) / log(member.pins.size()) > weight) {
      auto offset = log(member.w) / log(member.pins.size()) - weight;
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] += offset;
      }
    }
  }
};
template <template <class, typename> class BGM, class ResultTypeOperator>
class NaiveMatchingStreaming : public BGM<ResultTypeOperator, bool> {
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator, bool>::StreamableHypergraph::StreamMember& member,
      HypergraphStack<typename BGM<ResultTypeOperator, bool>::StreamableHypergraph, bool>& storage)
      override {
    bool weight = 0;
    for (auto p : member.pins) {
      weight |= storage.phi[p];
    }
    if (weight == 0) {
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] = 1;
      }
    }
  }
};
template <template <class> class BGM, class ResultTypeOperator>
class GreedyScaledMatchingStreaming : public BGM<ResultTypeOperator> {
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator>::StreamableHypergraph::StreamMember& member,
      HypergraphStack<typename BGM<ResultTypeOperator>::StreamableHypergraph>& storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (((double)member.w) / ((double)member.pins.size()) > weight) {
      auto offset = ((double)member.w) / ((double)member.pins.size()) - weight;
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] += offset;
      }
    }
  }
};
template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TP = false>
class GreedyHalfScaledMatchingStreaming : public BGM<ResultTypeOperator, double, TP> {
  // Stream into storage
  void Stream(const typename BGM<ResultTypeOperator, double,
                                 TP>::StreamableHypergraph::StreamMember& member,
              HypergraphStack<typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph>&
                  storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (((double)member.w) > weight) {
      auto offset = ((double)member.w) / ((double)member.pins.size());
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] = offset;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename StandardFromMemoryStreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    return operat.Transform2(stream, prev_result);
  }
};
template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TP = false>
class GreedyHalfLenientScaledMatchingStreaming : public BGM<ResultTypeOperator, double, TP> {
  double eps = 0;
  // Stream into storage
  void Stream(const typename BGM<ResultTypeOperator, double,
                                 TP>::StreamableHypergraph::StreamMember& member,
              HypergraphStack<typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph>&
                  storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (((double)member.w) > weight * (1.0 + eps)) {
      auto offset = ((double)member.w - weight) / ((double)member.pins.size());
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] += offset;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename StandardFromMemoryStreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    return operat.Transform2(stream, prev_result);
  }
  std::unique_ptr<
      HypergraphStack<typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph, double>>
  InitStorage(
      const AlgorithmConfig& config,
      const typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph& stream) override {
    auto double_params = config.double_params();
    eps = double_params["eps"];
    return std::make_unique<HypergraphStack<
        typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph, double>>(
        stream.currentNumNodes(), stream.currentNumEdges());
  }
};

template <template <class, typename, bool> class BGM, class ResultTypeOperator, bool TP = false>
class GreedyQuadratureHalfScaledMatchingStreaming : public BGM<ResultTypeOperator, double, TP> {
  // Stream into storage
  void Stream(const typename BGM<ResultTypeOperator, double,
                                 TP>::StreamableHypergraph::StreamMember& member,
              HypergraphStack<typename BGM<ResultTypeOperator, double, TP>::StreamableHypergraph>&
                  storage) override {
    double weight = 0;
    for (auto p : member.pins) {
      weight += storage.phi[p];
    }
    if (((double)member.w) / ((double)member.pins.size()) > weight) {
      auto offset = ((double)member.w) / ((double)member.pins.size() * (double)member.pins.size());
      storage.stack.push_back(member);
      for (auto p : member.pins) {
        storage.phi[p] = offset;
      }
    }
  }
  ResultTypeOperator operat;

  void StreamSecond(const typename BGM<ResultTypeOperator, double,
                                       TP>::StreamableHypergraph::StreamMember& stream,
                    typename ResultTypeOperator::ResultType& prev_result) {
    return operat.Transform2(stream, prev_result);
  }
};

template <template <class, typename> class BGM, class ResultTypeOperator>
class GreedyHalfAndGuaranteedScaledMatchingStreaming
    : public BGM<ResultTypeOperator, std::array<double, 2>> {
  double alpha = 1.0;
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator, double>::StreamableHypergraph::StreamMember& member,
      HypergraphStack<typename BGM<ResultTypeOperator, double>::StreamableHypergraph,
                      std::array<double, 2>>& storage) override {
    double weight0 = 0;
    double weight1 = 0;
    for (auto p : member.pins) {
      weight0 += storage.phi[p][0];
      weight1 += storage.phi[p][1];
    }
    weight0 *= alpha;
    weight1 *= alpha;

    if (((double)member.w) > weight0 || ((double)member.w) > weight1) {
      storage.stack.push_back(member);
    }
    if (((double)member.w) > weight0) {
      auto offset = ((double)member.w) / ((double)member.pins.size());
      for (auto p : member.pins) {
        storage.phi[p][0] = offset;
      }
    }
    if (((double)member.w) > weight1) {
      auto offset = member.w - weight1 / alpha;
      for (auto p : member.pins) {
        storage.phi[p][1] += offset;
      }
    }
  }
  std::unique_ptr<
      HypergraphStack<typename BGM<ResultTypeOperator, std::array<double, 2>>::StreamableHypergraph,
                      std::array<double, 2>>>
  InitStorage(const AlgorithmConfig& config,
              const typename BGM<ResultTypeOperator, std::array<double, 2>>::StreamableHypergraph&
                  stream) override {
    auto double_params = config.double_params();
    alpha = double_params["alpha"] == 0 ? 1.0 : double_params["alpha"];
    return std::make_unique<HypergraphStack<
        typename BGM<ResultTypeOperator, std::array<double, 2>>::StreamableHypergraph,
        std::array<double, 2>>>(stream.currentNumNodes(), stream.currentNumEdges());
  }
};

template <template <class, typename> class BGM, class ResultTypeOperator>
class GreedyHalfAndGuaranteedScaledMatchingStreamingGEQ
    : public BGM<ResultTypeOperator, std::array<double, 2>> {
  // Stream into storage
  void Stream(
      const typename BGM<ResultTypeOperator,
                         std::array<double, 2>>::StreamableHypergraph::StreamMember& member,
      HypergraphStack<typename BGM<ResultTypeOperator, std::array<double, 2>>::StreamableHypergraph,
                      std::array<double, 2>>& storage) override {
    double weight0 = 0;
    double weight1 = 0;
    for (auto p : member.pins) {
      weight0 += storage.phi[p][0];
      weight1 += storage.phi[p][1];
    }
    if (((double)member.w) >= weight0 || ((double)member.w) >= weight1) {
      storage.stack.push_back(member);
    }
    if (((double)member.w) >= weight0) {
      auto offset = ((double)member.w) / ((double)member.pins.size());
      for (auto p : member.pins) {
        storage.phi[p][0] = offset;
      }
    }
    if (((double)member.w) >= weight1) {
      auto offset = member.w - weight1;
      for (auto p : member.pins) {
        storage.phi[p][1] += offset;
      }
    }
  }
};
template <typename NT, typename WT>
struct BareHypergraph {
  using EdgeType1 = std::pair<WT, std::vector<NT>>;
  using EdgeType = NT;
  using NodeType = NT;
  using WeightType = WT;
  std::vector<EdgeType1> edges_;
  size_t n_e;
  size_t n_n;
  size_t currentNumNodes() const { return n_n; }
  size_t currentNumEdges() const { return n_e; }
  size_t initialNumNodes() const { return n_n; }
  size_t initialNumEdges() const { return n_e; }
  void shrink_to_size() {}
  BareHypergraph(size_t num_edges, size_t num_nodes, bool f1 = false, bool f = false)
      : n_e(num_edges), n_n(num_nodes) {
    edges_.reserve(num_edges);
  }
  void addEdge(const std::vector<NT>& edge, WT weight) {
    edges_.push_back(std::make_pair(weight, edge));
  }
  void addEdge(const NT u, const NT b, const WT) {}
  void resortEdge(size_t) {}
  void finish() {}
  void setNodeWeight(NT, WT) {}
  void setEdgeWeight(size_t i, WT w) { edges_[i].first = w; }
  utils::Range<utils::SimpleIterator> edges() const { return {0, edges_.size()}; }
  utils::Range<utils::SimpleIterator> nodes() const { return {0, n_n}; }
  auto edgeSize(size_t e) { return edges_[e].second.size(); }
};
template <class HT>
struct SimpleMatching {
  using EdgeType1 = typename HT::EdgeType1;
  using EdgeType = typename HT::EdgeType;
  std::unique_ptr<HT> graph;
  std::vector<bool> matchedAt;
  std::vector<typename HT::EdgeType1> result;
  bool isMatchable(const EdgeType1& edge) {
    for (const auto p : edge.second) {
      if (matchedAt[p]) {
        return false;
      }
    }
    return true;
  }
  void addToMatching(const EdgeType1& edge) {
    result.push_back(edge);
    for (const auto p : edge.second) {
      matchedAt[p] = true;
    }
  }
  int64_t weight() const {
    int64_t w = 0;
    for (const auto& [ws, _] : result) {
      w += ws;
    }
    return w;
  }
  bool valid() {
    std::fill(matchedAt.begin(), matchedAt.end(), false);
    for (const auto& r : result) {
      if (!isMatchable(r)) {
        return false;
      }
      for (auto p : r.second) {
        matchedAt[p] = true;
      }
    }
    return true;
  }
  int64_t size() const { return result.size(); }
  double quality() const { return 0; }
  int64_t free_edges_size() const { return 0; }
  const HT& instance() const { return *graph; }
  static constexpr absl::string_view ds_name = "simple_matching";
  explicit SimpleMatching(std::unique_ptr<HT> hg)
      : graph(std::move(hg)), matchedAt(graph->currentNumNodes(), false) {}
};
using SimpleMatchingT = SimpleMatching<BareHypergraph<size_t, int>>;
struct GreedySlim : AlgorithmImpl<SimpleMatchingT> {
  absl::StatusOr<std::unique_ptr<SimpleMatchingT>> Load(const app_io::RunConfig& run_config,
                                                        const app_io::Hypergraph& hypergraph) {
    auto hgr = loadFile<BareHypergraph<size_t, int>>(hypergraph);
    return std::make_unique<SimpleMatchingT>(std::move(hgr.value()));
  }
  absl::StatusOr<std::unique_ptr<SimpleMatchingT>> Execute(
      const AlgorithmConfig& config, std::unique_ptr<SimpleMatchingT> matching) {
    std::vector<typename SimpleMatchingT::EdgeType> order;
    for (auto e : matching->instance().edges()) {
      order.push_back(e);
    }
    std::random_shuffle(order.begin(), order.end());
    auto eval_function = [&](auto e) { return matching->graph->edges_[e].first; };
    std::sort(order.begin(), order.end(),
              [&](auto a, auto b) { return eval_function(a) > eval_function(b); });
    for (auto e1 : order) {
      const auto& e = matching->graph->edges_[e1];
      if (matching->isMatchable(e)) {
        matching->addToMatching(e);
      }
    }
    return matching;
  }
};

using GreedyMatchingListStreamingMatching =
    GreedyMatchingListStreaming<BasicGreedyMatchingListStreaming,
                                ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyMatchingListStreamingMatchingDisk =
    GreedyMatchingListStreaming<BasicGreedyMatchingListStreamingDisk,
                                ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>>;

using GreedyMatchingSetStreamingMatching =
    GreedyMatchingSetStreaming<BasicGreedyMatchingListStreaming,
                               ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyMatchingSetStreamingMatchingDisk =
    GreedyMatchingSetStreaming<BasicGreedyMatchingListStreamingDisk,
                               ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyMatchingSetStreamingMatchingBestEvict = GreedyMatchingSetStreamingBestApprox<
    BasicGreedyMatchingListStreaming,
    ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyMatchingSetStreamingMatchingBestEvictDisk = GreedyMatchingSetStreamingBestApprox<
    BasicGreedyMatchingListStreamingDisk,
    ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using TwoGreedyMatchingSetStreamingMatchingBestEvict = GreedyMatchingSetStreamingBestApprox<
    BasicGreedyMatchingListStreaming,
    ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using TwoGreedyMatchingSetStreamingMatchingBestEvictDisk = GreedyMatchingSetStreamingBestApprox<
    BasicGreedyMatchingListStreamingDisk,
    ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using TwoGreedyMatchingSetStreamingMatching =
    GreedyMatchingSetStreaming<BasicGreedyMatchingListStreaming,
                               ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>,
                               true>;
using TwoGreedyMatchingSetStreamingMatchingDisk =
    GreedyMatchingSetStreaming<BasicGreedyMatchingListStreamingDisk,
                               ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>,
                               true>;
using TwoGreedyMatchingListStreamingMatching =
    GreedyMatchingListStreaming<BasicGreedyMatchingListStreaming,
                                ListMatchingResultOperator<StandardFromMemoryStreamableHypergraph>,
                                true>;
using TwoGreedyMatchingListStreamingMatchingDisk =
    GreedyMatchingListStreaming<BasicGreedyMatchingListStreamingDisk,
                                ListMatchingResultOperator<StandardFromDiskStreamableHypergraph>,
                                true>;
using GreedyMatchingStreamingMatching =
    GreedyMatchingStreaming<BasicGreedyMatchingStreaming,
                            MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using NaiveMatchingStreamingMatching =
    NaiveMatchingStreaming<BasicGreedyMatchingStreaming,
                           MatchingResultOperator<StandardFromMemoryStreamableHypergraph, bool>>;
using NaiveMatchingStreamingMatchingDisk =
    NaiveMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                           MatchingResultOperator<StandardFromDiskStreamableHypergraph, bool>>;
using TwoPassGreedyMatchingStreamingMatching =
    GreedyMatchingStreaming<BasicGreedyMatchingStreaming,
                            MatchingResultOperator<StandardFromMemoryStreamableHypergraph>, true>;
using TwoPassGreedyMatchingStreamingMatchingOR =
    GreedyMatchingStreaming<BasicGreedyMatchingStreaming, MatchingResultOperatorOR<>, true>;
using GreedyQuadratureHalfScaledMatchingStreamingNormal =
    GreedyQuadratureHalfScaledMatchingStreaming<
        BasicGreedyMatchingStreaming,
        MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using TwoPassGreedyQuadratureHalfScaledMatchingStreamingNormal =
    GreedyQuadratureHalfScaledMatchingStreaming<
        BasicGreedyMatchingStreaming,
        MatchingResultOperator<StandardFromMemoryStreamableHypergraph>, true>;
using GreedyQuadratureHalfScaledMatchingStreamingFull = GreedyQuadratureHalfScaledMatchingStreaming<
    BasicGreedyMatchingStreaming, FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using TwoPassGreedyHalfScaledMatchingStreamingNormal = GreedyHalfScaledMatchingStreaming<
    BasicGreedyMatchingStreaming, MatchingResultOperator<StandardFromMemoryStreamableHypergraph>,
    true>;
using GreedyLogMatchingStreamingMatching =
    GreedyLogMatchingStreaming<BasicGreedyMatchingStreaming,
                               MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyScaledMatchingStreamingMatching =
    GreedyScaledMatchingStreaming<BasicGreedyMatchingStreaming,
                                  MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyHalfScaledMatchingStreamingMatching = GreedyHalfScaledMatchingStreaming<
    BasicGreedyMatchingStreaming, MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;

using GreedyMatchingStreamingMatchingFull =
    GreedyMatchingStreaming<BasicGreedyMatchingStreaming,
                            FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyLogMatchingStreamingMatchingFull =
    GreedyLogMatchingStreaming<BasicGreedyMatchingStreaming,
                               FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyScaledMatchingStreamingMatchingFull =
    GreedyScaledMatchingStreaming<BasicGreedyMatchingStreaming,
                                  FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyHalfScaledMatchingStreamingMatchingFull = GreedyHalfScaledMatchingStreaming<
    BasicGreedyMatchingStreaming, FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyHalfAndGuaranteedScaledMatchingStreamingFull =
    GreedyHalfAndGuaranteedScaledMatchingStreaming<
        BasicGreedyMatchingStreaming,
        FullBMatchingOperator<StandardFromMemoryStreamableHypergraph, std::array<double, 2>>>;
using GreedyHalfAndGuaranteedScaledMatchingStreamingGEQFull =
    GreedyHalfAndGuaranteedScaledMatchingStreamingGEQ<
        BasicGreedyMatchingStreaming,
        FullBMatchingOperator<StandardFromMemoryStreamableHypergraph, std::array<double, 2>>>;

// Disk
using GreedyMatchingStreamingMatchingDisk =
    GreedyMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                            MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using TwoPassGreedyMatchingStreamingMatchingDisk =
    GreedyMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                            MatchingResultOperator<StandardFromDiskStreamableHypergraph>, true>;
using GreedyQuadratureHalfScaledMatchingStreamingNormalDisk =
    GreedyQuadratureHalfScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using TwoPassGreedyQuadratureHalfScaledMatchingStreamingNormalDisk =
    GreedyQuadratureHalfScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        MatchingResultOperator<StandardFromDiskStreamableHypergraph>, true>;
using GreedyQuadratureHalfScaledMatchingStreamingFullDisk =
    GreedyQuadratureHalfScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;
using TwoPassGreedyHalfScaledMatchingStreamingNormalDisk =
    GreedyHalfScaledMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                                      MatchingResultOperator<StandardFromDiskStreamableHypergraph>,
                                      true>;
using GreedyLogMatchingStreamingMatchingDisk =
    GreedyLogMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                               MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyScaledMatchingStreamingMatchingDisk =
    GreedyScaledMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                                  MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyHalfScaledMatchingStreamingMatchingDisk =
    GreedyHalfScaledMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                                      MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyLenientHalfScaledMatchingStreamingMatchingDisk =
    GreedyHalfLenientScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        MatchingResultOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyLenientHalfScaledMatchingStreamingMatchingDiskFull =
    GreedyHalfLenientScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;

using TwoGreedyLenientHalfScaledMatchingStreamingMatchingDisk =
    GreedyHalfLenientScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        MatchingResultOperator<StandardFromDiskStreamableHypergraph>, true>;
using GreedyLenientHalfScaledMatchingStreamingMatching = GreedyHalfLenientScaledMatchingStreaming<
    BasicGreedyMatchingStreaming, MatchingResultOperator<StandardFromMemoryStreamableHypergraph>>;
using GreedyLenientHalfScaledMatchingStreamingMatchingFull =
    GreedyHalfLenientScaledMatchingStreaming<
        BasicGreedyMatchingStreaming,
        FullBMatchingOperator<StandardFromMemoryStreamableHypergraph>>;
using TwoGreedyLenientHalfScaledMatchingStreamingMatching =
    GreedyHalfLenientScaledMatchingStreaming<
        BasicGreedyMatchingStreaming,
        MatchingResultOperator<StandardFromMemoryStreamableHypergraph>, true>;
using GreedyMatchingStreamingMatchingFullDisk =
    GreedyMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                            FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyLogMatchingStreamingMatchingFullDisk =
    GreedyLogMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                               FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyScaledMatchingStreamingMatchingFullDisk =
    GreedyScaledMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                                  FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyHalfScaledMatchingStreamingMatchingFullDisk =
    GreedyHalfScaledMatchingStreaming<BasicGreedyMatchingStreamingDisk,
                                      FullBMatchingOperator<StandardFromDiskStreamableHypergraph>>;
using GreedyHalfAndGuaranteedScaledMatchingStreamingFullDisk =
    GreedyHalfAndGuaranteedScaledMatchingStreaming<
        BasicGreedyMatchingStreamingDisk,
        FullBMatchingOperator<StandardFromDiskStreamableHypergraph, std::array<double, 2>>>;
using GreedyHalfAndGuaranteedScaledMatchingStreamingGEQFullDisk =
    GreedyHalfAndGuaranteedScaledMatchingStreamingGEQ<
        BasicGreedyMatchingStreamingDisk,
        FullBMatchingOperator<StandardFromDiskStreamableHypergraph, std::array<double, 2>>>;
REGISTER_IMPL_NAMED(NaiveMatchingStreamingMatching, "naive");
REGISTER_IMPL_NAMED(NaiveMatchingStreamingMatchingDisk, "naive");

REGISTER_IMPL_NAMED(GreedyMatchingSetStreamingMatching, "greedy_set");
REGISTER_IMPL_NAMED(GreedyMatchingSetStreamingMatchingDisk, "greedy_set");
REGISTER_IMPL_NAMED(TwoGreedyMatchingSetStreamingMatching, "two_pass_greedy_set");
REGISTER_IMPL_NAMED(TwoGreedyMatchingSetStreamingMatchingDisk, "two_pass_greedy_set");
REGISTER_IMPL_NAMED(GreedyMatchingSetStreamingMatchingBestEvict, "best_evict");
REGISTER_IMPL_NAMED(GreedyMatchingSetStreamingMatchingBestEvictDisk, "best_evict");
REGISTER_IMPL_NAMED(TwoGreedyMatchingSetStreamingMatchingBestEvict, "two_pass_best_evict");
REGISTER_IMPL_NAMED(TwoGreedyMatchingSetStreamingMatchingBestEvictDisk, "two_pass_best_evict");
REGISTER_IMPL_NAMED(GreedyMatchingListStreamingMatching, "greedy_list");
REGISTER_IMPL_NAMED(TwoGreedyMatchingListStreamingMatching, "two_pass_greedy_list");
REGISTER_IMPL_NAMED(GreedyMatchingListStreamingMatchingDisk, "greedy_list");
REGISTER_IMPL_NAMED(TwoGreedyMatchingListStreamingMatchingDisk, "two_pass_greedy_list");
REGISTER_IMPL_NAMED(GreedyMatchingStreamingMatching, "greedy");
REGISTER_IMPL_NAMED(GreedyMatchingStreamingMatchingDisk, "greedy");
REGISTER_IMPL_NAMED(TwoPassGreedyMatchingStreamingMatching, "two_pass_greedy");
REGISTER_IMPL_NAMED(TwoPassGreedyMatchingStreamingMatchingDisk, "two_pass_greedy");
REGISTER_IMPL_NAMED(TwoPassGreedyMatchingStreamingMatchingOR, "two_pass_greedy_or");

REGISTER_IMPL_NAMED(GreedyMatchingStreamingMatchingFull, "greedy_full");
REGISTER_IMPL_NAMED(GreedyMatchingStreamingMatchingFullDisk, "greedy_full");

REGISTER_IMPL_NAMED(GreedyLogMatchingStreamingMatching, "greedy_log");
REGISTER_IMPL_NAMED(GreedyScaledMatchingStreamingMatching, "greedy_scaled");
REGISTER_IMPL_NAMED(GreedyScaledMatchingStreamingMatchingFull, "greedy_scaled_full");
REGISTER_IMPL_NAMED(GreedyHalfScaledMatchingStreamingMatching, "greedy_half_scaled");
REGISTER_IMPL_NAMED(GreedyHalfScaledMatchingStreamingMatchingDisk, "greedy_half_scaled");
REGISTER_IMPL_NAMED(GreedyHalfScaledMatchingStreamingMatchingFull, "greedy_half_scaled_full");
REGISTER_IMPL_NAMED(GreedyHalfScaledMatchingStreamingMatchingFullDisk, "greedy_half_scaled_full");
REGISTER_IMPL_NAMED(GreedyHalfAndGuaranteedScaledMatchingStreamingFull, "greedy_guaranteed_full");
REGISTER_IMPL_NAMED(GreedyHalfAndGuaranteedScaledMatchingStreamingGEQFull,
                    "greedy_guaranteed_geq_full");
REGISTER_IMPL_NAMED(GreedyQuadratureHalfScaledMatchingStreamingFull, "greedy_quadrature_full");
REGISTER_IMPL_NAMED(GreedyQuadratureHalfScaledMatchingStreamingNormal, "greedy_quadrature");
REGISTER_IMPL_NAMED(TwoPassGreedyQuadratureHalfScaledMatchingStreamingNormal,
                    "two_pass_greedy_quadrature");
REGISTER_IMPL_NAMED(GreedyQuadratureHalfScaledMatchingStreamingFullDisk, "greedy_quadrature_full");
REGISTER_IMPL_NAMED(GreedyQuadratureHalfScaledMatchingStreamingNormalDisk, "greedy_quadrature");
REGISTER_IMPL_NAMED(TwoPassGreedyQuadratureHalfScaledMatchingStreamingNormalDisk,
                    "two_pass_greedy_quadrature");
REGISTER_IMPL_NAMED(TwoPassGreedyHalfScaledMatchingStreamingNormal, "two_pass_permissive");
REGISTER_IMPL_NAMED(TwoPassGreedyHalfScaledMatchingStreamingNormalDisk, "two_pass_permissive");
REGISTER_IMPL_NAMED(GreedyLenientHalfScaledMatchingStreamingMatchingDisk, "lenient");
REGISTER_IMPL_NAMED(GreedyLenientHalfScaledMatchingStreamingMatching, "lenient");
REGISTER_IMPL_NAMED(TwoGreedyLenientHalfScaledMatchingStreamingMatchingDisk, "two_pass_lenient");
REGISTER_IMPL_NAMED(TwoGreedyLenientHalfScaledMatchingStreamingMatching, "two_pass_lenient");
REGISTER_IMPL_NAMED(GreedyLenientHalfScaledMatchingStreamingMatchingFull, "lenient_full");
REGISTER_IMPL_NAMED(GreedyLenientHalfScaledMatchingStreamingMatchingDiskFull, "lenient_full");
REGISTER_IMPL_NAMED(GreedySlim, "greedy_slim");
}  // namespace heihgm::app::algorithms::streaming::matching
template <>
struct heihgm::ds::is_hypergraph_type<
    heihgm::app::algorithms::streaming::matching::BareHypergraph<size_t, int>> : std::true_type {
};
