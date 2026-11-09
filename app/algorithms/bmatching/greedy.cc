#include <memory>
#include <vector>

#include "app/algorithms/algorithm_impl.h"
#include "utils/range.h"

namespace heihgm::app::algorithms {

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
  auto edgeWeight(size_t e) { return edges_[e].first; }
};
template <class HT>
struct SimpleBMatching {
  using EdgeType1 = typename HT::EdgeType1;
  using EdgeType = typename HT::EdgeType;
  std::unique_ptr<HT> graph;
  std::vector<int> matchedAt;
  std::vector<int> capacity;
  std::vector<typename HT::EdgeType1> result;
  bool isMatchable(const EdgeType1& edge) {
    for (const auto p : edge.second) {
      if (matchedAt[p] >= capacity[p]) {
        return false;
      }
    }
    return true;
  }
  void addToMatching(const EdgeType1& edge) {
    result.push_back(edge);
    for (const auto p : edge.second) {
      matchedAt[p]++;
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
    std::fill(matchedAt.begin(), matchedAt.end(), 0);
    for (const auto& r : result) {
      if (!isMatchable(r)) {
        return false;
      }
      for (auto p : r.second) {
        matchedAt[p]++;
      }
    }
    return true;
  }
  int64_t size() const { return result.size(); }
  double quality() const { return 0; }
  int64_t free_edges_size() const { return 0; }
  const HT& instance() const { return *graph; }
  static constexpr absl::string_view ds_name = "simple_bmatching";
  explicit SimpleBMatching(std::unique_ptr<HT> hg, int cap)
      : graph(std::move(hg)),
        matchedAt(graph->currentNumNodes(), false),
        capacity(graph->currentNumNodes(), cap) {
    if (cap == -1) {
      for (auto e : graph->edges()) {
        capacity[e] = graph->edgeWeight(e);
      }
    }
  }
};
using SimpleMatchingT = SimpleBMatching<BareHypergraph<size_t, int>>;
struct BGreedySlim : AlgorithmImpl<SimpleMatchingT> {
  absl::StatusOr<std::unique_ptr<SimpleMatchingT>> Load(const app_io::RunConfig& run_config,
                                                        const app_io::Hypergraph& hypergraph) {
    auto hgr = loadFile<BareHypergraph<size_t, int>>(hypergraph);
    auto gr = std::move(hgr.value());
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
    return std::make_unique<SimpleMatchingT>(std::move(gr), run_config.capacity());
  }
  absl::StatusOr<std::unique_ptr<SimpleMatchingT>> Execute(
      const AlgorithmConfig& config, std::unique_ptr<SimpleMatchingT> matching) {
    std::vector<typename SimpleMatchingT::EdgeType> order;
    for (auto e : matching->instance().edges()) {
      order.push_back(e);
    }
    // std::random_shuffle(order.begin(), order.end());
    auto eval_function = [&](auto e) {
      return std::make_pair(matching->graph->edges_[e].first, -((int64_t)e));
    };
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
REGISTER_IMPL_NAMED(BGreedySlim, "greedy_slim");
}  // namespace heihgm::app::algorithms
template <>
struct heihgm::ds::is_hypergraph_type<heihgm::app::algorithms::BareHypergraph<size_t, int>>
    : std::true_type {};