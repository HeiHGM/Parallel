#pragma once
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "benchmarks/MaximalMatching/RandomGreedy/MaximalMatching.h"
#include "gbbs/gbbs.h"

namespace heihgm::external::gbbs {
struct Matching {
  struct Wrapper {
    using GraphElemType = ::gbbs::uintT;
    using NodeType = GraphElemType;
    using EdgeType = ::gbbs::uintE;
    using GraphWeightType = float;
    using WeightType = GraphWeightType;

    // struct Helper {};
    ::gbbs::symmetric_graph<::gbbs::symmetric_vertex, GraphWeightType> graph_;
    size_t n, m;

    auto currentNumEdges() { return m / 2; }
    auto currentNumNodes() { return n; }
    template <class InputHypergraphType>
    explicit Wrapper(const InputHypergraphType& ref) {
      std::vector<std::vector<std::pair<GraphElemType, GraphWeightType>>> directed_graph(
          ref.initialNumNodes());

      for (auto e : ref.edges()) {
        if (ref.edgeSize(e) != 2) {
          throw;
        }
        std::vector<GraphElemType> pins;
        for (auto p : ref.pins(e)) {
          pins.push_back(p);
        }
        directed_graph[pins[0]].push_back(
            std::make_pair(pins[1], GraphWeightType{ref.edgeWeight(e)}));
        directed_graph[pins[1]].push_back(
            std::make_pair(pins[0], GraphWeightType{ref.edgeWeight(e)}));
      }
      n = directed_graph.size();
      GraphElemType* offsets = ::gbbs::new_array_no_init<GraphElemType>(n + 1);
      m = ref.currentNumEdges() * 2;
      using id_and_weight = std::tuple<EdgeType, GraphWeightType>;

      std::tuple<EdgeType, GraphWeightType>* edges =
          ::gbbs::new_array_no_init<id_and_weight>(2 * m);
      GraphElemType offset = 0;
      size_t i = 0;
      for (auto& edges_ : directed_graph) {
        std::sort(edges_.begin(), edges_.end());
        offsets[i] = offset;
        for (auto [tail, w] : edges_) {
          edges[offset++] = std::make_tuple(tail, w);
        }
        i++;
      }
      offsets[i] = offset;

      auto v_data = ::gbbs::new_array_no_init<::gbbs::vertex_data>(n);
      ::gbbs::parallel_for(0, n, [&](size_t i) {
        v_data[i].offset = offsets[i];
        v_data[i].degree = offsets[i + 1] - v_data[i].offset;
      });

      ::gbbs::free_array(offsets, n + 1);

      graph_ = ::gbbs::symmetric_graph<::gbbs::symmetric_vertex, WeightType>(
          v_data, n, m,
          [=]() {
            ::gbbs::free_array(v_data, n);
            ::gbbs::free_array(edges, m);
          },
          edges);
    }
    ~Wrapper() {}
  };

  using HypergraphType = Wrapper;
  using Graph_t = HypergraphType;
  using NodeID_t = typename Graph_t::NodeType;
  using EdgeID_t = typename Graph_t::EdgeType;
  using NodeType = typename Graph_t::NodeType;
  using EdgeType = typename Graph_t::EdgeType;
  using WeightType = typename Graph_t::WeightType;
  std::unique_ptr<Graph_t> _g;
  ::gbbs::sequence<std::tuple<EdgeType, EdgeType, WeightType>> solution;
  Matching(std::unique_ptr<Graph_t> g) : _g(std::move(g)) {}
  Graph_t& instance() { return *_g; }

  const static constexpr std::string_view ds_name = "sumac_matching";
  bool valid() { return true; }

  size_t size() const { return solution.size(); }

  int weight() const {
    WeightType w = 0;
    for (const auto [_1, _2, we] : solution) {
      w += we;
    }
    return w;
  }

  int free_edges_size() { return 0; }
  double quality() const { return 0; }

  void save(std::string filename) {}
  void solve() { solution = ::gbbs::MaximalMatching(_g->graph_); }
};
}  // namespace heihgm::external::gbbs