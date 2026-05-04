#pragma once
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "graph_gpu.hpp"

namespace heihgm::external::sumac {
struct Matching {
  struct GraphGPUWrapper {
    using GraphElemType = int64_t;
    using NodeType = GraphElemType;
    using EdgeType = NodeType;
    using GraphWeightType = double;
    using WeightType = GraphWeightType;
    using InternalType = GraphGPU;
    // struct Helper {};
    GraphGPU* wrapped = nullptr;
    std::vector<GraphElemType> offsets;
    std::vector<GraphElemType> tails;
    std::vector<GraphWeightType> weights;
    auto currentNumEdges() { return tails.size(); }
    auto currentNumNodes() { return offsets.size() - 1; }
    template <class InputHypergraphType>
    explicit GraphGPUWrapper(const InputHypergraphType& ref) {
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

      GraphElemType offset = 0;
      for (auto& edges : directed_graph) {
        std::sort(edges.begin(), edges.end());
        offsets.push_back(offset);
        for (const auto [tail, w] : edges) {
          tails.push_back(tail);
          weights.push_back(w);
        }
        offset += edges.size();
      }
      offsets.push_back(offset);
      wrapped = new GraphGPU(directed_graph.size(), tails.size(), offsets.data(), weights.data(),
                             tails.data());
      wrapped->initComm();
      wrapped->move_batch_to_GPU(0, 0, 0);
      CudaDeviceSynchronize();
    }
    ~GraphGPUWrapper() {
      if (wrapped != nullptr) {
        delete wrapped;
      }
    }
  };

  using HypergraphType = GraphGPUWrapper;
  using Graph_t = HypergraphType;
  using NodeID_t = typename Graph_t::NodeType;
  using EdgeID_t = typename Graph_t::EdgeType;
  using NodeType = typename Graph_t::NodeType;
  using EdgeType = typename Graph_t::EdgeType;
  using WeightType = typename Graph_t::WeightType;
  std::unique_ptr<Graph_t> _g;
  Matching(std::unique_ptr<Graph_t> g) : _g(std::move(g)) {}
  Graph_t& instance() { return *_g; }

  const static constexpr std::string_view ds_name = "sumac_matching";
  bool valid() { return true; }

  size_t size() const { return _g->wrapped->size(); }

  int weight() const { return _g->wrapped->weight(); }

  int free_edges_size() { return 0; }
  double quality() const { return 0; }

  void save(std::string filename) {}
  void solve() { _g->wrapped->run_pointer_chase(); }
};
}  // namespace heihgm::external::sumac