/**
 * @file modifiable_hypergraph.h
 * @author Henrik Reinstädtler (reinstaedtler@stud.uni-heidelberg.de)
 * @brief
 * @version 0.1
 * @date 2022-05-10
 *
 * @copyright Copyright (c) 2022 Henrik Reinstädtler
 *
 */
#pragma once
#include <algorithm>
#include <cassert>
#include <vector>

#include "absl/strings/string_view.h"
#include "ds/flat_bipartite_impl.h"
#include "ds/weight_sorted_bipartite_impl.h"
#include "utils/range.h"

namespace heihgm {
namespace ds {
namespace {
using heihgm::utils::Range;
}
template <template <typename, typename, typename> class BipartiteGraphImpl =
              WeightSortedBipartiteGraphImpl_,
          typename EType = size_t, typename NType = size_t, typename WType = int>
class ModifiableHypergraph {
 public:
  using EdgeType = EType;
  using NodeType = NType;
  using WeightType = WType;

 private:
  FlatBipartiteGraphImpl_<EdgeType, NodeType, WType> _edges;
  BipartiteGraphImpl<NodeType, EdgeType, WType> _nodes;
  size_t edges_count = 0;

 public:
  const static constexpr absl::string_view ds_name =
      "modifiable_hypergraph";  // TODO(reinstaedtler): use this for auto name gen.
  ModifiableHypergraph(size_t nEdges, size_t nNodes, bool nodesInEdgesSorted = true,
                       bool edgesInNodesSorted = true)
      : _edges(nEdges, nodesInEdgesSorted), _nodes(nNodes, _edges.weights()), edges_count(nEdges) {}
  // Public interface
  auto pins(EdgeType e) const { return _edges.included(e); }
  Range<typename std::vector<EdgeType>::const_iterator> incidentEdges(NodeType n) const {
    return _nodes.included(n);
  }
  auto edges() const { return _edges.all(); }
  auto nodes() const { return _nodes.all(); }
  auto edgeSize(EdgeType e) const { return _edges.size(e); }
  auto nodeDegree(NodeType n) const { return _nodes.size(n); }
  // auto establishConnection(EdgeType e, NodeType n) {
  //   _edges.addTo(e, n);
  //   _nodes.addTo(n, e);
  // }
  template <typename C>
  auto addEdge(const C& nodes, WeightType w) {
    auto e = _edges.add(nodes, w);
    for (auto n : nodes) {
      _nodes.addTo(n, e);
    }
    return e;
  }
  auto addEdge(NodeType u, NodeType v, WeightType w) {
    return addEdge(std::vector<NodeType>({u, v}), w);
  }
  auto edgeWeight(EdgeType e) const { return _edges.weight(e); }
  auto nodeWeight(NodeType n) const { return _nodes.weight(n); }
  auto capacity_old(NodeType n) const { return nodeWeight(n); }
  void setNodeWeight(NodeType n, WeightType w) { _nodes.setWeight(n, w); }
  void setEdgeWeight(EdgeType n, WeightType w) { _edges.setWeight(n, w); }
  size_t currentNumEdges() const { return _edges.enabledCount(); }
  size_t currentNumNodes() const { return _nodes.enabledCount(); }
  size_t initialNumEdges() const { return edges_count; }
  size_t initialNumNodes() const { return _nodes.initialCount(); }
  void removeNode(NodeType n) {
    for (auto e : _nodes.included(n)) {
      _edges.disableIn(e, n);
    }
    _nodes.changeEnablement(n);
  }
  void removeNode_sorted(NodeType n) {
    assert(!_nodes.disabled(n));
    for (auto e : incidentEdges(n)) {
      _edges.disableIn_sorted(e, n);
    }
    _nodes.changeEnablement(n);
  }
  void removeEdge(EdgeType e) {
    for (auto n : _edges.included(e)) {
      _nodes.disableIn(n, e);
    }
    _edges.changeEnablement(e);
  }
  void removeEdge_sorted(EdgeType e) {
    assert(!_edges.disabled(e));
    for (auto n : _edges.included(e)) {
      _nodes.disableIn_sorted(n, e);
    }
    _edges.changeEnablement(e);
  }
  void enableEdge(EdgeType e) {
    _edges.changeEnablement(e);
    // update edges accordingly
    auto last = _edges.included(e).end();
    for (auto n = _edges.included(e).begin(); n < last; n++) {
      if (!_nodes.disabled(*n)) {
        _nodes.enableIn(*n, e);
      } else {
        _edges.disableIn(e, *n);
        last = _edges.included(e).end();
      }
    }
  }
  void resortEdge(EdgeType e) { _edges.resort(e); }
  void enableEdge_sorted(EdgeType e) {
    _edges.changeEnablement(e);
    _edges.resort(e);
    // update edges accordingly
    auto last = _edges.included(e).end();
    std::vector<NodeType> to_disable;
    for (auto n = _edges.included(e).begin(); n < last; n++) {
      if (!_nodes.disabled(*n)) {
        _nodes.enableIn_sorted(*n, e);
      } else {
        to_disable.push_back(*n);
      }
    }
    for (auto v : to_disable) {
      _edges.disableIn_sorted(e, v);
    }
  }
  bool edgeIsEnabled(EdgeType e) const { return !_edges.disabled(e); }
  bool nodeIsEnabled(NodeType e) const { return !_nodes.disabled(e); }
  const std::vector<WeightType>& edgeWeights() const { return _edges.weights(); }
  const auto& containedNodes() { return _edges.containedNodes(); }
  const auto containedEdges() { return _nodes.contained(); }
  auto edgeOffests() const { return _edges.offsets(); }

  // assumes a \cap b = \emptyset
  auto mergeEdges_sorted(EdgeType a, EdgeType b) {
    // TODO(henrik): better performance
    // TODO replace in linear work
    for (auto p : pins(b)) {
      _nodes.replace_sorted(p, b, a);
    }
    return _edges.merge_sorted(a, b);
  }
  void unmergeEdge(EdgeType a, EdgeType b, size_t& removals, std::vector<NodeType>& replaced_at) {
    _edges.unmerge(a, b, removals);
    for (auto p : replaced_at) {
      _nodes.replace(p, a, b);
    }
    for (auto p : pins(b)) {
      if (_nodes.disabled(p)) {
        _nodes.changeEnablement(p);
      }
      if (!_nodes.contains(p, b)) {
        _nodes.enableIn(p, b);
      }
    }
  }
  bool isSorted() const { return _nodes.isSorted(); }
  auto realEdgeSize() const { return _edges.count(); }
  bool unmergeEdge_sorted(EdgeType a, EdgeType b) {
    bool res = _edges.unmerge_sorted(a, b);
    std::vector<NodeType> todisable;
    for (auto p : pins(b)) {
      if (!_nodes.disabled(p)) {
        if (_nodes.contains(p, a)) {
          _nodes.replace_sorted(p, a, b);  // maybe !_nodes.contains(p, b)
        }
      } else {
        todisable.push_back(p);
      }
    }
    for (auto p : todisable) {
      _edges.disableIn_sorted(b, p);
    }
    return res;
  }
  void sort() {
    _nodes.sort();  // not needed (always) as sort is already implicit because of
    // reading
    _edges.sort();
  }
  void finish() { sort(); }
  void sort_weight() { _nodes.sort(); }
  EdgeType addMergeEdge(EdgeType initial, WeightType w) {
    assert(edgeIsEnabled(initial));
    std::vector<NodeType> nodes(pins(initial).begin(), pins(initial).end());
    auto new_edge = _edges.add(nodes, w);
    for (auto v : nodes) {
      _nodes.replace_sorted(v, initial, new_edge);
    }
    _edges.changeEnablement(initial);
    return new_edge;
  }
  bool edgeIsInPin(EdgeType e, NodeType n) { return _nodes.isIn(e, n); }
  void shrink_to_size() { exit(1); }
  size_t pinCount() const {
    size_t result = 0;
    for (auto e : edges()) {
      result += edgeSize(e);
    }
    return result;
  }
  void printInfo() {
    std::cout << "InitialNumEdges: " << initialNumEdges() << std::endl
              << "InitialNumNodes: " << initialNumNodes() << std::endl;
  }
  void enableAll() {
    _nodes.enableAll();
    _edges.enableAll();
  }
};
using StandardIntegerHypergraph = ModifiableHypergraph<>;
/**
 * @brief Compability function to convert from kahypar interface Graph
 *
 * @tparam ModifiableHypergraph
 * @tparam Hypergraph
 * @param h
 * @return ModifiableHypergraph
 */
template <class ModifiableHypergraph, class Hypergraph>
ModifiableHypergraph convertFrom(Hypergraph& h) {
  ModifiableHypergraph m(h.initialNumEdges(), h.initialNumNodes());
  for (auto e : h.edges()) {
    for (auto p : h.pins(e)) {
      m.establishConnection(e, p);
    }
    m.setEdgeWeight(e, h.edgeWeight(e));
  }
  for (auto n : h.nodes()) {
    m.setNodeWeight(n, h.nodeWeight(n));
  }
  return m;
}
template <typename T>
struct is_hypergraph_type : std::false_type {};
template <>
struct is_hypergraph_type<ModifiableHypergraph<>> : std::true_type {};
}  // namespace ds
}  // namespace heihgm