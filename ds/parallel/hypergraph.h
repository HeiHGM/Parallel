#include <Kokkos_Core.hpp>

#include "utils/logging.h"
#include "utils/range.h"
namespace heihgm {
namespace ds {
namespace parallel {
template <typename NT, typename ET, typename RT, typename WT>
struct KokkosHypergraph {
  using NodeType = NT;
  using EdgeType = ET;
  using RefType = RT;
  using WeightType = WT;
  Kokkos::View<RefType*> node_offsets;
  Kokkos::View<RefType*> edge_offsets;
  Kokkos::View<WeightType*> edge_weights;

  Kokkos::View<NodeType*> edge_to_node;
  Kokkos::View<EdgeType*> node_to_edge;
  Kokkos::View<RefType*, Kokkos::HostSpace> Hedge_offsets;
  Kokkos::View<WeightType*, Kokkos::HostSpace> Hedge_weights;
  Kokkos::View<RefType*, Kokkos::HostSpace> Hnode_offsets;

  Kokkos::View<NodeType*, Kokkos::HostSpace> Hedge_to_node;
  Kokkos::View<EdgeType*, Kokkos::HostSpace> Hnode_to_edge;
  size_t m, n;

 public:
  KokkosHypergraph(size_t n, size_t m, size_t pins)
      : node_offsets("node_offset", n + 1),
        edge_offsets("edge_offsets", m + 1),
        edge_weights("edge_weights", m),
        edge_to_node("edge_reps", pins),
        node_to_edge("node_reps", pins),
        Hedge_offsets("Hedge_offset", m + 1),
        Hnode_offsets("Hnode_offset", n + 1),
        Hedge_weights("hwe", m),
        Hedge_to_node("hedge2node", pins),
        Hnode_to_edge("Hnode2edge", pins),
        m(m),
        n(n) {}
  template <class Hypergraph>
  explicit KokkosHypergraph(const Hypergraph& ref)
      : KokkosHypergraph(ref.initialNumNodes(), ref.initialNumEdges(), ref.pinCount()) {
    TIMED_FUNC(timer);
    size_t pincount = ref.pinCount();

    RefType offset = 0;
    for (auto e : ref.edges()) {
      Hedge_weights(e) = ref.edgeWeight(e);
      Hedge_offsets(e) = offset;
      for (auto p : ref.pins(e)) {
        Hedge_to_node(offset++) = p;
      }
    }
    Hedge_offsets(ref.initialNumEdges()) = offset;
    offset = 0;
    for (auto n : ref.nodes()) {
      Hnode_offsets(n) = offset;
      for (auto p : ref.incidentEdges(n)) {
        Hnode_to_edge(offset++) = p;
      }
    }
    Hnode_offsets(ref.initialNumNodes()) = offset;

    Kokkos::deep_copy(node_to_edge, Hnode_to_edge);
    Kokkos::deep_copy(edge_to_node, Hedge_to_node);
    Kokkos::deep_copy(edge_weights, Hedge_weights);
    Kokkos::deep_copy(edge_offsets, Hedge_offsets);
    Kokkos::deep_copy(node_offsets, Hnode_offsets);
  }
  auto pins(EdgeType e) {
    return heihgm::utils::PointerRange(Hedge_to_node.data() + Hedge_offsets(e),
                                       Hedge_to_node.data() + Hedge_offsets(e + 1));
  }
  auto incidentEdges(NodeType n) {
    return heihgm::utils::PointerRange(Hnode_to_edge.data() + Hnode_offsets(n),
                                       Hnode_to_edge.data() + Hnode_offsets(n + 1));
  }
  auto edgeSize(EdgeType e) const { return Hedge_offsets(e + 1) - Hedge_offsets(e); }
  auto edgeWeight(EdgeType e) const { return Hedge_weights(e); }
  void setEdgeWeight(EdgeType edge, WeightType weight) {
    if (edge < 0 || edge >= Hedge_weights.extent(0)) {
      printf("Error: Edge index out of bounds.\n");
      return;
    }
    Hedge_weights(edge) = weight;
    Kokkos::deep_copy(edge_weights, Hedge_weights);
  }
  auto initialNumEdges() const { return m; }
  auto initialNumNodes() const { return n; }
  auto currentNumEdges() const { return m; }
  auto currentNumNodes() const { return n; }
  auto edges() const { return heihgm::utils::PointerRange<utils::SimpleIterator>(0UL, m); }
  auto& nodeOffsetsD() { return node_offsets; }
  auto& edgeOffsetsD() { return edge_offsets; }
  auto& edgeToNodeD() { return edge_to_node; }
  auto& nodeToEdgeD() { return node_to_edge; }
  auto& edgeWeightD() { return edge_weights; }
};
}  // namespace parallel
}  // namespace ds
}  // namespace heihgm