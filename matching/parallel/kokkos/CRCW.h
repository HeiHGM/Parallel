#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "ds/parallel/matching.h"
namespace heihgm::matching::parallel::kokkos {

template <typename PMatching>
struct CRCWModel {
  using RefType = typename PMatching::HypergraphType::RefType;
  using NodeType = typename PMatching::NodeType;
  using EdgeType = typename PMatching::EdgeType;
  using EdgeStatus = typename PMatching::EdgeStatusType;
  using NodeStatus = typename PMatching::NodeStatusType;

  Kokkos::View<RefType*>& d_nodeOffsets;
  Kokkos::View<RefType*>& d_edgeOffsets;
  Kokkos::View<NodeType*>& d_EdgeToPins;
  Kokkos::View<EdgeType*>& d_NodeToPins;

  Kokkos::View<EdgeStatus*>& d_edge_status;
  Kokkos::View<NodeStatus*>& d_node_status;
  using PermType = float;
  Kokkos::View<PermType*>& d_perm;

  bool h_hasFreeEdges = true;
  Kokkos::View<int*>& d_pin_counter;

  const size_t m;
  const size_t n;
  static constexpr int threadsPerBlock = 256;

  CRCWModel(PMatching& instance)
      : m(instance.instance().initialNumEdges()),
        n(instance.instance().initialNumNodes()),
        d_edge_status(instance.edgeStatusD()),

        d_nodeOffsets(instance.instance().nodeOffsetsD()),
        d_edgeOffsets(instance.instance().edgeOffsetsD()),

        d_EdgeToPins(instance.instance().edgeToNodeD()),
        d_NodeToPins(instance.instance().nodeToEdgeD()),
        d_node_status(instance.nodeStatusD()),
        d_perm(instance.dPermD()),
        d_pin_counter(instance.pinCounterD()) {}
  Kokkos::View<float*>& permutationsD() { return d_perm; }
  bool do_round() {
    {
      TIMED_SCOPE(timer, "n");

      Kokkos::parallel_for(
          "Outer Loop over Nodes", n,
          KOKKOS_LAMBDA(const typename PMatching::HypergraphType::NodeType node) {
            if (d_node_status(node) ==
                ds::parallel::NodeStatus::Active) {  // Check, whether node is active
              float local_max_weight = std::numeric_limits<float>::lowest();
              EdgeType local_max_id = 0;

              // Inner loop: over neighbours/connected edges
              for (auto nx = d_nodeOffsets(node); nx < d_nodeOffsets(node + 1); nx++) {
                auto edge = d_NodeToPins(nx);
                if (d_edge_status(edge) ==
                    heihgm::ds::parallel::EdgeStatus::Free) {  // maybe unnessecary
                  // edge_counter++;
                  if (d_perm(edge) > local_max_weight) {
                    local_max_weight = d_perm(edge);  // update max edge weight
                    local_max_id = edge;              // update max edge id
                  }
                }
              }
              if (local_max_weight == std::numeric_limits<float>::lowest()) {
                d_node_status(node) = ds::parallel::NodeStatus::Deactivated;
              } else {  // Check if we found a new maximal
                        // edge or if it stayed the same
                Kokkos::atomic_inc(&d_pin_counter(local_max_id));
              }
            }
          });
    }
    Kokkos::fence();
    {
      TIMED_SCOPE(timer, "m");

      Kokkos::parallel_for(
          "Outer Loop over Edges", m,
          KOKKOS_LAMBDA(const typename PMatching::HypergraphType::EdgeType local_max_id) {
            // Instantly delete node if pin_counter is maximal
            if (d_edge_status(local_max_id) == heihgm::ds::parallel::EdgeStatus::Free &&
                d_pin_counter(local_max_id) ==
                    (d_edgeOffsets(local_max_id + 1) - d_edgeOffsets(local_max_id))) {
              for (auto ex = d_edgeOffsets(local_max_id); ex < d_edgeOffsets(local_max_id + 1);
                   ex++) {
                auto adj_node = d_EdgeToPins(ex);
                // Deactivate Nodes
                d_node_status(adj_node) = ds::parallel::NodeStatus::NewlyCompleted;
              }

              d_edge_status(local_max_id) = heihgm::ds::parallel::EdgeStatus::InMatching;
            } else if (d_edge_status(local_max_id) == heihgm::ds::parallel::EdgeStatus::Free) {
              // std::cout << "Pins" << d_pin_counter(local_max_id) << std::endl;
            }
          });
    }
    Kokkos::fence();  // Fencing
    {
      TIMED_SCOPE(timer, "n");
      Kokkos::parallel_for(
          "Deactivate edges", n,
          KOKKOS_LAMBDA(const typename PMatching::HypergraphType::NodeType node) {
            if (d_node_status(node) != ds::parallel::NodeStatus::NewlyCompleted) {
              return;
            }
            for (RefType nx = d_nodeOffsets(node); nx < d_nodeOffsets(node + 1); nx++) {
              EdgeType adj_edge = d_NodeToPins[nx];

              // Add edge to matching
              if (d_edge_status(adj_edge) == heihgm::ds::parallel::EdgeStatus::Free) {
                d_edge_status(adj_edge) = heihgm::ds::parallel::EdgeStatus::NeighborInMatching;
              }
            }
            // Deactivate Nodes
            d_node_status(node) = ds::parallel::NodeStatus::Deactivated;
          });
    }
    // std::cout << Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(),
    //                                          Kokkos::Experimental::cbegin(d_edge_status),
    //                                          Kokkos::Experimental::cend(d_edge_status),
    //                                          heihgm::ds::parallel::EdgeStatus::Free)
    //           << std::endl;
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_pin_counter, 0);
    Kokkos::fence();
    return Kokkos::Experimental::any_of(
        Kokkos::DefaultExecutionSpace(), Kokkos::Experimental::cbegin(d_edge_status),
        Kokkos::Experimental::cend(d_edge_status),
        KOKKOS_LAMBDA(heihgm::ds::parallel::EdgeStatus i) {
          return i == heihgm::ds::parallel::EdgeStatus::Free;
        });
  }
  ~CRCWModel() {
    // cudaFree(d_hasFreeEdges);
  }
};

}  // namespace heihgm::matching::parallel::kokkos