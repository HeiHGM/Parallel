#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cstddef>
#include <utility>

#include "matching/parallel/cuda/common.h"
namespace heihgm::matching::parallel::cuda {
template <typename EdgeStatusType>
__global__ void checkIfEdgesFreeCREW(const EdgeStatusType* __restrict__ d_edgeStatus, const int m,
                                     bool* d_hasFreeEdges) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= m) return;

  if (d_edgeStatus[i] == heihgm::ds::parallel::EdgeStatus::Free) {
    // atomicOr((int*)d_hasFreeEdges, 1);
    *d_hasFreeEdges = true;
  }
}
// Adapted version for the CREW algorithms
template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void find_maximal_edges_kernel_CREW(NodeStatusType* d_node_status,
                                               const RefType* d_nodeOffsets,
                                               const EdgeType* d_NodeToPins,
                                               const EdgeStatusType* d_edgeStatus,
                                               const float* d_perm, EdgeType* d_selectedEdge,
                                               const int n) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n) return;

  if (d_node_status[node] == ds::parallel::NodeStatus::Active) {
    double local_max_weight = -DBL_MAX;
    EdgeType local_max_id = 0;

    // Inner loop: over neighbours/connected edges
    for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
      EdgeType adj_edge = d_NodeToPins[nx];

      if (d_edgeStatus[adj_edge] == heihgm::ds::parallel::EdgeStatus::Free) {
        if (d_perm[adj_edge] > local_max_weight) {
          local_max_weight = d_perm[adj_edge];
          local_max_id = adj_edge;
        }
      }
    }

    // Update nodes and pin counter for edges
    if (local_max_weight == -DBL_MAX) {
      d_node_status[node] = ds::parallel::NodeStatus::Deactivated;
    } else {
      d_selectedEdge[node] = local_max_id;
    }
  }
}

// Adapted version for the CREW algorithms
template <typename RefType, typename NodeType, typename EdgeType, typename EdgeStatusType>
__global__ void add_maximal_edges_to_matching_kernel_CREW(const EdgeType* d_selectedEdge,
                                                          const RefType* d_edgeOffsets,
                                                          EdgeStatusType* d_edgeStatus,
                                                          const NodeType* d_EdgeToPins,
                                                          const int m) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;

  if (edge < m && d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
    int pin_counter = 0;

    for (RefType ex = d_edgeOffsets[edge]; ex < d_edgeOffsets[edge + 1]; ex++) {
      NodeType adj_node = d_EdgeToPins[ex];
      if (d_selectedEdge[adj_node] == edge) {
        pin_counter++;
      }
    }

    if (pin_counter == (d_edgeOffsets[edge + 1] - d_edgeOffsets[edge])) {
      // Add edge to matching
      d_edgeStatus[edge] = heihgm::ds::parallel::EdgeStatus::InMatching;
    }
  }
}

template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void mark_nodes_newly_completed_kernel_CREW(const EdgeStatusType* d_edgeStatus,
                                                       const EdgeType* d_NodeToPins,
                                                       const RefType* d_nodeOffsets,
                                                       NodeStatusType* d_node_status, const int n) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n || d_node_status[node] != ds::parallel::NodeStatus::Active) return;

  for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
    EdgeType adj_edge = d_NodeToPins[nx];
    if (d_edgeStatus[adj_edge] == heihgm::ds::parallel::EdgeStatus::InMatching) {
      d_node_status[node] = ds::parallel::NodeStatus::NewlyCompleted;
      break;
    }
  }
}

template <typename RefType, typename NodeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void deactivate_blocked_edges_kernel_CREW(EdgeStatusType* d_edgeStatus,
                                                     const NodeType* d_EdgeToPins,
                                                     const RefType* d_edgeOffsets,
                                                     const NodeStatusType* d_node_status,
                                                     const int m) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;

  if (edge >= m || d_edgeStatus[edge] != heihgm::ds::parallel::EdgeStatus::Free) return;

  for (RefType ex = d_edgeOffsets[edge]; ex < d_edgeOffsets[edge + 1]; ex++) {
    NodeType adj_node = d_EdgeToPins[ex];
    if (d_node_status[adj_node] == ds::parallel::NodeStatus::NewlyCompleted) {
      d_edgeStatus[edge] = heihgm::ds::parallel::EdgeStatus::NeighborInMatching;
      break;
    }
  }
}
template <typename NodeStatusType>
__global__ void deactivate_blocked_nodes_kernel_CREW(NodeStatusType* d_node_status, const int n) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n) return;

  if (d_node_status[node] == ds::parallel::NodeStatus::NewlyCompleted) {
    // Deactivate Nodes
    d_node_status[node] = ds::parallel::NodeStatus::Deactivated;
  }
}
template <typename PMatching>
struct CREWModel {
  using RefType = typename PMatching::HypergraphType::RefType;
  using NodeType = typename PMatching::NodeType;
  using EdgeType = typename PMatching::EdgeType;
  using EdgeStatus = typename PMatching::EdgeStatusType;
  using NodeStatus = typename PMatching::NodeStatusType;

  RefType* d_nodeOffsets;
  RefType* d_edgeOffsets;
  NodeType* d_EdgeToPins;
  EdgeType* d_NodeToPins;

  EdgeStatus* d_edge_status;
  NodeStatus* d_node_status;

  // bool* d_hasFreeEdges;
  using PermType = float;

  float* d_perm;
  EdgeType* d_selectedEdge;

  bool h_hasFreeEdges = true;

  const size_t m;
  const size_t n;
  static constexpr int threadsPerBlock = 256;
  std::pair<int, int> kernelConfigEdges;
  std::pair<int, int> kernelConfigNodes;
  CREWModel(PMatching& instance)
      : m(instance.instance().initialNumEdges()), n(instance.instance().initialNumNodes()) {
    d_edge_status = instance.edgeStatusD();

    d_nodeOffsets = instance.instance().nodeOffsetsD();
    d_edgeOffsets = instance.instance().edgeOffsetsD();

    d_EdgeToPins = instance.instance().edgeToNodeD();
    d_NodeToPins = instance.instance().nodeToEdgeD();
    kernelConfigEdges = {threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock};
    kernelConfigNodes = {threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock};
    d_node_status = instance.nodeStatusD();
    d_selectedEdge = instance.selectedEdgeD();
    // Initialize d_active_nodes, d_newly_completed_nodes, d_selectedEdge
    // cudaMalloc(&d_hasFreeEdges, sizeof(bool));
    // cudaMemcpy(d_hasFreeEdges, &h_hasFreeEdges, sizeof(bool), cudaMemcpyHostToDevice);
    d_perm = instance.dPermD();
    // extra
  }
  float* permutationsD() { return d_perm; }
  bool do_round() {
    find_maximal_edges_kernel_CREW<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_node_status, d_nodeOffsets, d_NodeToPins, d_edge_status, d_perm, d_selectedEdge, n);

    add_maximal_edges_to_matching_kernel_CREW<<<kernelConfigEdges.second,
                                                kernelConfigEdges.first>>>(
        d_selectedEdge, d_edgeOffsets, d_edge_status, d_EdgeToPins, m);

    mark_nodes_newly_completed_kernel_CREW<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_edge_status, d_NodeToPins, d_nodeOffsets, d_node_status, n);

    deactivate_blocked_edges_kernel_CREW<<<kernelConfigEdges.second, kernelConfigEdges.first>>>(
        d_edge_status, d_EdgeToPins, d_edgeOffsets, d_node_status, m);

    deactivate_blocked_nodes_kernel_CREW<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_node_status, n);
    // Check if there are still free edges
    // cudaMemset(d_hasFreeEdges, 0, sizeof(bool));
    // checkIfEdgesFreeCREW<<<kernelConfigEdges.second, kernelConfigEdges.first>>>(d_edge_status, m,
    //                                                                         d_hasFreeEdges);
    // cudaMemcpy(&h_hasFreeEdges, d_hasFreeEdges, sizeof(bool), cudaMemcpyDeviceToHost);
    bool has_true = thrust::any_of(
        thrust::device, thrust::device_ptr<EdgeStatus>(d_edge_status),
        thrust::device_ptr<EdgeStatus>(d_edge_status) + m,
        ds::parallel::IsEdgeStatus<EdgeStatus>(heihgm::ds::parallel::EdgeStatus::Free));
    return has_true;
  }
  ~CREWModel() {
    // cudaFree(d_hasFreeEdges);
    // cudaFree(d_perm);
    // cudaFree(d_active_nodes);
    // cudaFree(d_selectedEdge);
    // cudaFree(d_newly_completed_nodes);
  }
};

}  // namespace heihgm::matching::parallel::cuda