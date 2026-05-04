#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cstddef>
#include <utility>

#include "matching/parallel/cuda/common.h"
namespace heihgm::matching::parallel::cuda {
template <typename EdgeStatusType>
__global__ void checkIfEdgesFree(const EdgeStatusType* __restrict__ d_edgeStatus, const int m,
                                 bool* d_hasFreeEdges) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= m) return;

  if (d_edgeStatus[i] == heihgm::ds::parallel::EdgeStatus::Free) {
    // atomicOr((int*)d_hasFreeEdges, 1);
    *d_hasFreeEdges = true;
  }
}

template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void find_maximal_edges_kernel_crcw(
    NodeStatusType* d_node_status, const RefType* d_nodeOffsets, const EdgeType* d_NodeToPins,
    const EdgeStatusType* d_edgeStatus, const float* d_perm, int* d_pin_counter, const int n) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n) return;

  if (d_node_status[node] == ds::parallel::NodeStatus::Active) {
    double local_max_weight = -DBL_MAX;
    EdgeType local_max_id = 0;

    // Inner loop: over neighbours/connected edges
    for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
      EdgeType edge = d_NodeToPins[nx];

      if (d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
        if (d_perm[edge] > local_max_weight) {
          local_max_weight = d_perm[edge];
          local_max_id = edge;
        }
      }
    }

    // Update nodes and pin counter for edges
    if (local_max_weight == -DBL_MAX) {
      d_node_status[node] = ds::parallel::NodeStatus::Deactivated;
    } else {
      atomicAdd(&d_pin_counter[local_max_id], 1);
    }
  }
}

template <typename RefType, typename NodeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void add_maximal_edges_to_matching_kernel_crcw(const int* d_pin_counter,
                                                          const RefType* d_edgeOffsets,
                                                          EdgeStatusType* d_edgeStatus,
                                                          const NodeType* d_EdgeToPins, const int m,
                                                          NodeStatusType* d_node_status) {
  int local_max_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (local_max_id < m) {
    if (d_edgeStatus[local_max_id] == heihgm::ds::parallel::EdgeStatus::Free &&
        d_pin_counter[local_max_id] ==
            (d_edgeOffsets[local_max_id + 1] - d_edgeOffsets[local_max_id])) {
      for (RefType ex = d_edgeOffsets[local_max_id]; ex < d_edgeOffsets[local_max_id + 1]; ex++) {
        NodeType adj_node = d_EdgeToPins[ex];

        // Mark nodes as newly completed
        d_node_status[adj_node] = ds::parallel::NodeStatus::NewlyCompleted;
      }
      // Add edge to matching
      d_edgeStatus[local_max_id] = heihgm::ds::parallel::EdgeStatus::InMatching;
    }
  }
}

template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void remove_nodes_and_edges_kernel_crcw(EdgeStatusType* d_edgeStatus,
                                                   const EdgeType* d_NodeToPins,
                                                   const RefType* d_nodeOffsets, const int n,
                                                   NodeStatusType* d_node_status) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;

  if (node >= n || d_node_status[node] != ds::parallel::NodeStatus::NewlyCompleted) return;

  for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
    EdgeType adj_edge = d_NodeToPins[nx];

    // Add edge to matching
    if (d_edgeStatus[adj_edge] == heihgm::ds::parallel::EdgeStatus::Free) {
      d_edgeStatus[adj_edge] = heihgm::ds::parallel::EdgeStatus::NeighborInMatching;
    }
  }
  // Deactivate Nodes
  d_node_status[node] = ds::parallel::NodeStatus::Deactivated;
}

template <typename PMatching>
struct CRCWModel {
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
  using PermType = float;
  PermType* d_perm;

  bool h_hasFreeEdges = true;
  int* d_pin_counter;

  const size_t m;
  const size_t n;
  static constexpr int threadsPerBlock = 256;
  std::pair<int, int> kernelConfigEdges;
  std::pair<int, int> kernelConfigNodes;
  CRCWModel(PMatching& instance)
      : m(instance.instance().initialNumEdges()), n(instance.instance().initialNumNodes()) {
    d_edge_status = instance.edgeStatusD();

    d_nodeOffsets = instance.instance().nodeOffsetsD();
    d_edgeOffsets = instance.instance().edgeOffsetsD();

    d_EdgeToPins = instance.instance().edgeToNodeD();
    d_NodeToPins = instance.instance().nodeToEdgeD();
    kernelConfigEdges = {threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock};
    kernelConfigNodes = {threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock};
    d_node_status = instance.nodeStatusD();
    // cudaMalloc(&d_hasFreeEdges, sizeof(bool));
    // cudaMemcpy(d_hasFreeEdges, &h_hasFreeEdges, sizeof(bool), cudaMemcpyHostToDevice);
    d_perm = instance.dPermD();
    // extra
    d_pin_counter = instance.pinCounterD();
  }
  float* permutationsD() { return d_perm; }
  bool do_round() {
    cudaMemset(d_pin_counter, 0,
               m * sizeof(int));  // Reset pin counter since the algorithm allocated new weights
    find_maximal_edges_kernel_crcw<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_node_status, d_nodeOffsets, d_NodeToPins, d_edge_status, d_perm, d_pin_counter, n);

    add_maximal_edges_to_matching_kernel_crcw<<<kernelConfigEdges.second,
                                                kernelConfigEdges.first>>>(
        d_pin_counter, d_edgeOffsets, d_edge_status, d_EdgeToPins, m, d_node_status);

    remove_nodes_and_edges_kernel_crcw<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_edge_status, d_NodeToPins, d_nodeOffsets, n, d_node_status);
    // Check if there are still free edges
    // cudaMemset(d_hasFreeEdges, 0, sizeof(bool));
    // checkIfEdgesFree<<<kernelConfigEdges.second, kernelConfigEdges.first>>>(d_edge_status, m,
    //                                                                         d_hasFreeEdges);
    // cudaMemcpy(&h_hasFreeEdges, d_hasFreeEdges, sizeof(bool), cudaMemcpyDeviceToHost);
    bool has_true = thrust::any_of(
        thrust::device, thrust::device_ptr<EdgeStatus>(d_edge_status),
        thrust::device_ptr<EdgeStatus>(d_edge_status) + m,
        ds::parallel::IsEdgeStatus<EdgeStatus>(heihgm::ds::parallel::EdgeStatus::Free));
    return has_true;
  }
  ~CRCWModel() {
    // cudaFree(d_hasFreeEdges);
  }
};

}  // namespace heihgm::matching::parallel::cuda