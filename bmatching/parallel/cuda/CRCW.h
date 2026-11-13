#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/sort.h>

#include <cstddef>
#include <cub/cub.cuh>
#include <utility>

#include "ds/parallel/bmatching.h"

namespace {

// Device helper function to swap two elements
template <typename T>
__device__ void swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}
#define MAX_STACK_SIZE \
  64  // Bounded to 64. Since we push the larger sub-array first,
      // stack size is mathematically guaranteed to be < 2 * log2(N).
// In-place partition scheme (Lomuto partition)
template <typename T>
__device__ int partition(T* arr, int low, int high, T* prio) {
  int pivot = prio[arr[high]];
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++) {
    T curr = prio[arr[j]];
    if (curr > pivot || (curr == pivot && arr[j] < arr[high])) {
      i++;
      swap(arr[i], arr[j]);
    }
  }
  swap(arr[i + 1], arr[high]);
  return (i + 1);
}
template <typename T>
__device__ void cudaQuicksort2(T* data, int n, T* prio) {
  // Explicit stack to store 'low' and 'high' boundaries
  int stack[MAX_STACK_SIZE];
  int top = -1;

  // Push initial boundaries of the array
  stack[++top] = 0;
  stack[++top] = n - 1;

  // Process the stack
  while (top >= 0) {
    // Pop high and low
    int h = stack[top--];
    int l = stack[top--];

    // Find the pivot index
    int p = partition(data, l, h, prio);

    // Optimization: Push the larger partition onto the stack first.
    // This guarantees the stack depth never exceeds O(log N) depth.
    if (p - 1 > l && p + 1 < h) {
      if ((p - 1 - l) < (h - (p + 1))) {
        // Right side is larger: push right side first, then left
        stack[++top] = p + 1;
        stack[++top] = h;
        stack[++top] = l;
        stack[++top] = p - 1;
      } else {
        // Left side is larger: push left side first, then right
        stack[++top] = l;
        stack[++top] = p - 1;
        stack[++top] = p + 1;
        stack[++top] = h;
      }
    } else {
      // Only one side needs sorting
      if (p - 1 > l) {
        stack[++top] = l;
        stack[++top] = p - 1;
      }
      if (p + 1 < h) {
        stack[++top] = p + 1;
        stack[++top] = h;
      }
    }
  }
}

template <typename T>
__global__ void cudaQuicksort(T* data, T* offsets, size_t n, T* prio) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n || offsets[idx + 1] == offsets[idx]) {
    return;
  }
  cudaQuicksort2(data + offsets[idx], offsets[idx + 1] - offsets[idx], prio);
}
}  // namespace
namespace heihgm::bmatching::parallel::cuda {
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
template <typename EdgeStatusType>
__global__ void read_n_set_priority(EdgeStatusType* __restrict__ d_prios,
                                    const EdgeStatusType* __restrict__ d_array,
                                    const EdgeStatusType* __restrict__ prio, const int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= m) return;
  d_prios[i] = prio[d_array[i]];
}
template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void vote_edges_kernel_crcw(NodeStatusType* d_node_status, const RefType* d_nodeOffsets,
                                       const EdgeType* d_NodeToPins, EdgeStatusType* d_edgeStatus,
                                       const int* d_residual, int* d_pin_counter, const int n,
                                       RefType* d_negativeAtVertexOffset) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n || d_node_status[node] != ds::parallel::NodeStatus::Active) return;
  int residual_capacity = d_residual[node];

  if (residual_capacity <= 0) {
    for (RefType nx = d_nodeOffsets[node + 1] - d_negativeAtVertexOffset[node];
         nx < d_nodeOffsets[node + 1]; nx++) {
      EdgeType edge = d_NodeToPins[nx];

      if (d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
        d_edgeStatus[edge] = heihgm::ds::parallel::EdgeStatus::JustBlocked;
      }
    }
    return;
  }
  // Inner loop: over neighbours/connected edges
  for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
    EdgeType edge = d_NodeToPins[nx];

    if (d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
      atomicAdd(&d_pin_counter[edge], 1);
      residual_capacity--;
    }
    if (residual_capacity == 0) {
      break;
    }
  }
}
template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void vote_edges_kernel_crcw_single(NodeStatusType* d_node_status,
                                              const RefType* d_nodeOffsets,
                                              const EdgeType* d_NodeToPins,
                                              EdgeStatusType* d_edgeStatus, const int* d_residual,
                                              int* d_pin_counter, const int n,
                                              RefType* d_negativeAtVertexOffset) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= n || d_node_status[node] != ds::parallel::NodeStatus::Active) return;
  int residual_capacity = d_residual[node];

  if (residual_capacity <= 0) {
    for (RefType nx = d_nodeOffsets[node + 1] - d_negativeAtVertexOffset[node];
         nx < d_nodeOffsets[node + 1]; nx++) {
      EdgeType edge = d_NodeToPins[nx];

      if (d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
        d_edgeStatus[edge] = heihgm::ds::parallel::EdgeStatus::JustBlocked;
      }
    }
    return;
  }
  // Inner loop: over neighbours/connected edges
  for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
    EdgeType edge = d_NodeToPins[nx];

    if (d_edgeStatus[edge] == heihgm::ds::parallel::EdgeStatus::Free) {
      atomicAdd(&d_pin_counter[edge], 1);
      residual_capacity--;
      return;
    }
    if (residual_capacity == 0) {
      break;
    }
  }
}

template <typename RefType, typename NodeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void commit_maximal_edges_to_matching_kernel_crcw(
    const int* d_pin_counter, const RefType* d_edgeOffsets, EdgeStatusType* d_edgeStatus,
    const NodeType* d_EdgeToPins, const int m, NodeStatusType* d_node_status, int* d_residual) {
  int local_max_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (local_max_id < m) {
    if (d_edgeStatus[local_max_id] == heihgm::ds::parallel::EdgeStatus::Free &&
        d_pin_counter[local_max_id] ==
            (d_edgeOffsets[local_max_id + 1] - d_edgeOffsets[local_max_id])) {
      for (RefType ex = d_edgeOffsets[local_max_id]; ex < d_edgeOffsets[local_max_id + 1]; ex++) {
        NodeType adj_node = d_EdgeToPins[ex];

        // Mark nodes as newly completed
        d_node_status[adj_node] = ds::parallel::NodeStatus::NewlyCompleted;
        atomicAdd(&d_residual[adj_node], -1);
      }
      // Add edge to matching
      d_edgeStatus[local_max_id] = heihgm::ds::parallel::EdgeStatus::InMatching;
    }
  }
}

template <typename RefType>
__global__ void set_negative_offsets(RefType* neg_offsets, RefType* offsets, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) {
    return;
  }
  neg_offsets[id] = offsets[id + 1] - offsets[id];
}
template <typename RefType, typename EdgeType, typename EdgeStatusType, typename NodeStatusType>
__global__ void remove_nodes_and_edges_kernel_crcw(EdgeStatusType* d_edgeStatus,
                                                   const EdgeType* d_NodeToPins,
                                                   const RefType* d_nodeOffsets, const int n,
                                                   NodeStatusType* d_node_status, int* d_residual,
                                                   RefType* neg_off) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;

  if (node >= n || d_node_status[node] != ds::parallel::NodeStatus::NewlyCompleted) return;
  if (d_residual[node] > 0) {  // still open
    d_node_status[node] = ds::parallel::NodeStatus::Active;
    return;
  }
  int skipped = 0;
  bool continous = true;
  for (RefType nx = d_nodeOffsets[node]; nx < d_nodeOffsets[node + 1]; nx++) {
    EdgeType adj_edge = d_NodeToPins[nx];

    // Add edge to matching
    if (d_edgeStatus[adj_edge] == heihgm::ds::parallel::EdgeStatus::Free) {
      d_edgeStatus[adj_edge] = heihgm::ds::parallel::EdgeStatus::NeighborInMatching;
    }
    if (continous) {
      skipped += d_edgeStatus[adj_edge] != heihgm::ds::parallel::EdgeStatus::Free;
    }
    continous &= d_edgeStatus[adj_edge] != heihgm::ds::parallel::EdgeStatus::Free;
  }
  neg_off[node] -= skipped;
  // Deactivate Nodes
  d_node_status[node] = ds::parallel::NodeStatus::Deactivated;
}

template <typename PMatching, bool singleVote>
struct CRCWModel {
  using RefType = typename PMatching::HypergraphType::RefType;
  using NodeType = typename PMatching::NodeType;
  using EdgeType = typename PMatching::EdgeType;
  using WeightType = typename PMatching::WeightType;

  using EdgeStatus = typename PMatching::EdgeStatusType;
  using NodeStatus = typename PMatching::NodeStatusType;

  RefType *d_nodeOffsets, *h_nodeOffsets;
  RefType* d_edgeOffsets;
  RefType* d_negativeAtVertexOffset;
  NodeType* d_EdgeToPins;
  EdgeType* d_NodeToPins;
  EdgeType* h_NodeToPins;

  size_t pinsCount;
  EdgeType* d_edgePriority;
  WeightType* h_weight;
  // EdgeType* d_edges_by_priority;

  EdgeStatus* d_edge_status;
  NodeStatus* d_node_status;
  int* d_residual;

  bool h_hasFreeEdges = true;
  int* d_pin_counter;

  const size_t m;
  const size_t n;
  static constexpr int threadsPerBlock = 256;
  std::pair<int, int> kernelConfigEdges;
  std::pair<int, int> kernelConfigNodes;
  std::vector<EdgeType> priority_per_edge_;
  bool deviceWeight = false;
  CRCWModel(PMatching& instance, const std::vector<EdgeType>& priority_per_edge)
      : m(instance.instance().initialNumEdges()),
        n(instance.instance().initialNumNodes()),
        pinsCount(instance.instance().Pins()),
        priority_per_edge_(priority_per_edge) {
    d_edge_status = instance.edgeStatusD();

    d_nodeOffsets = instance.instance().nodeOffsetsD();
    h_nodeOffsets = instance.instance().nodeOffsetsH();
    d_edgeOffsets = instance.instance().edgeOffsetsD();

    d_EdgeToPins = instance.instance().edgeToNodeD();
    d_NodeToPins = instance.instance().nodeToEdgeD();
    kernelConfigEdges = {threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock};
    kernelConfigNodes = {threadsPerBlock, (n + threadsPerBlock - 1) / threadsPerBlock};
    d_node_status = instance.nodeStatusD();
    // cudaMalloc(&d_hasFreeEdges, sizeof(bool));
    // cudaMemcpy(d_hasFreeEdges, &h_hasFreeEdges, sizeof(bool), cudaMemcpyHostToDevice);
    d_residual = instance.dResidual();
    // extra
    d_pin_counter = instance.pinCounterD();
    cudaMalloc(&d_negativeAtVertexOffset, sizeof(RefType) * n);
    set_negative_offsets<RefType><<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
        d_negativeAtVertexOffset, d_nodeOffsets, n);
    if (priority_per_edge.size() > 0) {
      cudaMalloc(&d_edgePriority, sizeof(EdgeType) * m);
      cudaMemcpy(d_edgePriority, priority_per_edge.data(), sizeof(EdgeType) * m,
                 cudaMemcpyHostToDevice);
    } else {
      d_edgePriority = instance.instance().edgeWeightD();
      deviceWeight = true;
    }
    // cudaMalloc(&d_edges_by_priority, sizeof(EdgeType) * m);
    // cudaMemcpy(d_edges_by_priority, edges_by_priority.data(), sizeof(EdgeType) * m,
    //            cudaMemcpyHostToDevice);
    h_NodeToPins = instance.instance().hNodeToPins();
    h_weight = instance.instance().edgeWeightH();
  }
  bool do_round(size_t start, size_t end) {
    TIMED_SCOPE(bla, "round")

    start = 0;
    end = m;
    cudaMemset(d_pin_counter, 0,
               m * sizeof(int));  // Reset pin counter
    cudaDeviceSynchronize();
    if (singleVote) {
      TIMED_SCOPE(bla, "single_vote")

      vote_edges_kernel_crcw_single<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
          d_node_status, d_nodeOffsets, d_NodeToPins, d_edge_status, d_residual, d_pin_counter, n,
          d_negativeAtVertexOffset);
    } else {
      TIMED_SCOPE(bla, "vote")

      vote_edges_kernel_crcw<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
          d_node_status, d_nodeOffsets, d_NodeToPins, d_edge_status, d_residual, d_pin_counter, n,
          d_negativeAtVertexOffset);
    }
    cudaDeviceSynchronize();
    {
      TIMED_SCOPE(bla, "commit")
      commit_maximal_edges_to_matching_kernel_crcw<<<kernelConfigEdges.second,
                                                     kernelConfigEdges.first>>>(
          d_pin_counter, d_edgeOffsets, d_edge_status, d_EdgeToPins, m, d_node_status, d_residual);
    }
    cudaDeviceSynchronize();

    {
      TIMED_SCOPE(bla, "removed")

      remove_nodes_and_edges_kernel_crcw<<<kernelConfigNodes.second, kernelConfigNodes.first>>>(
          d_edge_status, d_NodeToPins, d_nodeOffsets, n, d_node_status, d_residual,
          d_negativeAtVertexOffset);
    }
    cudaDeviceSynchronize();

    // Check if there are still free edges
    // cudaMemset(d_hasFreeEdges, 0, sizeof(bool));
    // checkIfEdgesFree<<<kernelConfigEdges.second, kernelConfigEdges.first>>>(d_edge_status, m,
    //                                                                         d_hasFreeEdges);
    // cudaMemcpy(&h_hasFreeEdges, d_hasFreeEdges, sizeof(bool), cudaMemcpyDeviceToHost);
    {
      TIMED_SCOPE(bla, "true")
      int has_true = thrust::any_of(
          thrust::device, thrust::device_ptr<EdgeStatus>(d_edge_status) + start,
          thrust::device_ptr<EdgeStatus>(d_edge_status) + end,
          ds::parallel::IsEdgeStatus<EdgeStatus>(heihgm::ds::parallel::EdgeStatus::Free));
      return has_true;
    }
  }
  ~CRCWModel() {
    if (!deviceWeight) {
      cudaFree(d_edgePriority);
    }
    // cudaFree(d_edges_by_priority);
    cudaFree(d_negativeAtVertexOffset);
    // cudaFree(d_hasFreeEdges);
  }
  void sortIncidentEdges(bool onDevice) {
    auto im = h_nodeOffsets[0];
    for (size_t i = 0; i < n; i++) {
      im = std::max(h_nodeOffsets[i + 1] - h_nodeOffsets[i], im);
    }
    if (onDevice && im < 10000) {
      TIMED_SCOPE(bla, "onDevice");
      cudaQuicksort<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
          d_NodeToPins, d_nodeOffsets, n, d_edgePriority);

      // // Wait for the GPU to finish all nested child kernels
      // cudaDeviceSynchronize();
      // cudaMemcpy(h_NodeToPins, d_NodeToPins, sizeof(EdgeType) * pinsCount,
      // cudaMemcpyDeviceToHost);

      // for (size_t i = 0; i < n; i++) {
      //   for (int j = h_nodeOffsets[i]; j < h_nodeOffsets[i + 1] - 1; j++) {
      //     // assert(false);
      //     if (!(priority_per_edge_[h_NodeToPins[j]] < priority_per_edge_[h_NodeToPins[j + 1]])) {
      //       std::cout << priority_per_edge_[h_NodeToPins[j]] << " "
      //                 << priority_per_edge_[h_NodeToPins[j + 1]] << " " << h_NodeToPins[j] << " "
      //                 << h_NodeToPins[j + 1] << " " << i << " " << j << " "
      //                 << h_nodeOffsets[i + 1] - h_nodeOffsets[i] << std::endl;
      //       // exit(1);
      //     }
      //   }
      // }
      // 6. Cleanup

    } else {
      cudaMemcpy(h_NodeToPins, d_NodeToPins, sizeof(EdgeType) * pinsCount, cudaMemcpyDeviceToHost);

      for (size_t i = 0; i < n; i++) {
        std::sort(h_NodeToPins + h_nodeOffsets[i], h_NodeToPins + h_nodeOffsets[i + 1],
                  [&](auto a, auto b) {
                    return h_weight[a] > h_weight[b] || (h_weight[a] == h_weight[b] && a < b);
                  });

        // for (int j = h_nodeOffsets[i]; j < h_nodeOffsets[i + 1] - 1; j++) {
        //   // assert(false);
        //   if (!(priority_per_edge_[h_NodeToPins[j]] < priority_per_edge_[h_NodeToPins[j + 1]])) {
        //     exit(1);
        //   }
        // }
      }
      cudaMemcpy(d_NodeToPins, h_NodeToPins, sizeof(EdgeType) * pinsCount, cudaMemcpyHostToDevice);
    }
  }
};

}  // namespace heihgm::bmatching::parallel::cuda