#pragma once
__global__ void init_ActiveNodes(bool* __restrict__ d_active_nodes, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d_active_nodes[idx] = true;
  }
}

template <typename EdgeType>
__global__ void init_selectedEdge(EdgeType* __restrict__ d_selectedEdge, const int n, const int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_selectedEdge[i] = m;
  }
}
