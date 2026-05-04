#include "utils/range.h"
namespace heihgm {
namespace ds {
namespace parallel {
template <typename T>
struct PointerRange {
  T* begin_;
  T* end_;

  __host__ __device__ PointerRange(T* begin, T* end) : begin_(begin), end_(end) {}

  __host__ __device__ T* begin() const { return begin_; }

  __host__ __device__ T* end() const { return end_; }
};

template <typename NT, typename ET, typename RT, typename WT>
struct CudaHypergraph {
  using NodeType = NT;
  using EdgeType = ET;
  using RefType = RT;
  using WeightType = WT;

  // Device-Memory
  RefType* d_node_offsets;
  RefType* d_edge_offsets;
  WeightType* d_edge_weights;
  NodeType* d_edge_to_node;
  EdgeType* d_node_to_edge;

  // Host-Memory
  RefType* h_node_offsets;
  RefType* h_edge_offsets;
  WeightType* h_edge_weights;
  NodeType* h_edge_to_node;
  EdgeType* h_node_to_edge;

  size_t m, n;

 public:
  // Constructor
  CudaHypergraph(size_t n, size_t m, size_t pins) : m(m), n(n) {
    // std::cout << "Initializing CUDA Hypergraph Data Structure (new)..." << n << " " << m << " "
    //           << pins << std::endl;

    // Allocate Memory
    cudaMalloc(&d_node_offsets, (n + 1) * sizeof(RefType));
    cudaMalloc(&d_edge_offsets, (m + 1) * sizeof(RefType));
    cudaMalloc(&d_edge_weights, m * sizeof(WeightType));
    cudaMalloc(&d_edge_to_node, pins * sizeof(NodeType));
    cudaMalloc(&d_node_to_edge, pins * sizeof(EdgeType));

    // Allocate host memory
    h_node_offsets = new RefType[n + 1];
    h_edge_offsets = new RefType[m + 1];
    h_edge_weights = new WeightType[m];
    h_edge_to_node = new NodeType[pins];
    h_node_to_edge = new EdgeType[pins];

    // std::cout << "CUDA Memory Allocated" << std::endl;
  }

  // Destructor
  ~CudaHypergraph() {
    // std::cout << "Freeing CUDA Hypergraph Memory..." << std::endl;

    // Free Device-Memory
    cudaFree(d_node_offsets);
    cudaFree(d_edge_offsets);
    cudaFree(d_edge_weights);
    cudaFree(d_edge_to_node);
    cudaFree(d_node_to_edge);

    // Free Host-Memory
    delete[] h_node_offsets;
    delete[] h_edge_offsets;
    delete[] h_edge_weights;
    delete[] h_edge_to_node;
    delete[] h_node_to_edge;

    // std::cout << "CUDA Memory Freed" << std::endl;
  }

  template <class Hypergraph>
  explicit CudaHypergraph(const Hypergraph& ref)
      : CudaHypergraph(ref.initialNumNodes(), ref.currentNumEdges(), ref.pinCount()) {
    TIMED_FUNC(timer);
    size_t pincount = ref.pinCount();

    // Initialize edge data
    RefType offset = 0;
    for (auto e : ref.edges()) {
      h_edge_weights[e] = ref.edgeWeight(e);
      h_edge_offsets[e] = offset;
      for (auto p : ref.pins(e)) {
        h_edge_to_node[offset++] = p;
      }
    }
    h_edge_offsets[m] = offset;  // Set the last offset value

    // Initialize node data
    offset = 0;
    for (auto n : ref.nodes()) {
      h_node_offsets[n] = offset;
      for (auto p : ref.incidentEdges(n)) {
        h_node_to_edge[offset++] = p;
      }
    }
    h_node_offsets[n] = offset;  // Set the last offset value

    // Copy data from host to device
    copyToDevice(pincount);
  }

  void copyToDevice(size_t pincount) {
    // Allocate device memory
    // cudaMalloc(&d_edge_weights, m * sizeof(WeightType));
    // cudaMalloc(&d_edge_offsets, (m + 1) * sizeof(RefType));
    // cudaMalloc(&d_edge_to_node, pincount * sizeof(NodeType));
    // cudaMalloc(&d_node_offsets, (n + 1) * sizeof(RefType));
    // cudaMalloc(&d_node_to_edge, pincount * sizeof(EdgeType));

    // Copy data from host to device
    cudaMemcpy(d_edge_weights, h_edge_weights, m * sizeof(WeightType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_offsets, h_edge_offsets, (m + 1) * sizeof(RefType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_to_node, h_edge_to_node, pincount * sizeof(NodeType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_offsets, h_node_offsets, (n + 1) * sizeof(RefType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_to_edge, h_node_to_edge, pincount * sizeof(EdgeType), cudaMemcpyHostToDevice);
  }

  // Returns incident nodes (pointer-range)
  __device__ PointerRange<NodeType> pinsD(EdgeType e) const {
    return PointerRange<NodeType>(d_edge_to_node + d_edge_offsets[e],
                                  d_edge_to_node + d_edge_offsets[e + 1]);
  }
  // Returns incident nodes (pointer-range)
  __host__ PointerRange<NodeType> pins(EdgeType e) const {
    return PointerRange<NodeType>(h_edge_to_node + h_edge_offsets[e],
                                  h_edge_to_node + h_edge_offsets[e + 1]);
  }

  // Returns incident edges (pointer-range)
  __device__ PointerRange<EdgeType> incidentEdgesD(NodeType n) const {
    return PointerRange<EdgeType>(d_node_to_edge + d_node_offsets[n],
                                  d_node_to_edge + d_node_offsets[n + 1]);
  }
  // Returns incident edges (pointer-range)
  __host__ PointerRange<EdgeType> incidentEdges(NodeType n) const {
    return PointerRange<EdgeType>(h_node_to_edge + h_node_offsets[n],
                                  h_node_to_edge + h_node_offsets[n + 1]);
  }

  // Returns edge size
  __device__ RefType edgeSize(EdgeType e) const {
    return d_edge_offsets[e + 1] - d_edge_offsets[e];
  }

  // Returns edge weight
  __host__ WeightType* edgeWeightD() const { return d_edge_weights; }
  __host__ WeightType edgeWeight(EdgeType e) const {
    // DO not sync everytime
    //  Make sure host memory is updated
    return h_edge_weights[e];
  }

  // Sets edge weight
  __device__ void setEdgeWeight(EdgeType edge, WeightType weight) {
    if (edge < 0 || edge >= m) {
      printf("Error: Edge index out of bounds.\n");
      return;
    }
    d_edge_weights[edge] = weight;
  }

  // Other getters
  __host__ RefType* nodeOffsetsD() const { return d_node_offsets; }
  __host__ RefType* edgeOffsetsD() const { return d_edge_offsets; }
  __host__ NodeType* edgeToNodeD() const { return d_edge_to_node; }
  __host__ EdgeType* nodeToEdgeD() const { return d_node_to_edge; }

  // Number of edges and nodes
  __host__ __device__ size_t initialNumEdges() const { return m; }
  __host__ __device__ size_t initialNumNodes() const { return n; }
  __host__ __device__ size_t currentNumEdges() const { return m; }
  __host__ __device__ size_t currentNumNodes() const { return n; }

  // Returns all edges (pointer-range)
  // __host__ PointerRange<size_t> edges_value_range() const { return PointerRange<size_t>(0, m); }
  __host__ auto edges() const { return Range<utils::SimpleIterator>(0UL, m); }
};
}  // namespace parallel
}  // namespace ds
}  // namespace heihgm