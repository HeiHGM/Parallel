#pragma once

#ifdef CUDA_IS_ENABLED
#include <curand_kernel.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <memory>
#include <numeric>

#include "absl/strings/string_view.h"
#include "ds/modifiable_hypergraph.h"

namespace heihgm {
namespace ds {
namespace parallel {
enum class EdgeStatus : char { Free = 0, InMatching = 1, NeighborInMatching = 2, JustBlocked = 3 };
enum class NodeStatus : char { Active = 0, Deactivated = 1, NewlyCompleted = 2, Completed = 3 };

// LSG Parkmiller random number generator Kokkos inline function
KOKKOS_INLINE_FUNCTION uint32_t lcg_parkmiller_kokkos(uint32_t* state) {
  uint64_t product = (uint64_t)*state * 48271;
  uint32_t x = (product & 0x7fffffff) + (product >> 31);

  x = (x & 0x7fffffff) + (x >> 31);
  return *state = x;
}

#ifdef CUDA_IS_ENABLED
template <typename EdgeStatus>
struct IsEdgeStatus {
  EdgeStatus target;

  // Constructor to set the value we are looking for
  IsEdgeStatus(EdgeStatus s) : target(s) {}

  // The operator that Thrust calls for every element
  __host__ __device__ bool operator()(const EdgeStatus& x) const { return x == target; }
};
// LSG Parkmiller random number generator
__device__ inline uint32_t lcg_parkmiller(uint32_t* state) {
  uint64_t product = (uint64_t)*state * 48271;
  uint32_t x = (product & 0x7fffffff) + (product >> 31);

  x = (x & 0x7fffffff) + (x >> 31);
  return *state = x;
}

// LSG Parkmiller random number generator control
template <typename NoiseWeightType>
__global__ void random_weight_generator_init_v1(NoiseWeightType* d_noise_weights,
                                                const size_t numEdges, const double b_left,
                                                const double b_right,
                                                const unsigned int random_seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= numEdges) return;
  uint32_t state_e = (id + 1) * 48271u ^ random_seed;

  d_noise_weights[id] =
      b_left + (b_right - b_left) * (((double)lcg_parkmiller(&state_e)) / ((double)0x7fffffff));
}

// XORWOW random number generator
template <typename NoiseWeightType>
__global__ void generate_random_numbers_init_v2(NoiseWeightType* d_noise_weights,
                                                const size_t numEdges, const double b_left,
                                                const double b_right,
                                                const unsigned int random_seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= numEdges) return;
  curandStateXORWOW_t state;
  curand_init(random_seed, id, 0, &state);

  double randomValue = b_left + (curand_uniform(&state) * (b_right - b_left));
  d_noise_weights[id] = static_cast<NoiseWeightType>(randomValue);
}
#endif

template <class Graph = heihgm::ds::StandardIntegerHypergraph>
class Matching {
  std::unique_ptr<Graph> _g;

 public:
  Kokkos::View<EdgeStatus*> edge_status;

  using Graph_t = Graph;
  using HypergraphType = Graph;
  using NodeID_t = typename Graph::NodeType;
  using EdgeID_t = typename Graph::EdgeType;
  using NodeType = typename Graph::NodeType;
  using EdgeType = typename Graph::EdgeType;
  using WeightType = typename Graph::WeightType;
  explicit Matching(std::unique_ptr<Graph> g)
      : _g(std::move(g)), edge_status("edges", _g->initialNumEdges()) {}
  Graph& instance() { return *_g; }

  static constexpr absl::string_view ds_name = "parallel_bmatching";
  bool valid() { return isMatching(); }
  bool isMaximal() {
    for (auto e : _g->edges()) {
      if (edge_status[e] == EdgeStatus::Free) {
        return false;
      }
    }
    return true;
  }
  bool isMatching() {
    for (auto e : _g->edges()) {
      if (edge_status[e] == EdgeStatus::InMatching) {
        for (auto p : _g->pins(e)) {
          for (auto e2 : _g->incidentEdges(p)) {
            if (edge_status[e2] == EdgeStatus::InMatching && e2 != e) {
              // std::cout << e2 << " " << e << " " << p << std::endl;
              return false;
            }
          }
        }
      }
    }
    return true;
  }
  size_t free_edges_size() const {
    size_t result = 0;
    for (auto e : _g->edges()) {
      result += edge_status[e] == EdgeStatus::Free;
    }
    return result;
  }
  size_t size() {
    size_t result = 0;
    for (auto e : _g->edges()) {
      result += edge_status[e] == EdgeStatus::InMatching;
    }
    return result;
  }
  size_t weight() {
    size_t w = 0;
    for (auto e : _g->edges()) {
      w += _g->edgeWeight(e) * (edge_status[e] == EdgeStatus::InMatching);
    }
    return w;
  }
  double quality() { return 0; }
  void addToMatching(EdgeType e) { edge_status[e] = EdgeStatus::InMatching; }
  bool isFree(EdgeType e) { return edge_status[e] == EdgeStatus::Free; }
  void markNeighborInMatching(EdgeType e) { edge_status[e] = EdgeStatus::NeighborInMatching; }
  bool justBlocked(EdgeType e) { return edge_status[e] == EdgeStatus::JustBlocked; }
  void markNeigborJustBlocked(EdgeType e) { edge_status[e] = EdgeStatus::JustBlocked; }
};

template <class Graph = heihgm::ds::StandardIntegerHypergraph,
          template <typename> class Container = Kokkos::View,
          typename VarEdgeStatusType = EdgeStatus*>
class KokkosMatching {
 public:
  using Graph_t = Graph;
  using HypergraphType = Graph;
  using NodeID_t = typename Graph::NodeType;
  using EdgeID_t = typename Graph::EdgeType;
  using NodeType = typename Graph::NodeType;
  using EdgeType = typename Graph::EdgeType;
  using WeightType = typename Graph::WeightType;
  using EdgeStatusType = EdgeStatus;
  using NodeStatusType = NodeStatus;
  using NoiseWeightType = float;

 private:
  std::unique_ptr<Graph> _g;
  Container<VarEdgeStatusType> edge_status;
  bool generated_noise_weights = false;
  Kokkos::View<double*> noise_weights;
  Kokkos::View<NodeStatus*> d_node_status;
  Kokkos::View<EdgeType*> d_selectedEdge;
  Kokkos::View<int*> d_pin_counter;
  Kokkos::View<float*> d_perm;

 public:
  explicit KokkosMatching(std::unique_ptr<Graph> g)
      : _g(std::move(g)),
        edge_status("edge_status", _g->initialNumEdges()),
        d_node_status("node_status", _g->initialNumNodes()),
        d_selectedEdge("selected Edge", _g->initialNumNodes()),
        d_pin_counter("pinsCounter", _g->initialNumEdges()),
        d_perm("permut", _g->initialNumEdges()) {
    copy_weights();
  }
  Graph& instance() { return *_g; }
  void copy_weights() {
    auto& weights = _g->edgeWeightD();
    Kokkos::fence();
    Kokkos::parallel_for(
        "copy weights", _g->initialNumEdges(), KOKKOS_LAMBDA(size_t i) { d_perm(i) = weights(i); });
    auto m = _g->initialNumEdges();
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), edge_status, EdgeStatus::Free);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_pin_counter, 0);
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_selectedEdge, m);

    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_node_status, NodeStatus::Active);
    Kokkos::fence();
  }
  static constexpr absl::string_view ds_name = "parallel_bmatching";
  float* get_noise_weightsD() { return nullptr; }
  bool valid() { return isMatching(); }
  bool isMaximal() {
    Kokkos::View<EdgeStatus*, Kokkos::HostSpace> Hedge_status("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(Hedge_status, edge_status);
    for (auto e : _g->edges()) {
      if (Hedge_status[e] == EdgeStatus::Free) {
        return false;
      }
    }
    return true;
  }
  bool isMatching() {
    Kokkos::View<EdgeStatus*, Kokkos::HostSpace> Hedge_status("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(Hedge_status, edge_status);
    for (auto e : _g->edges()) {
      if (Hedge_status[e] == EdgeStatus::InMatching) {
        for (auto p : _g->pins(e)) {
          for (auto e2 : _g->incidentEdges(p)) {
            if (Hedge_status[e2] == EdgeStatus::InMatching && e2 != e) {
              // std::cout << e2 << " " << e << " " << p << std::endl;
              return false;
            }
          }
        }
      }
    }
    return true;
  }
  size_t free_edges_size() const {
    Kokkos::View<EdgeStatus*, Kokkos::HostSpace> Hedge_status("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(Hedge_status, edge_status);
    size_t result = 0;
    for (auto e : _g->edges()) {
      result += Hedge_status[e] == EdgeStatus::Free;
    }
    return result;
  }
  auto& nodeStatusD() { return d_node_status; }
  auto& selectedEdgeD() { return d_selectedEdge; }
  auto& dPermD() { return d_perm; }
  auto& pinCounterD() { return d_pin_counter; }
  size_t size() {
    Kokkos::View<EdgeStatus*, Kokkos::HostSpace> Hedge_status("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(Hedge_status, edge_status);
    size_t result = 0;
    for (auto e : _g->edges()) {
      result += Hedge_status[e] == EdgeStatus::InMatching;
    }
    return result;
  }
  size_t weight() {
    Kokkos::View<EdgeStatus*, Kokkos::HostSpace> Hedge_status("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(Hedge_status, edge_status);
    size_t w = 0;
    for (auto e : _g->edges()) {
      w += _g->edgeWeight(e) * (Hedge_status[e] == EdgeStatus::InMatching);
    }
    return w;
  }
  double quality() { return 0; }
  void addToMatching(EdgeType e) { edge_status[e] = EdgeStatus::InMatching; }
  bool isFree(EdgeType e) { return edge_status[e] == EdgeStatus::Free; }
  void markNeighborInMatching(EdgeType e) { edge_status[e] = EdgeStatus::NeighborInMatching; }
  bool justBlocked(EdgeType e) { return edge_status[e] == EdgeStatus::JustBlocked; }
  void markNeigborJustBlocked(EdgeType e) { edge_status[e] = EdgeStatus::JustBlocked; }
  Container<VarEdgeStatusType>& get_edge_status() { return edge_status; }
  auto& edgeStatusD() { return edge_status; }
  void initialize_noise_weights(const double b_left, const double b_right, const int random_gen,
                                const unsigned int random_seed) {
    const int m = this->_g->initialNumEdges();

    auto edge_status_view = this->edge_status;
    auto noise_weights_view = this->noise_weights;

    switch (random_gen) {
      case 3: {  // LCG Park-Miller
        Kokkos::parallel_for(
            "Init Noise V1", Kokkos::RangePolicy<>(0, m), KOKKOS_LAMBDA(const int edge) {
              if (edge_status_view(edge) == heihgm::ds::parallel::EdgeStatus::Free) {
                uint32_t state_e = (edge + 1) * 48271u ^ random_seed;
                noise_weights_view(edge) =
                    b_left + (b_right - b_left) *
                                 (double(lcg_parkmiller_kokkos(&state_e)) / double(0x7fffffff));
              }
            });
      } break;
      case 4: {  // Kokkos XorShift64
        Kokkos::Random_XorShift64_Pool<> rand_pool(random_seed);
        Kokkos::parallel_for(
            "Init Noise V2", Kokkos::RangePolicy<>(0, m), KOKKOS_LAMBDA(const int edge) {
              if (edge_status_view(edge) == heihgm::ds::parallel::EdgeStatus::Free) {
                auto state_e = rand_pool.get_state();
                noise_weights(edge) =
                    Kokkos::rand<decltype(state_e), double>::draw(state_e, b_left, b_right);
                rand_pool.free_state(state_e);
              }
            });
      } break;
      default: {  // Kokkos XorShift64
        Kokkos::Random_XorShift64_Pool<> rand_pool(random_seed);
        Kokkos::parallel_for(
            "Init Noise V2", Kokkos::RangePolicy<>(0, m), KOKKOS_LAMBDA(const int edge) {
              if (edge_status_view(edge) == heihgm::ds::parallel::EdgeStatus::Free) {
                auto state_e = rand_pool.get_state();
                noise_weights(edge) =
                    Kokkos::rand<decltype(state_e), double>::draw(state_e, b_left, b_right);
                rand_pool.free_state(state_e);
              }
            });
      } break;
    }

    // Debug: Print first 100 values of noise_weights for debugging
    /*Kokkos::View<double*, Kokkos::HostSpace> host_noise_weights("hedge", _g->initialNumEdges());
    Kokkos::deep_copy(host_noise_weights, noise_weights_view);
    for (int i = 0; i < 100; ++i) {
      printf("noise_weights[%d] = %f\n", i, host_noise_weights[i]);
    }*/

    generated_noise_weights = true;
  }

  Kokkos::View<double*> get_noise_weights() const {
    if (!generated_noise_weights) {
      std::cerr << "Error: Noise weights have not been initialized." << std::endl;
      return Kokkos::View<double*>();
    }
    return noise_weights;
  }
};

#ifdef CUDA_IS_ENABLED
template <typename EdgeType>
__global__ void init_selectedEdge(EdgeType* __restrict__ d_selectedEdge, const int n, const int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    d_selectedEdge[i] = m;
  }
}
__global__ void IntToDoubleArray(int* input, float* output, int n);
template <class Graph = heihgm::ds::StandardIntegerHypergraph>
class CudaMatching {
 public:
  using Graph_t = Graph;
  using HypergraphType = Graph;
  using NodeID_t = typename Graph::NodeType;
  using EdgeID_t = typename Graph::EdgeType;
  using NodeType = typename Graph::NodeType;
  using EdgeType = typename Graph::EdgeType;
  using WeightType = typename Graph::WeightType;
  using NoiseWeightType = double;
  using EdgeStatusType = EdgeStatus;
  using NodeStatusType = NodeStatus;

 private:
  std::unique_ptr<Graph> _g;
  EdgeStatus* h_edge_status;  // Host pointer for edge_status
  bool generated_noise_weights = false;
  NoiseWeightType* d_noise_weights = nullptr;
  EdgeStatus* d_edge_status;
  NodeStatus* d_node_status;
  EdgeType* d_selectedEdge;
  int* d_pin_counter;
  float* d_perm;

 public:
  explicit CudaMatching(std::unique_ptr<Graph> g) : _g(std::move(g)) {
    size_t numEdges = _g->initialNumEdges();
    std::cout << _g->initialNumEdges() << " " << _g->initialNumNodes() << std::endl;
    // Allocate memory for edge_status on both host and device
    cudaMalloc(&d_edge_status, numEdges * sizeof(EdgeStatus));
    h_edge_status = new EdgeStatus[numEdges]();  // Initialize with "Free" by default

    // Also update Device memory to "Free"
    cudaMemset(d_edge_status, static_cast<int>(EdgeStatus::Free), numEdges * sizeof(EdgeStatus));

    cudaMalloc(&d_node_status, _g->initialNumNodes() * sizeof(NodeStatus));
    cudaMemset(d_node_status, static_cast<int>(NodeStatus::Active),
               _g->initialNumNodes() * sizeof(NodeStatus));

    cudaMalloc(&d_selectedEdge, _g->initialNumNodes() * sizeof(EdgeType));

    init_selectedEdge<<<(_g->initialNumNodes() + 256 - 1) / 256, 256>>>(
        d_selectedEdge, _g->initialNumNodes(), numEdges);
    cudaMalloc(&d_pin_counter, numEdges * sizeof(int));
    cudaMalloc(&d_perm, numEdges * sizeof(float));
    IntToDoubleArray<<<(numEdges + 256 - 1) / 256, 256>>>(_g->edgeWeightD(), d_perm, numEdges);
  }
  int* pinCounterD() { return d_pin_counter; }
  float* dPermD() { return d_perm; }
  ~CudaMatching() {
    cudaFree(d_pin_counter);
    cudaFree(d_perm);
    // Free device memory
    cudaFree(d_selectedEdge);
    cudaFree(d_node_status);
    cudaFree(d_edge_status);
    delete[] h_edge_status;
    if (d_noise_weights != nullptr) {
      cudaFree(d_noise_weights);
    }
  }

  Graph& instance() { return *_g; }

  static constexpr absl::string_view ds_name = "parallel_bmatching_cuda";

  bool valid() { return isMatching(); }

  bool isMaximal() {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);

    for (auto e : _g->edges()) {
      if (h_edge_status[e] == EdgeStatus::Free) {
        return false;
      }
    }
    for (auto e : _g->edges()) {
      if (h_edge_status[e] == EdgeStatus::NeighborInMatching) {
        bool matched_neighbor = false;
        for (auto p : _g->pins(e)) {
          for (auto e2 : _g->incidentEdges(p)) {
            if (h_edge_status[e2] == EdgeStatus::InMatching && e2 != e) {
              matched_neighbor = true;
            }
          }
        }
        if (!matched_neighbor) {
          return false;
        }
      }
    }
    return true;
  }

  bool isMatching() {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);

    for (auto e : _g->edges()) {
      if (h_edge_status[e] == EdgeStatus::InMatching) {
        for (auto p : _g->pins(e)) {
          for (auto e2 : _g->incidentEdges(p)) {
            if (h_edge_status[e2] == EdgeStatus::InMatching && e2 != e) {
              // std::cout << e2 << " " << e << " " << p << std::endl;
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  size_t free_edges_size() const {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);

    size_t result = 0;
    for (auto e : _g->edges()) {
      result += h_edge_status[e] == EdgeStatus::Free;
    }
    return result;
  }

  size_t size() {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);

    size_t result = 0;
    for (auto e : _g->edges()) {
      result += h_edge_status[e] == EdgeStatus::InMatching;
    }
    return result;
  }

  size_t weight() {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);

    size_t w = 0;
    for (auto e : _g->edges()) {
      w += _g->edgeWeight(e) * (h_edge_status[e] == EdgeStatus::InMatching);
    }
    return w;
  }

  double quality() { return 0; }

  void addToMatching(EdgeType e) {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);
    h_edge_status[e] = EdgeStatus::InMatching;
    cudaMemcpy(d_edge_status, h_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyHostToDevice);
  }

  bool isFree(EdgeType e) { return h_edge_status[e] == EdgeStatus::Free; }

  void markNeighborInMatching(EdgeType e) {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);
    h_edge_status[e] = EdgeStatus::NeighborInMatching;
    cudaMemcpy(d_edge_status, h_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyHostToDevice);
  }

  bool justBlocked(EdgeType e) { return h_edge_status[e] == EdgeStatus::JustBlocked; }

  void markNeigborJustBlocked(EdgeType e) {
    cudaMemcpy(h_edge_status, d_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyDeviceToHost);
    h_edge_status[e] = EdgeStatus::JustBlocked;
    cudaMemcpy(d_edge_status, h_edge_status, _g->initialNumEdges() * sizeof(EdgeStatus),
               cudaMemcpyHostToDevice);
  }

  __host__ EdgeStatus* edgeStatusD() const { return d_edge_status; }
  __host__ NodeStatus* nodeStatusD() const { return d_node_status; }
  __host__ EdgeType* selectedEdgeD() const { return d_selectedEdge; }

  void initialize_noise_weights(const NoiseWeightType b_left, const NoiseWeightType b_right,
                                const int random_gen, const unsigned int random_seed) {
    auto numEdges = _g->initialNumEdges();

    cudaMalloc(&d_noise_weights, numEdges * sizeof(NoiseWeightType));

    int threadsPerBlock = 256;
    int blocks = (numEdges + threadsPerBlock - 1) / threadsPerBlock;

    switch (random_gen) {
      case 3:  // LDG Park-Miller
        random_weight_generator_init_v1<<<blocks, threadsPerBlock>>>(d_noise_weights, numEdges,
                                                                     b_left, b_right, random_seed);
        // printf("Initialized noise weights on device with random generator ID 1.\n");
        break;
      case 4:  // Cuda XORWOW
        generate_random_numbers_init_v2<<<blocks, threadsPerBlock>>>(d_noise_weights, numEdges,
                                                                     b_left, b_right, random_seed);
        // printf("Initialized noise weights on device with random generator ID 3.\n");
        break;
      default:  // Cuda XORWOW
        generate_random_numbers_init_v2<<<blocks, threadsPerBlock>>>(d_noise_weights, numEdges,
                                                                     b_left, b_right, random_seed);
        // printf("Initialized noise weights on device with random generator ID 3.\n");
    }
    cudaDeviceSynchronize();  // Fence

    // Debug: Print first 100 values of d_noise_weights for debugging
    /*double host_noise_weights[100];
    cudaMemcpy(host_noise_weights, d_noise_weights, 100 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("d_noise_weights:\n");
    for (int i = 0; i < 100; ++i) {
      printf("d_noise_weights[%d] = %f\n", i, host_noise_weights[i]);
    }*/

    generated_noise_weights = true;
  }

  __host__ NoiseWeightType* get_noise_weightsD() const {
    if (!generated_noise_weights) {
      //  std::cerr << "Error: Noise weights have not been initialized." << std::endl;
      return nullptr;
    }
    return d_noise_weights;
  }
};
#endif
}  // namespace parallel
}  // namespace ds
}  // namespace heihgm