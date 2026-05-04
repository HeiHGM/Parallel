#pragma once

namespace heihgm::matching::parallel::cuda {
__global__ void initCurand(curandStateXORWOW_t* state, int m, unsigned long seed) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < m) {
    curand_init(seed, idx, 0, &state[idx]);
  }
  __syncthreads();
}
// XORWOW random number generator for initializations
template <typename RT, typename EdgeStatusType>
__global__ void generate_random_numbers_xorxow(RT* d_perm, const EdgeStatusType* d_edgeStatus,
                                               const int m, const double b_left,
                                               const double b_length, curandStateXORWOW_t* state) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < m && d_edgeStatus[id] == heihgm::ds::parallel::EdgeStatus::Free) {
    double randomValue = b_left + (curand_uniform(&state[id]) * (b_length));
    d_perm[id] += static_cast<RT>(randomValue);
  }
}
template <typename EdgeStatus, typename NoiseWeightType, typename RT, int Ib_left = 0,
          int Ib_length = 100>
struct XORXOWGenerator {
  curandStateXORWOW_t* devState = nullptr;
  // NoiseWeightType* noise = nullptr;
  const std::pair<int, int> kernel_config_;
  static constexpr double b_left = Ib_left;
  static constexpr double b_length = Ib_length;
  static constexpr int threadsPerBlock = 256;

  size_t m_;
  XORXOWGenerator(size_t m, size_t seed,
                  NoiseWeightType* noise_weight)  // ignore noise weights
      : kernel_config_({threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock}), m_(m) {
    cudaMalloc((void**)&devState, m * sizeof(curandStateXORWOW_t));
    initCurand<<<kernel_config_.second, kernel_config_.first>>>(devState, m, seed);
    // synchronize the device and the host
    cudaDeviceSynchronize();
  }

  void randomize(RT* d_perm, EdgeStatus* d_edge) {
    generate_random_numbers_xorxow<<<kernel_config_.second, kernel_config_.first>>>(
        d_perm, d_edge, m_, b_left, b_length, devState);
  }
  ~XORXOWGenerator() { cudaFree(devState); }
};

// LSG
// LSG Parkmiller random number generator
__device__ uint32_t lcg_parkmiller(uint32_t* state) {
  uint64_t product = (uint64_t)*state * 48271;
  uint32_t x = (product & 0x7fffffff) + (product >> 31);

  x = (x & 0x7fffffff) + (x >> 31);
  return *state = x;
}

// LSG Parkmiller random number generator control
template <typename RT, typename EdgeStatusType>
__global__ void random_weight_generator_lsg_parkmiller(RT* d_perm,
                                                       const EdgeStatusType* d_edgeStatus,
                                                       const int m, const double b_left,
                                                       const double b_length,
                                                       const unsigned int seed) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t state_e = edge + seed;
  if (edge < m) {
    double random_value =
        b_left + (b_length) * (((double)lcg_parkmiller(&state_e)) / ((double)0x7fffffff));
    d_perm[edge] += random_value;
  }
}
// LSG Parkmiller random number generator control
template <typename RT, typename EdgeStatusType>
__global__ void random_weight_generator_lsg_parkmiller_init_edge(RT* d_perm,
                                                                 const EdgeStatusType* d_edgeStatus,
                                                                 const int m, const double b_left,
                                                                 const double b_length,
                                                                 const unsigned int seed) {
  int edge = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t state_e = edge + seed;
  if (edge < m) {
    double random_value =
        b_left + (b_length) * (((double)lcg_parkmiller(&state_e)) / ((double)0x7fffffff));
    d_perm[edge] = edge + random_value;
  }
}

template <typename EdgeStatus, typename NoiseWeightType, typename RT, bool initEdge = false,
          int Ib_left = 0, int Ib_length = 100>
struct LSGParkMillerGenerator {
  // NoiseWeightType* noise = nullptr;
  const std::pair<int, int> kernel_config_;
  static constexpr double b_left = Ib_left;
  static constexpr double b_length = Ib_length;
  static constexpr int threadsPerBlock = 256;
  size_t seed;
  size_t m_;
  size_t round = 0;
  bool first = true;
  LSGParkMillerGenerator(size_t m, size_t seed,
                         NoiseWeightType* noise_weight)  // ignore noise weights
      : kernel_config_({threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock}),
        m_(m),
        seed(seed) {}

  void randomize(RT* d_perm, EdgeStatus* d_edge) {
    // if (round % 2 == 1) {
    //   return;
    // }
    if constexpr (initEdge) {
      random_weight_generator_lsg_parkmiller_init_edge<<<kernel_config_.second,
                                                         kernel_config_.first>>>(
          d_perm, d_edge, m_, b_left, b_length, seed + m_ * (round++));

    } else {
      random_weight_generator_lsg_parkmiller<<<kernel_config_.second, kernel_config_.first>>>(
          d_perm, d_edge, m_, b_left, b_length, seed + m_ * (round++));
    }
  }
};
template <typename RT>
__global__ void transfer_noise_weights(RT* d_perm, double* d_noise_weights, const int shift,
                                       const int m) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= m) return;
  id = (id + shift) % m;  // Shift index to new value

  d_perm[id] += d_noise_weights[id];
}
template <typename EdgeStatus, typename NoiseWeightType, typename RT>
struct TransferWeights {
  const std::pair<int, int> kernel_config_;
  static constexpr int threadsPerBlock = 256;
  size_t seed;
  size_t m_;
  size_t round = 0;
  bool first = true;
  NoiseWeightType* nw;
  TransferWeights(size_t m, size_t seed,
                  NoiseWeightType* noise_weight)  // ignore noise weights
      : kernel_config_({threadsPerBlock, (m + threadsPerBlock - 1) / threadsPerBlock}),
        m_(m),
        seed(seed),
        nw(noise_weight) {}
  void randomize(RT* d_perm, EdgeStatus* d_edge) {
    // Random Generation using pre-computed noise weights
    transfer_noise_weights<<<kernel_config_.second, kernel_config_.first>>>(
        d_perm, nw, 0,
        m_);  // Transfer noise weights to d_perm
  }
};
}  // namespace heihgm::matching::parallel::cuda