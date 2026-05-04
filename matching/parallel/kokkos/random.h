#pragma once
namespace heihgm::matching::parallel::kokkos {
// XORWOW random number generator for initializations

template <typename EdgeStatus, typename NoiseWeightType, typename RT, int Ib_left = 0,
          int Ib_length = 100>
struct XORShiftGenerator {
  Kokkos::Random_XorShift64_Pool<> random_pool;

  // NoiseWeightType* noise = nullptr;
  static constexpr double b_left = Ib_left;
  static constexpr double b_length = Ib_length;

  size_t m_;
  XORShiftGenerator(size_t m, size_t seed,
                    NoiseWeightType* noise_weight)  // ignore noise weights
      : random_pool(seed), m_(m) {
    // synchronize the device and the host
  }

  void randomize(Kokkos::View<RT*>& d_perm, Kokkos::View<EdgeStatus*>& d_edgeStatus) {
    Kokkos::parallel_for(
        "randomize_xorshift", m_, KOKKOS_LAMBDA(size_t k) {
          auto generator = random_pool.get_state();
          auto i = k;
          //   for (auto i = k * 100; i < (k + 1) * 100; i++) {
          if (i < m_ && d_edgeStatus(i) == heihgm::ds::parallel::EdgeStatus::Free) {
            float x = b_left + generator.drand(0., 1.) * b_length;
            d_perm(i) += x;
          }
          //  }
          random_pool.free_state(generator);

          // do not forget to release the state of the engine
        });
    // int i = 0;
    // while (i < m_ && d_edgeStatus(i) != heihgm::ds::parallel::EdgeStatus::Free) {
    //   i++;
    // }
    // std::cout << "NeW" << std::endl;
    // for (int i = 0; i < m_; i++) {
    //   if (d_edgeStatus(i) == heihgm::ds::parallel::EdgeStatus::Free && i % 100 == 0) {
    //     std::cout << i << " " << d_perm(i) << std::endl;
    //   }
    // }
    Kokkos::fence();
  }
};
}  // namespace heihgm::matching::parallel::kokkos