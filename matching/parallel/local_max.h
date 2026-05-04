#pragma once
#include <utility>
// struct Timer2 {
//   std::string name;
//   Timer2(std::string n) : name(n) {}
//   ~Timer2() {
//     if (auto code = cudaPeekAtLastError(); code != cudaSuccess) {
//       std::cout << "nmae " << name << std::endl;
//       fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__);
//     } else {
//       std::cout << name << " passed" << std::endl;
//     }
//   }
// };
namespace heihgm::matching::parallel {

// #define TIMED_SCOPE(timer, text) Timer2 timer(text);
template <typename ComputationModel,
          template <typename, typename, typename> typename RandomGenerator, class PMatching,
          typename DebugInformation>
void local_max(PMatching& pm, DebugInformation& mutable_debug, size_t random_seed) {
  using EdgeStatus = typename PMatching::EdgeStatusType;
  using NoiseWeightType = typename PMatching::NoiseWeightType;
  auto& instance = pm.instance();
  const auto m = instance.initialNumEdges();
  const auto n = instance.initialNumNodes();
  // Initialization (copy) of Offsets and edge weight

  auto d_edge_status = pm.edgeStatusD();

  ComputationModel model(pm);

  RandomGenerator<EdgeStatus, NoiseWeightType, typename ComputationModel::PermType> random_gen(
      m, random_seed, pm.get_noise_weightsD());
  size_t round_counter = 0;

  while (true) {
    LOG(7) << "Round " << round_counter << std::endl;

    // ############### - Add noise - ###############
    {
      TIMED_SCOPE(bla, "random generator")
      random_gen.randomize(model.permutationsD(), d_edge_status);
    }
    {
      TIMED_SCOPE(bla, "computation")
      bool more_work = model.do_round();
      round_counter++;
      if (!more_work) {
        break;
      }
    }
  }

  mutable_debug.mutable_int64_info()->insert({"rounds", static_cast<long int>(round_counter)});
}
}  // namespace heihgm::matching::parallel