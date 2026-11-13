#pragma once
#include <cassert>
#include <numeric>

#include "utils/logging.h"
namespace heihgm::bmatching::parallel {
template <template <bool> typename ComputationModel, class PBMatching, typename DebugInformation,
          bool onDevice, bool single_vote>
void pg_hgbm(PBMatching& pm, DebugInformation& mutable_debug, size_t random_seed, bool weights) {
  using EdgeStatus = typename PBMatching::EdgeStatusType;
  using NoiseWeightType = typename PBMatching::NoiseWeightType;
  auto& instance = pm.instance();
  const auto m = instance.initialNumEdges();
  const auto n = instance.initialNumNodes();
  // Initialization (copy) of Offsets and edge weight

  auto d_edge_status = pm.edgeStatusD();
  std::vector<typename PBMatching::EdgeType> priorities_per_edge;
  if (weights) {
    // empty use weights
  } else {
    priorities_per_edge.resize(m);
    std::iota(priorities_per_edge.rbegin(), priorities_per_edge.rend(), 0);
  }  // TODO mix
  ComputationModel<single_vote> model(pm, priorities_per_edge);

  model.sortIncidentEdges(onDevice);
  // RandomGenerator<EdgeStatus, NoiseWeightType, typename ComputationModel::PermType> random_gen(
  //     m, random_seed, pm.get_noise_weightsD());
  size_t round_counter = 0;
  size_t start = 0;
  size_t curr_elem = 128;
  size_t stop = std::min(curr_elem, m);
  // assert(pm.instance().sorted(priorities_per_edge));
  while (true) {
    LOG(7) << "Round " << round_counter << std::endl;

    // ############### - Add noise - ###############
    {
      TIMED_SCOPE(bla, "computation")
      bool more_work = model.do_round(start, stop);
      // std::cout << pm.weight() << " " << pm.isMaximal() << " " << pm.free_edges_size() <<
      // std::endl;
      round_counter++;
      if (!more_work) {
        break;
        curr_elem *= 2;
        start = stop;
        stop = std::min(curr_elem + start, m);
        if (start == m) {
          break;
        }
      }
    }
  }

  mutable_debug.mutable_int64_info()->insert({"rounds", static_cast<long int>(round_counter)});
}
}  // namespace heihgm::bmatching::parallel