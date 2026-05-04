#include "app/algorithms/matching/local_max.h"
#include "ds/parallel/hypergraph.h"
#include "ds/parallel/matching.h"
#include "matching/parallel/kokkos/CRCW.h"
#include "matching/parallel/kokkos/random.h"
namespace heihgm::algorithms::matching::kokkos {
namespace {
using PMatching = ds::parallel::KokkosMatching<ds::parallel::KokkosHypergraph<int, int, int, int>>;
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
}  // namespace
template <typename ComputationModel,
          template <typename, typename, typename> typename RandomGenerator>
using Kokkos_Local_Max_Matching_Parallel =
    local_max::Local_Max_Matching_Parallel<PMatching, ComputationModel, RandomGenerator>;
using Kokkos_CRCW =
    Kokkos_Local_Max_Matching_Parallel<heihgm::matching::parallel::kokkos::CRCWModel<PMatching>,
                                       heihgm::matching::parallel::kokkos::XORShiftGenerator>;
REGISTER_IMPL_NAMED(Kokkos_CRCW, "local_max_crcw_xorshift");
}  // namespace heihgm::algorithms::matching::kokkos