#include <nvtx3/nvToolsExt.h>

#include "app/algorithms/algorithm_impl.h"
#include "app/algorithms/matching/local_max.h"
#include "ds/parallel/hypergraph.h"
#include "ds/parallel/hypergraph_cuda.h"
#include "ds/parallel/matching.h"
#include "matching/parallel/cuda/CRCW.h"
#include "matching/parallel/cuda/CREW.h"
#include "matching/parallel/cuda/random.h"
#include "matching/parallel/local_max.h"

namespace heihgm::algorithms::matching::cuda {
namespace {
using PMatching = ds::parallel::CudaMatching<ds::parallel::CudaHypergraph<int, int, int, int>>;
using heihgm::app::app_io::AlgorithmConfig;
using heihgm::app::app_io::DebugInformation;
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::RunConfig;
}  // namespace
template <typename ComputationModel,
          template <typename, typename, typename> typename RandomGenerator>
using Cuda_Local_Max_Matching_Parallel =
    local_max::Local_Max_Matching_Parallel<PMatching, ComputationModel, RandomGenerator>;
using CREW_XORXOW =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CREWModel<PMatching>,
                                     ::heihgm::matching::parallel::cuda::XORXOWGenerator>;
using CREW_LSGPark =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CREWModel<PMatching>,
                                     ::heihgm::matching::parallel::cuda::LSGParkMillerGenerator>;
template <typename A, typename B, typename C>
using ParkmillerEdge = ::heihgm::matching::parallel::cuda::LSGParkMillerGenerator<A, B, C, true>;
template <typename A, typename B, typename C>
using Parkmiller1 =
    ::heihgm::matching::parallel::cuda::LSGParkMillerGenerator<A, B, C, false, -1, 1>;
using CREW_LSGParkEdge =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CREWModel<PMatching>,
                                     ParkmillerEdge>;
REGISTER_IMPL_NAMED(CREW_XORXOW, "local_max_crew_xorxow");
REGISTER_IMPL_NAMED(CREW_LSGPark, "local_max_crew_parkmiller");
REGISTER_IMPL_NAMED(CREW_LSGParkEdge, "local_max_crew_parkmiller_id");

// Crcw

using CRCW_XORXOW =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CRCWModel<PMatching>,
                                     ::heihgm::matching::parallel::cuda::XORXOWGenerator>;
using CRCW_LSGPark =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CRCWModel<PMatching>,
                                     ::heihgm::matching::parallel::cuda::LSGParkMillerGenerator>;
using CRCW_LSGPark1 =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CRCWModel<PMatching>,
                                     Parkmiller1>;
using CRCW_LSGParkEdge =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CRCWModel<PMatching>,
                                     ParkmillerEdge>;

REGISTER_IMPL_NAMED(CRCW_XORXOW, "local_max_crcw_xorxow");
REGISTER_IMPL_NAMED(CRCW_LSGPark, "local_max_crcw_parkmiller");
REGISTER_IMPL_NAMED(CRCW_LSGPark1, "local_max_crcw_parkmiller1");

REGISTER_IMPL_NAMED(CRCW_LSGParkEdge, "local_max_crcw_parkmiller_id");
template <typename A, typename B, typename C>
using TransferWeights = ::heihgm::matching::parallel::cuda::TransferWeights<A, B, C>;

using CRCW_Transfer =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CRCWModel<PMatching>,
                                     TransferWeights>;
using CREW_Transfer =
    Cuda_Local_Max_Matching_Parallel<::heihgm::matching::parallel::cuda::CREWModel<PMatching>,
                                     TransferWeights>;
REGISTER_IMPL_NAMED(CRCW_Transfer, "local_max_crcw_transfer");
REGISTER_IMPL_NAMED(CREW_Transfer, "local_max_crew_transfer");

}  // namespace heihgm::algorithms::matching::cuda