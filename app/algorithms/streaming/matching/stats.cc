#include <algorithm>
#include <memory>
#include <utility>

#include "app/algorithms/algorithm_impl.h"
#include "ds/empty_problem.h"

namespace heihgm::app::algorithms::streaming::stats {
struct EmptyHypergraph : public ds::EmptyProblem<ds::StandardIntegerHypergraph> {
  static constexpr absl::string_view ds_name = "empty_hypergraph";
};

}  // namespace heihgm::app::algorithms::streaming::stats