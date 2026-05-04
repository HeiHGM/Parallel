#pragma once
#include <iostream>
#include <istream>
#include <memory>
#include <numeric>
#include <sstream>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "ds/modifiable_hypergraph.h"
#include "utils/logging.h"

namespace heihgm {
namespace io {

template <class Hypergraph>
std::enable_if_t<heihgm::ds::is_hypergraph_type<Hypergraph>::value,
                 absl::StatusOr<std::unique_ptr<Hypergraph>>>
readTNSHypergraph(std::istream& stream, bool verbose = false) {
  TIMED_FUNC(timerObj);

  if (!stream.good()) {
    return absl::NotFoundError("stream not good error");
  }
  std::string header;
  std::vector<std::vector<typename Hypergraph::NodeType>> edges;
  std::vector<size_t> dims;
  while (stream) {
    if (stream.eof()) {
      return absl::NotFoundError("File does not contain enough lines. Could not read header.");
    }
    std::getline(stream, header);
    if (header.length() > 0 && header[0] == '#') {
      header = "";
      continue;
    }
    std::stringstream s(header);
    typename Hypergraph::NodeType node;
    edges.push_back(std::vector<typename Hypergraph::NodeType>{});
    while (s >> node) {
      edges.back().push_back(node - 1);
      if (dims.size() < edges.back().size()) {
        dims.resize(edges.back().size());
      }
      dims[edges.back().size() - 1] = std::max(dims[edges.back().size() - 1], node);
    }
    if (edges.back().empty()) {
      edges.resize(edges.size() - 1);
    }
  }
  typename Hypergraph::EdgeType num_edges = edges.size();
  typename Hypergraph::NodeType num_nodes = std::accumulate(dims.begin(), dims.end() - 1, 0L);
  size_t offset = 0;
  std::vector<size_t> offsets;
  for (auto di = dims.begin(); di != dims.end() - 1; di++) {
    offsets.push_back(offset);
    offset += *di;
  }

  auto h = std::make_unique<Hypergraph>(num_edges, num_nodes, true, true);
  for (typename Hypergraph::EdgeType edge = 0; edge < num_edges; edge++) {
    if (edges[edge].size() < 2) {
      std::cout << "Warning edge size " << edge << " is " << edges[edge].size() << std::endl;
      continue;
    }
    typename Hypergraph::WeightType weight = edges[edge].back();

    typename Hypergraph::NodeType previousNode = 0;
    std::vector<typename Hypergraph::NodeType> nodes;
    for (auto node = edges[edge].begin(); node != edges[edge].end() - 1; node++) {
      nodes.push_back(*node + offsets[node - edges[edge].begin()]);  // 0 based in memory
    }
    h->addEdge(nodes, weight);
    h->resortEdge(edge);
    if (verbose && edge % 1000 == 0) {
      std::cout << edge << " loaded." << std::endl;
    }
  }
  for (const auto n : h->nodes()) {
    h->setNodeWeight(n, 1);
  }
  return h;
}
template <class Hypergraph>
std::enable_if_t<not heihgm::ds::is_hypergraph_type<Hypergraph>::value,
                 absl::StatusOr<std::unique_ptr<Hypergraph>>>
readTNSHypergraph(std::istream& stream, bool verbose = false) {
  return absl::UnimplementedError("Only for true hgrs");
}
}  // namespace io
}  // namespace heihgm