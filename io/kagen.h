#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "kagen/kagen.h"

namespace heihgm {
namespace io {
template <class Graph>
absl::StatusOr<std::unique_ptr<Graph>> readKaGen(const std::string& input, bool verbose = false) {
  TIMED_FUNC(timer);
  char** argv = nullptr;
  int argc = 0;
  MPI_Init(&argc, &argv);

  kagen::KaGen generator(MPI_COMM_WORLD);

  // Generate a RGG2D graph with 16 nodes and 32 edges
  const kagen::Graph graph = generator.GenerateFromOptionString(input);
  const kagen::Edgelist& edges = graph.edges;
  const kagen::VertexRange& range = graph.vertex_range;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto intermediate = std::make_unique<Graph>(0, range.second);

  {
    for (const auto& [u, v] : edges) {  // only add one edge direction
      intermediate->addEdge(u, v, 1);
    }
  }
  intermediate->shrink_to_size();
  return intermediate;
}
}  // namespace io
}  // namespace heihgm