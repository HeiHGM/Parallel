#pragma once
namespace heihgm::io {
template <class Hypergraph>
std::enable_if_t<heihgm::ds::is_hypergraph_type<Hypergraph>::value,
                 absl::StatusOr<std::unique_ptr<Hypergraph>>>
readPace(std::istream& stream, bool verbose = false) {
  TIMED_FUNC(timerObj);

  if (!stream.good()) {
    return absl::NotFoundError("stream not good error");
  }
  std::string header;
  while (header == "") {
    if (stream.eof()) {
      return absl::NotFoundError("File does not contain enough lines. Could not read header.");
    }
    std::getline(stream, header);
    if (header.length() > 0 && header[0] == 'c') {
      header = "";
    }
  }
  std::stringstream s(header);
  std::string p;
  s >> p;
  if (p != "p") {
    return absl::NotFoundError("problem statement not found");
  }
  s >> p;
  if (p == "hs") {
    typename Hypergraph::EdgeType num_edges = 0;
    typename Hypergraph::NodeType num_nodes = 0;
    s >> num_nodes;
    s >> num_edges;
    auto h = std::make_unique<Hypergraph>(num_edges, num_nodes, true, true);

    for (size_t lines = 0; lines < num_edges; lines++) {
      std::string line;
      std::getline(stream, line);

      while (line.size() > 0 && line[0] == 'c') {
        std::getline(stream, line);
      }
      std::stringstream line_stream{std::string(line)};

      typename Hypergraph::NodeType node;
      std::vector<typename Hypergraph::NodeType> nodes;
      while (line_stream >> node) {
        nodes.push_back(node - 1);  // 0 based in memory
      }
      std::sort(nodes.begin(), nodes.end());
      h->addEdge(nodes, 1);
    }
    return h;

  } else if (p == "ds") {
    typename Hypergraph::EdgeType num_edges = 0;
    typename Hypergraph::NodeType num_nodes = 0;
    s >> num_nodes;
    s >> num_edges;
    auto h = std::make_unique<Hypergraph>(num_nodes, num_nodes, true, true);
    std::vector<std::vector<typename Hypergraph::NodeType>> n_adj(num_nodes);
    for (size_t lines = 0; lines < num_edges; lines++) {
      std::string line;
      std::getline(stream, line);

      while (line.size() > 0 && line[0] == 'c') {
        std::getline(stream, line);
      }
      std::stringstream line_stream{std::string(line)};

      typename Hypergraph::NodeType node1, node2;
      line_stream >> node1;
      node1--;
      line_stream >> node2;
      node2--;
      n_adj[node1].push_back(node2);
      n_adj[node2].push_back(node1);
    }
    size_t e = 0;
    for (auto& neigh : n_adj) {
      neigh.push_back(e++);
      std::sort(neigh.begin(), neigh.end());
      h->addEdge(neigh, 1);
    }
    return h;
  }
  return absl::UnimplementedError("Problem not described implemented");
}
// TODO implement reverse
template <class Hypergraph>
std::enable_if_t<not heihgm::ds::is_hypergraph_type<Hypergraph>::value,
                 absl::StatusOr<std::unique_ptr<Hypergraph>>>
readPace(std::istream& stream, bool verbose = false) {
  return absl::UnimplementedError("Only for true hgrs");
}
}  // namespace heihgm::io