#pragma once
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "ds/modifiable_hypergraph.h"

namespace heihgm::ds::streaming {
template <typename NT, typename WT>
class FromMemoryStreamableHypergraph {
  struct Hyperedge {
    std::vector<NT> pins;
    WT w;
  };
  std::vector<Hyperedge> stream;
  size_t position = 0;
  size_t n_edges = 0;
  size_t n_vertices = 0;

 public:
  const static constexpr absl::string_view ds_name = "from_mem_stream_hypergraph";
  using NodeType = NT;
  using EdgeType = NT;
  using WeightType = WT;
  using StreamMember = Hyperedge;
  FromMemoryStreamableHypergraph(size_t num_edges, size_t num_nodes, bool = false, bool = false)
      : n_edges(num_edges), n_vertices(num_nodes) {
    stream.reserve(num_edges);
  }
  bool good() { return position < stream.size(); }
  StreamMember next() { return stream[position++]; }
  // IO methods
  void addEdge(const std::vector<NodeType>& nodes, WeightType w) { stream.push_back({nodes, w}); }
  void addEdge(NodeType u, NodeType v, WeightType w) { addEdge({u, v}, w); }
  void resortEdge(EdgeType e) {}
  void finish() {}
  void shrink_to_size() {}
  void setNodeWeight(NodeType n, WeightType w) {}
  void setEdgeWeight(EdgeType e, WeightType w) { stream[e].w = w; }
  auto edges() { return utils::Range(utils::SimpleIterator(0), utils::SimpleIterator(n_edges)); }
  auto nodes() { return utils::Range(utils::SimpleIterator(0), utils::SimpleIterator(n_vertices)); }
  NodeType edgeSize(EdgeType e) { return 0; }
  size_t currentNumEdges() const { return n_edges; }
  size_t currentNumNodes() const { return n_vertices; }
  size_t initialNumEdges() const { return n_edges; }
  size_t initialNumNodes() const { return n_vertices; }
  void shuffle() { std::random_shuffle(stream.begin(), stream.end()); }
  void resetStream() { position = 0; }
};
template <typename NT, typename WT>
class FromDiskStreamableHypergraph {
  struct Hyperedge {
    std::vector<NT> pins;
    WT w;
  };
  std::ifstream stream, weight_dscr;
  std::string type, weightfile;
  size_t position = 0;
  size_t n_edges = 0;
  size_t n_vertices = 0;
  size_t mode = 0;
  void readHeader() {
    std::string header;
    while (header == "") {
      if (stream.eof()) {
        std::cerr << ("File does not contain enough lines. Could not read header.") << std::endl;
        exit(1);
      }
      std::getline(stream, header);
      if (header.length() > 0 && header[0] == '%') {
        header = "";
      }
    }
    std::stringstream s(header);
    if (type == "hgr") {
      if (!(s >> n_edges)) {
        std::cerr << ("Failure to read in num_edges") << std::endl;
        exit(1);
      }
      if (!(s >> n_vertices)) {
        std::cerr << ("Failure to read in n_nodes") << std::endl;
        exit(1);
      }
      if (!(s >> mode)) {
        mode = 0;
      }
    }
    if (type == "graph") {
      if (!(s >> n_edges)) {
        std::cerr << ("Failure to read in num_edges") << std::endl;
        exit(1);
      }
      n_vertices = n_edges;
    }
  }

 public:
  const static constexpr absl::string_view ds_name = "from_disk_stream_hypergraph";
  using NodeType = NT;
  using EdgeType = NT;
  using WeightType = WT;
  using StreamMember = Hyperedge;
  FromDiskStreamableHypergraph(const std::string& _filename, const std::string& type,
                               std::string weights_filename = "")
      : stream(_filename), type(type), weightfile(weights_filename) {
    if (weights_filename != "") {
      weight_dscr = std::ifstream(weights_filename);
    }
    readHeader();
  }
  bool good() { return position < n_edges; }
  StreamMember next() {
    if (type == "hgr") {
      std::string str = "";
      if (stream.eof()) {
        std::cerr << "File does not contain enough lines. Read " << position << ". Expected "
                  << n_edges << std::endl;
        exit(1);
      }
      while ((str == "" || str[0] == '%')) {
        std::getline(stream, str);
        absl::StripLeadingAsciiWhitespace(&str);
      };

      std::stringstream line_stream{str};
      WeightType weight = 1;
      // Check if we read in weights
      if ((mode % 10) == 1) {
        line_stream >> weight;
      }
      position++;
      std::vector<NodeType> nodes;
      NodeType node;
      bool sorted = true;
      NodeType previousNode = 0;
      while (line_stream >> node) {
        nodes.push_back(node - 1);  // 0 based in memory
        sorted &= previousNode < node;
        previousNode = node;
      }
      return StreamMember{nodes, weight};
    }
    if (type == "graph") {
      std::string str = "";
      if (stream.eof()) {
        std::cerr << "File does not contain enough lines. Read " << position << ". Expected "
                  << n_edges << std::endl;
        exit(1);
      }
      while ((str == "" || str[0] == '%')) {
        std::getline(stream, str);
        absl::StripLeadingAsciiWhitespace(&str);
      };

      std::stringstream line_stream{str};
      WeightType weight = 1;
      // Check if we read in weights
      if (weightfile != "") {
        weight_dscr >> weight;
      }
      std::vector<NodeType> nodes{position};

      position++;
      NodeType node;
      bool sorted = true;
      NodeType previousNode = 0;
      while (line_stream >> node) {
        nodes.push_back(node - 1);  // 0 based in memory
      }
      return StreamMember{nodes, weight};
    }
    return {{}, 0};
  }
  // IO methods
  void addEdge(const std::vector<NodeType>& nodes, WeightType w) {}
  void addEdge(NodeType u, NodeType v, WeightType w) {}
  void resortEdge(EdgeType e) {}
  void finish() {}
  void shrink_to_size() {}
  void setNodeWeight(NodeType n, WeightType w) {}
  void setEdgeWeight(EdgeType e, WeightType w) {}
  auto edges() { return utils::Range(utils::SimpleIterator(0), utils::SimpleIterator(n_edges)); }
  auto nodes() { return utils::Range(utils::SimpleIterator(0), utils::SimpleIterator(n_vertices)); }
  NodeType edgeSize(EdgeType e) { return 0; }
  size_t currentNumEdges() const { return n_edges; }
  size_t currentNumNodes() const { return n_vertices; }
  size_t initialNumEdges() const { return n_edges; }
  size_t initialNumNodes() const { return n_vertices; }
  void shuffle() {}
  void resetStream() {
    position = 0;
    stream.clear();                  // clear fail and eof bits
    stream.seekg(0, std::ios::beg);  // back to the start!
    if (weightfile != "") {
      weight_dscr.clear();                  // clear fail and eof bits
      weight_dscr.seekg(0, std::ios::beg);  // back to the start!
    }
    readHeader();
  }
};
using StandardFromMemoryStreamableHypergraph = FromMemoryStreamableHypergraph<size_t, int>;
using StandardFromDiskStreamableHypergraph = FromDiskStreamableHypergraph<size_t, int>;

}  // namespace heihgm::ds::streaming
namespace heihgm::ds {
template <>
struct is_hypergraph_type<streaming::StandardFromMemoryStreamableHypergraph> : std::true_type {};
}  // namespace heihgm::ds