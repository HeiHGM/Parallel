#pragma once
#include <filesystem>

#include "absl/strings/string_view.h"
#include "app/app_io.pb.h"

namespace heihgm::runner::hypergraph_storage_system {
namespace {
using heihgm::app::app_io::Hypergraph;
using heihgm::app::app_io::HypergraphCollection;
using heihgm::app::app_io::HypergraphStorageSystemConfig;

}  // namespace
class HypergraphStorageSystem {
 public:
  HypergraphStorageSystem(absl::string_view directory);
  std::vector<std::string> collection_names() const;
  std::vector<HypergraphCollection> collections() const;
  void reload();
  HypergraphStorageSystemConfig config() const;
  size_t hypergraph_count() const;
  std::string root_path() const;

 private:
  HypergraphStorageSystemConfig _config;
  std::string _directory;
  std::vector<HypergraphCollection> _hypergraph_collections;
  // TODO(reinstaedtler): add max depth?
  void find_collection(const std::filesystem::path& path);
};
}  // namespace heihgm::runner::hypergraph_storage_system