
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/time_util.h>

#include <fstream>
#include <sstream>

#include "absl/strings/match.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "app/app_io.pb.h"

namespace {
using heihgm::app::app_io::ExperimentConfig;
using heihgm::app::app_io::ExperimentResultMain;
using heihgm::app::app_io::ExperimentResultPart;

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    std::ifstream exp_file(argv[i]);
    if (!exp_file.good()) {
      std::cerr << "The file '" << argv[i] << "' must exist." << std::endl;
      return 1;
    }
    ExperimentResultMain part;
    std::stringstream buffer;
    buffer << exp_file.rdbuf();
    if (!google::protobuf::TextFormat::ParseFromString(buffer.str(), &part)) {
      std::cout << "something went wrong " << argv[i] << std::endl;
      exit(0);
    }
    std::ofstream exp_fileO(absl::StrReplaceAll(argv[i], {{"textproto", "binary_proto"}}));
    part.SerializeToOstream(&exp_fileO);
  }
}