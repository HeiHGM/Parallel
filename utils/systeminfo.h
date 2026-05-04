#pragma once

#include <string>

namespace heihgm {
namespace utils {
namespace systeminfo {
std::string getHostname();
std::string getUsername();
double process_mem_usage();
double process_mem_usage(pid_t);

}  // namespace systeminfo
}  // namespace utils
}  // namespace heihgm