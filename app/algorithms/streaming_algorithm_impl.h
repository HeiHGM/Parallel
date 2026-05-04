#pragma once
#include <memory>

#include "app/algorithms/algorithm_impl.h"

namespace heihgm::app::algorithms {
/*
 * StreamableInputType has good():bool method to return whether there is a next stream element
 * StreamableInputType has next():StreamMember returning the next element in Stream
 * TempStorage stores the intermediate Results.
 * Output Type should conform to the general Result interface
 * Following methods need to be implemented:
 *   - std::unique_ptr<TempStorage> InitStorage(const app_io::AlgorithmConfig& config,const
 * StreamableInputType& stream) Creates the storage object (might use const properties of stream)
 *   - void Stream(const StreamableInputType& element, TempStorage& storage)
 *     Streams the element, might apply changes to TempStorage
 *   - absl::StatusOr<std::unique_ptr<OutputType>> Transform(TempStorage& storage)
 *     Transforms the storage into the result (OutputType)
 */
template <class StreamableInputType, class TempStorage, class OutputType, bool TwoPass = false,
          bool preStream = false>
class StreamingAlgorithmImpl : public AlgorithmImpl<StreamableInputType, OutputType> {
  virtual std::unique_ptr<TempStorage> InitStorage(const app_io::AlgorithmConfig& config,
                                                   const StreamableInputType& stream) = 0;
  virtual std::unique_ptr<TempStorage> InitStorage(const app_io::AlgorithmConfig& config,
                                                   const StreamableInputType& stream,
                                                   app_io::DebugInformation& debug) {
    return InitStorage(config, stream);
  }
  // Stream into storage
  virtual void Stream(const typename StreamableInputType::StreamMember& member,
                      TempStorage& storage) = 0;
  virtual void PreStream(const typename StreamableInputType::StreamMember& member) {}
  virtual absl::StatusOr<std::unique_ptr<OutputType>> Stream2(
      std::unique_ptr<StreamableInputType> stream, std::unique_ptr<OutputType> prev_result) {
    return absl::UnimplementedError("If you want to use Stream2 implement it");
  }
  virtual void StreamSecond(const typename StreamableInputType::StreamMember& member,
                            OutputType& prev_result) {}
  // Virtual method called once to Transform storage into the result
  virtual absl::StatusOr<std::unique_ptr<OutputType>> Transform(TempStorage& storage) = 0;
  // Overriden method to stream
  absl::Status Run(const Hypergraph& h, const RunConfig& run_config, int index,
                   std::unique_ptr<StreamableInputType> input,
                   std::vector<AlgorithmRunInformation>& results) override {
    TIMED_FUNC(timer2);
    const auto& config = run_config.algorithm_configs().at(index);
    AlgorithmRunInformation result;
    int64_t total_duration = 0;
    auto t1_sys = std::chrono::system_clock::now();
    if (preStream) {
      while (input->good()) {
        typename StreamableInputType::StreamMember next = input->next();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        PreStream(next);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        auto dur = t2 - t1;
        total_duration +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64_t>(dur).count();
      }
      input->resetStream();
    }
    std::unique_ptr<TempStorage> storage =
        InitStorage(config, *input, *result.mutable_debug_information());
    bool is_exact = false;
    while (input->good()) {
      typename StreamableInputType::StreamMember next = input->next();
      std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
      Stream(next, *storage);
      std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
      auto dur = t2 - t1;
      total_duration += std::chrono::duration_cast<std::chrono::nanoseconds, int64_t>(dur).count();
    }
    // Now transform, add the time
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto status = Transform(*storage);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto dur = t2 - t1;
    total_duration += std::chrono::duration_cast<std::chrono::nanoseconds, int64_t>(dur).count();
    std::unique_ptr<OutputType> solution = std::move(status.value());
    if (TwoPass) {
      //
      input->resetStream();
      while (input->good()) {
        typename StreamableInputType::StreamMember next = input->next();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        StreamSecond(next, *solution);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        auto dur = t2 - t1;
        total_duration +=
            std::chrono::duration_cast<std::chrono::nanoseconds, int64_t>(dur).count();
      }
    }
    auto t2_sys = std::chrono::system_clock::now();
    result.set_is_exact(is_exact);

    result.set_weight(solution->weight());
    result.set_quality(solution->quality());
    result.set_size(solution->size());
    result.set_edge_count(solution->instance().currentNumEdges());
    result.set_node_count(solution->instance().currentNumNodes());
    result.set_free_edges(solution->free_edges_size());
    if (!solution->valid()) {
      return absl::InternalError("Not a valid");
    }
    // TODO(reinstaedtler): Update to exact timing
    *result.mutable_start_time() = google::protobuf::util::TimeUtil::TimeTToTimestamp(
        std::chrono::system_clock::to_time_t(t1_sys));
    *result.mutable_end_time() = google::protobuf::util::TimeUtil::TimeTToTimestamp(
        std::chrono::system_clock::to_time_t(t2_sys));
    *result.mutable_algo_duration() =
        google::protobuf::util::TimeUtil::NanosecondsToDuration(total_duration);

    *result.mutable_algorithm_config() = config;
    results.push_back(result);
    if (index + 1 < run_config.algorithm_configs_size()) {
      return AlgorithmImplFactory<OutputType>::getInstance().Run(h, run_config, index + 1,
                                                                 std::move(solution), results);
    }
    return absl::OkStatus();
  }
};
}  // namespace heihgm::app::algorithms