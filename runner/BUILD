load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_library(
    name = "fork_runner_impl",
    srcs = ["fork_runner.cc"],
    defines = ["USE_KOKKOS_IMPL"],
    linkopts = select({
        "//:cuda": ["-lcurand"],  # needed for linking in the right order
        "//conditions:default": [],
    }),
    #malloc = select({
    #    "//:gperftools_tcmalloc": "@gperftools",
    #    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
    #}),
    deps = [
        "//app:app_io_cc_proto",
        "//app/algorithms/matching",
        "//app/algorithms/streaming/matching",
        "//utils:logging",
        "//utils:random_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "kokkos_runner_pure",
    srcs = ["fork_runner.cc"],
    #malloc = select({
    #    "//:gperftools_tcmalloc": "@gperftools",
    #    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
    #}),
    cxxopts = ["-fopenmp"],
    defines = ["USE_KOKKOS_IMPL"],
    linkopts = ["-fopenmp"],
    deps = [
        "//app:app_io_cc_proto",
        "//app/algorithms/matching",
        "//app/algorithms/streaming/matching",
        "//utils:logging",
        "//utils:random_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cuda_library(
    name = "kokkos_runner_cuda",
    srcs = ["fork_runner.cc"],
    defines = ["USE_KOKKOS_IMPL"],
    linkopts = select({
        "//:cuda": ["-lcurand"],  # needed for linking in the right order
        "//conditions:default": [],
    }),
    #malloc = select({
    #    "//:gperftools_tcmalloc": "@gperftools",
    #    "//conditions:default": "@bazel_tools//tools/cpp:malloc",
    #}),
    deps = [
        "//app:app_io_cc_proto",
        "//app/algorithms/matching",
        "//app/algorithms/streaming/matching",
        "//utils:logging",
        "//utils:random_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_binary(
    name = "kokkos_runner",
    deps = select({
        "//:cuda": [":kokkos_runner_cuda"],  # needed for linking in the right order
        "//conditions:default": [":kokkos_runner_pure"],
    }),
)

cc_binary(
    name = "fork_runner",
    deps = [":fork_runner_impl"],
)

cc_binary(
    name = "generate_experiment_config",
    srcs = ["generate_experiment_config.cc"],
    deps = [
        ":hypergraph_storage_system",
        "//app:app_io_cc_proto",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_binary(
    name = "humanize",
    srcs = ["humanize.cc"],
    deps = [
        "//app:app_io_cc_proto",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "to_csv",
    srcs = ["to_csv.cc"],
    deps = [
        "//app:app_io_cc_proto",
        "@abseil-cpp//absl/strings",
        "@protobuf//src/google/protobuf/util:time_util",
    ],
)

cc_library(
    name = "hypergraph_storage_system",
    srcs = ["hypergraph_storage_system.cc"],
    hdrs = ["hypergraph_storage_system.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//app:app_io_cc_proto",
        "@abseil-cpp//absl/strings",
    ],
)
