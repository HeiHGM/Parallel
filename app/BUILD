load("@protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

cc_binary(
    name = "app",
    srcs = [
        "app.cc",
        "//:build-info.h",
    ],
    deps = [
        ":app_io_cc_proto",
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
    name = "kokkos_app_pure",
    srcs = [
        "kokkos_app.cc",
        "//:build-info.h",
    ],
    cxxopts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    deps = [
        ":app_io_cc_proto",
        "//app/algorithms/matching",
        "//app/algorithms/streaming/matching",
        "//utils:logging",
        "//utils:random_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
        "@kokkos",
    ],
)

cuda_library(
    name = "kokkos_app_cuda",
    srcs = [
        "kokkos_app.cc",
        "//:build-info.h",
    ],
    linkopts = select({
        "//:cuda": ["-lcurand"],  # needed for linking in the right order
        "//conditions:default": [],
    }),
    deps = [
        ":app_io_cc_proto",
        "//app/algorithms/matching",
        "//app/algorithms/streaming/matching",
        "//utils:logging",
        "//utils:random_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
        "@kokkos",
    ],
)

cc_binary(
    name = "kokkos_app",
    deps = select({
        "//:cuda": [":kokkos_app_cuda"],  # needed for linking in the right order
        "//conditions:default": [":kokkos_app_pure"],
    }),
)

cc_proto_library(
    name = "app_io_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":app_io_proto"],
)

proto_library(
    name = "app_io_proto",
    srcs = ["app_io.proto"],
    deps = [
        "@protobuf//:duration_proto",
        "@protobuf//:timestamp_proto",
    ],
)
