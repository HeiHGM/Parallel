load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "matching_cuda",
    srcs = ["matching.cc"],
    hdrs = [
        "hypergraph.h",
        "hypergraph_cuda.h",
        "matching.h",
    ],
    defines = ["CUDA_IS_ENABLED"],
    visibility = ["//visibility:public"],
    deps = ["//ds:modifiable_hypergraph_lib"],
)

cc_library(
    name = "matching_cc",
    srcs = ["matching.cc"],
    hdrs = [
        "hypergraph.h",
        "hypergraph_cuda.h",
        "matching.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["//ds:modifiable_hypergraph_lib"],
)

cuda_library(
    name = "bmatching_cuda",
    hdrs = [
        "bmatching.h",
        "hypergraph.h",
        "hypergraph_cuda.h",
    ],
    defines = ["CUDA_IS_ENABLED"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "@cuda//:thrust",
        "@kokkos",
    ],
)

cc_library(
    name = "bmatching_cc",
    hdrs = [
        "bmatching.h",
        "hypergraph.h",
        "hypergraph_cuda.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["//ds:modifiable_hypergraph_lib"],
)

cc_library(
    name = "matching",
    visibility = ["//visibility:public"],
    deps = select({
        "//:cuda": [":matching_cuda"],  # needed for linking in the right order
        "//conditions:default": [":matching_cc"],
    }),
)

cc_library(
    name = "bmatching",
    visibility = ["//visibility:public"],
    deps = select({
        "//:cuda": [":bmatching_cuda"],  # needed for linking in the right order
        "//conditions:default": [":bmatching_cc"],
    }),
)
