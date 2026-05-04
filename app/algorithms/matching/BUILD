load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "matching_cuda",
    srcs = [
        "cuda_local_max.cc",
        "empty_problem.cc",
        "gbbs.cc",
        "kokkos_local_max.cc",
        "sumac.cc",
    ],
    hdrs = [
        "gbbs.h",
        "local_max.h",
        "sumac.h",
    ],
    defines = [
        "CUDA_IS_ENABLED",
    ],
    linkopts = select({
        "//:cuda": ["-lcurand"],  # needed for linking in the right order
        "//conditions:default": [],
    }),
    deps = [
        "//app/algorithms:algorithm_base_impl_lib",
        "//ds:bmatching_lib",
        "//ds:empty_problem",
        "//ds:undirected_graph",
        "//ds/parallel:matching",
        "//matching/parallel",
        "//matching/parallel/cuda",
        "//matching/parallel/kokkos",
        "@gbbs//benchmarks/MaximalMatching/RandomGreedy:MaximalMatching",
        "@kokkos",
        "@nccl_prebuilt",
        "@sumac",
    ],
    alwayslink = 1,
)

cc_library(
    name = "matching_cc",
    srcs = [
        "empty_problem.cc",
        "gbbs.cc",
        "kokkos_local_max.cc",
    ],
    hdrs = [
        "gbbs.h",
        "local_max.h",
    ],
    cxxopts = ["-fopenmp"],
    linkopts = select({
        "//:cuda": ["-lcurand"],  # needed for linking in the right order
        "//conditions:default": ["-fopenmp"],
    }),
    deps = [
        "//app/algorithms:algorithm_base_impl_lib",
        "//ds:bmatching_lib",
        "//ds:empty_problem",
        "//ds:undirected_graph",
        "//ds/parallel:matching",
        "//matching/parallel",
        "//matching/parallel/kokkos",
        "@gbbs//benchmarks/MaximalMatching/RandomGreedy:MaximalMatching",
        "@kokkos",
    ],
    alwayslink = 1,
)

cc_library(
    name = "matching",
    visibility = ["//visibility:public"],
    deps = select({
        "//:cuda": [
            ":matching_cuda",
            "//:local_curand",
        ],  # needed for linking in the right order
        "//conditions:default": [":matching_cc"],
    }),
    alwayslink = 1,
)
