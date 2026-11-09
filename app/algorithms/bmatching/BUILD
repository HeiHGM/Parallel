load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cuda_library(
    name = "bmatching_cuda",
    srcs = ["pg_hgbm.cc"],
    defines = [
        "CUDA_IS_ENABLED",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//app/algorithms:algorithm_base_impl_lib",
        "//bmatching/parallel",
        "//bmatching/parallel/cuda",
        "//ds:empty_problem",
    ],
    alwayslink = 1,
)

cc_library(
    name = "bmatching",
    srcs = ["greedy.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//app/algorithms:algorithm_base_impl_lib",
        "//ds:empty_problem",
    ] + select({
        "//:cuda": [":bmatching_cuda"],  # needed for linking in the right order
        "//conditions:default": [],
    }),
    alwayslink = 1,
)
