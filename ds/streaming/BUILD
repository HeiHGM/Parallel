load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "hypergraph",
    hdrs = ["hypergraph.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:range",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "matching",
    hdrs = ["matching.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)
