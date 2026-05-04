load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

cc_library(
    name = "modifiable_hypergraph_lib",
    srcs = ["modifiable_hypergraph.cc"],
    hdrs = [
        "bipartite_impl.h",
        "flat_bipartite_impl.h",
        "modifiable_hypergraph.h",
        "weight_sorted_bipartite_impl.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:range",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "modifiable_hypergraph_test",
    srcs = ["modifiable_hypergraph_test.cc"],
    deps = [
        ":bmatching_lib",
        ":modifiable_hypergraph_lib",
        "//bmatching/reductions_sorted",
        "//io:hgr_reader",
        "@googletest//:gtest_main",
    ],
)

config_setting(
    name = "hashing_config",
    values = {"define": "hashing=enabled"},
)

genrule(
    name = "hashing-config",
    srcs = [],
    outs = ["hashing-config.h"],
    cmd = select({
        ":hashing_config": "echo \"#define HASHING_ACTIVATED 1\">>$@",
        "//conditions:default": "echo \"#undef HASHING_ACTIVATED\">>$@",
    }),
    stamp = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "bmatching_lib",
    srcs = ["bmatching.cc"],
    hdrs = [
        "bmatching.h",
        ":hashing-config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":modifiable_hypergraph_lib",
        "//utils:logging",
        "//utils:random_lib",
        "//utils:range",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "empty_problem",
    hdrs = [
        "empty_problem.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:logging",
        "//utils:random_lib",
        "//utils:range",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "undirected_graph",
    hdrs = ["undirected_graph.h"],
    visibility = ["//visibility:public"],
    deps = ["//utils:range"],
)
