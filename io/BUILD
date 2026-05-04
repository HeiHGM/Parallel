load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "hgr_reader",
    srcs = ["hgr_reader.cc"],
    hdrs = ["hgr_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "pace_reader",
    hdrs = ["pace_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "tns_reader",
    hdrs = ["tns_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "graph_reader",
    hdrs = ["graph_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "binary_reader",
    hdrs = [
        "binary_reader.h",
        "binary_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "binary_writer",
    hdrs = [
        "binary_utils.h",
        "binary_writer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_binary(
    name = "mtx_to_unihypergraph",
    srcs = ["mtx_to_unihypergraph.cc"],
    deps = [
        ":hgr_writer",
        ":mtx_reader",
    ],
)

cc_binary(
    name = "hgr_validator",
    srcs = ["hgr_validator.cc"],
    deps = [
        ":hgr_reader",
        ":hgr_writer",
        "//runner:hypergraph_storage_system",
    ],
)

cc_library(
    name = "kagen",
    hdrs = ["kagen.h"],
    copts = ["-no-canonical-prefixes"],
    visibility = ["//visibility:public"],
    deps = [
        "@kagen",
    ],
)

cc_binary(
    name = "write_mtx",
    srcs = ["write_mtx.cc"],
    deps = [
        ":hgr_reader",
        ":hgr_writer",
    ],
)

cc_binary(
    name = "hgr_without_weights_to_weighted_and_mtx",
    srcs = ["hgr_without_weights_to_weighted_and_mtx.cc"],
    deps = [
        ":hgr_reader",
        ":hgr_writer",
        "//app:app_io_cc_proto",
    ],
)

cc_library(
    name = "mtx_reader",
    hdrs = ["mtx_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "hgr_writer",
    srcs = ["hgr_reader.cc"],
    hdrs = ["hgr_writer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//ds:modifiable_hypergraph_lib",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "hgr_reader_test",
    size = "small",
    srcs = ["hgr_reader_test.cc"],
    deps = [
        ":hgr_reader",
        "//ds:modifiable_hypergraph_lib",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "hgr_import",
    srcs = ["import.cc"],
    deps = [
        ":hgr_reader",
        ":hgr_writer",
        ":mtx_reader",
        ":seq_reader",
        "//app:app_io_cc_proto",
        "//ds:undirected_graph",
        "//runner:hypergraph_storage_system",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "seq_reader",
    hdrs = ["seq_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:logging",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "to_binary",
    srcs = ["to_binary.cc"],
    deps = [
        "binary_writer",
        "graph_reader",
        "//ds:undirected_graph",
    ],
)

cc_binary(
    name = "sort_hgr",
    srcs = ["sort_hgr.cc"],
    deps = [":hgr_reader"],
)

cc_binary(
    name = "mtx_to_enc",
    srcs = ["mtx_to_enc.cc"],
    deps = [
        "binary_reader",
        "binary_writer",
        "//io:mtx_reader",
    ],
)
