load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "algorithm_base_impl_lib",
    srcs = ["algorithm_impl.cc"],
    hdrs = [
        "algorithm_impl.h",
        "dyn_algorithm_impl.h",
        "streaming_algorithm_impl.h",
        "//:build-info.h",
    ],
    defines = select({
        "//:kagen": ["USE_KAGEN_IMPL"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//app:app_io_cc_proto",
        "//io:binary_reader",
        "//io:graph_reader",
        "//io:hgr_reader",
        "//io:mtx_reader",
        "//io:pace_reader",
        "//io:seq_reader",
        "//io:tns_reader",
        "//utils:systeminfo",
        "@abseil-cpp//absl/status:statusor",
        "@protobuf//src/google/protobuf/util:time_util",
    ] + select({
        "//:kagen": ["//io:kagen"],
        "//conditions:default": [],
    }),
)
