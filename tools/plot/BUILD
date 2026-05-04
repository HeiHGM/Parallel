load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_library(
    name = "dataloader",
    srcs = ["dataloader.cc"],
    hdrs = ["dataloader.h"],
    deps = [
        "//app:app_io_cc_proto",
        "@abseil-cpp//absl/status:statusor",
        "@protobuf//src/google/protobuf/util:time_util",
    ],
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@my_deps//:requirements.bzl", "requirement")

pybind_extension(
    name = "plot_lib",
    srcs = [
        "gaussian_error.h",
        "pivot_table.h",
        "plot.cc",
        "plot.h",
    ],
    deps = [
        ":dataloader",
        "@matplotlibcpp17",
        #"@pybind11_bazel",
        # requirement("matplotlib"),
        # "@my_deps//matplotlib:pkg",
        #requirement("scikit-learn"),
        "@protobuf//src/google/protobuf/util:time_util",
        #"@spack_python",
    ],
)

py_binary(
    name = "plot_cc",
    srcs = ["plot_cc.py"],
    imports = ["."],
    deps = [
        ":plot_lib",
        requirement("matplotlib"),
        requirement("seaborn"),
        requirement("scikit-learn"),
    ],
)
