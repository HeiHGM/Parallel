genrule(
    name = "build-info",
    srcs = [],
    outs = ["build-info.h"],
    cmd = select({
        "//conditions:default": "cat bazel-out/stable-status.txt | while read line; do echo \"#define $$line\" >> $@; done; echo \"#define MALLOC_IMPLEMENTATION \\\"default\\\"\">>$@",
    }),
    stamp = True,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "jemalloc",
    values = {"define": "jemalloc=enabled"},
)

config_setting(
    name = "counting_allocs",
    values = {"define": "counting_allocs=enabled"},
)

config_setting(
    name = "gurobi",
    values = {"define": "gurobi=enabled"},
)

config_setting(
    name = "spack",
    values = {"define": "spack=enabled"},
)

config_setting(
    name = "karp_sipser",
    values = {"define": "karp_sipser=enabled"},
)

config_setting(
    name = "bsuitor",
    values = {"define": "bsuitor=enabled"},
)

config_setting(
    name = "logging",
    values = {"define": "logging=enabled"},
)

config_setting(
    name = "kagen",
    values = {"define": "kagen=enabled"},
)

config_setting(
    name = "detailed_logging",
    values = {"define": "detailed_logging=enabled"},
)

config_setting(
    name = "cuda",
    values = {"define": "cuda=enabled"},
)

cc_import(
    name = "local_curand",
    shared_library = ".spack-env/view/lib64/libcurand.so",
    visibility = ["//visibility:public"],
)
