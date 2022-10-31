load(
    "//tensorflow/lite/micro:build_def.bzl",
    "micro_copts",
)

package(
    licenses = ["notice"],
)

cc_library(
    name = "signal_ops",
    srcs = [
        "window.cc",
    ],
    copts = micro_copts(),
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/micro:flatbuffer_utils",
        "//tensorflow/lite/micro:memory_helpers",
        "//tensorflow/lite/micro:micro_context",
        "//tensorflow/lite/micro:micro_utils",
        "//signal/src:window",
        "//tensorflow/lite/micro/kernels:kernel_util",
    ],
)