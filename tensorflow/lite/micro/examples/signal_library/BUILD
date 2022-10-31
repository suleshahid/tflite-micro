load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "micro_copts",
)

# generate_cc_arrays(
#     name = "generated_window_example_model_cc",
#     src = "/tmp/window_example/window_example.tflite",
#     out = "window_example_model_data.cc",
# )

# generate_cc_arrays(
#     name = "generated_window_example_model_hdr",
#     src = "/tmp/window_example/window_example.tflite",
#     out = "window_example_model_data.h",
# )

# cc_library(
#     name = "window_example_model",
#     srcs = [
#         ":generated_window_example_model_cc",
#     ],
#     hdrs = [
#         ":generated_window_example_model_hdr",
#     ],
#     copts = micro_copts(),
# )

# cc_binary(
#     name = "micro_signal_example",
#     srcs = [
#         "micro_signal_example.cc",
#     ],
#     copts = [
#         "-Werror",
#         "-Wsign-compare",
#     ],
#     deps = [
#         ":window_example_model",
#         "//tensorflow/lite/micro:micro_framework",
#         "//tensorflow/lite/micro:micro_log",
#         "//tensorflow/lite/micro:op_resolvers",
#     ],
# )

py_binary(
    name = "signal_example",
    srcs = ["signal_example.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols from interpreter_wrapper_pybind.so
        "noubsan",
    ],
    deps = [
        requirement("numpy"),
        requirement("tensorflow-cpu"),
        "//tensorflow/lite/micro/python/interpreter/src:tflm_runtime",
    ],
)

cc_binary(
    name = "tf_use_test",
    srcs = [
        "tf_use_test.cc",
    ],
    deps = [
        "@tensorflow_headers",
        "@tensorflow_headers//:tensorflow_headers_lib",
    ],
)