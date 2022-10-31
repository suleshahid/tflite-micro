#load("@tflm_pip_deps//:requirements.bzl", "requirement")

def py_tflm_signal_library(
        name,
        srcs = [],
        deps = [],
        visibility = None,
        cc_op_defs = [],
        cc_op_kernels = []):
    """Generates a Python library for the ops defined in `op_defs` and `py_srcs

    Defines three targets:
    <name>
        Python library that exposes all ops defined in `cc_op_defs` and `py_srcs`.
    <name>_cc
        C++ library that registers any c++ ops in `cc_op_defs`, and includes the
        kernels from `cc_op_kernels`.
    gen_<name>_py
        Python library that exposes any c++ ops.
    Args:
      name: The name for the python library target build by this rule.
      srcs: Python source files for the Python library.
      deps: Dependencies for the Python library.
      visibility: Visibility for the Python library.
      cc_op_defs: A list of c++ src files containing REGISTER_OP definitions.
      cc_op_kernels: A list of c++ targets containing kernels that are used
          by the Python library.
    """
    binary_path = "python/ops"
    if srcs:
        binary_path_end_pos = srcs[0].rfind("/")
        binary_path = srcs[0][0:binary_path_end_pos]
    binary_name = binary_path + "/_" + cc_op_kernels[0][1:] + ".so"
    if cc_op_defs:
        binary_name = binary_path + "/_" + name + ".so"
        library_name = name + "_cc"
        native.cc_library(
            name = library_name,
            srcs = cc_op_defs,
            copts = [],
            alwayslink = 1,
            deps = cc_op_kernels +[],
        )

        native.cc_binary(
            name = binary_name,
            copts = [],
            linkshared = 1,
            linkopts = [],
            deps = [
                ":" + library_name
            ],
        )

    native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY2AND3",
        visibility = visibility,
        data = [":" + binary_name],
        deps = deps,
    )


# A rule to build a TensorFlow OpKernel.
def tflm_signal_kernel_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        alwayslink = 1):
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        alwayslink = alwayslink,
    )
