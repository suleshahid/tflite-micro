"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_window_op_py.cc
"""

import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export

from typing import TypeVar

@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('window')
def window(input, weights, shift, name=None):
  r"""Apply a window to an input signal with a right shift to each element

  Args:
    input: A `Tensor` of type `int16`. An N-D time domain input signal
    weights: A `Tensor` of type `int16`.
      Constant 1-D window weights. Size must match innermost input dimension.
    shift: An `int`.
      An amount of right shifts to perform on each element before writing
      to the output
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int16`.
    An N-D time domain output signal. Size must match input.
  """
  _ctx = _context._context or _context.context()
  tld = _ctx._thread_local_data
  if tld.is_eager:
    try:
      _result = pywrap_tfe.TFE_Py_FastPathExecute(
        _ctx, "Window", name, input, weights, "shift", shift)
      return _result
    except _core._NotOkStatusException as e:
      _ops.raise_from_not_ok_status(e, name)
    except _core._FallbackException:
      pass
    try:
      _result = _dispatcher_for_window(
          (input, weights, shift, name,), None)
      if _result is not NotImplemented:
        return _result
      return window_eager_fallback(
          input, weights, shift=shift, name=name, ctx=_ctx)
    except _core._SymbolicException:
      pass  # Add nodes to the TensorFlow graph.
    except (TypeError, ValueError):
      _result = _dispatch.dispatch(
            window, (), dict(input=input, weights=weights, shift=shift,
                             name=name)
          )
      if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
        return _result
      raise
  else:
    _result = _dispatcher_for_window(
        (input, weights, shift, name,), None)
    if _result is not NotImplemented:
      return _result
  # Add nodes to the TensorFlow graph.
  shift = _execute.make_int(shift, "shift")
  try:
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
        "Window", input=input, weights=weights, shift=shift, name=name)
  except (TypeError, ValueError):
    _result = _dispatch.dispatch(
          window, (), dict(input=input, weights=weights, shift=shift,
                           name=name)
        )
    if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
      return _result
    raise
  _result = _outputs[:]
  if _execute.must_record_gradient():
    _attrs = ("shift", _op._get_attr_int("shift"))
    _inputs_flat = _op.inputs
    _execute.record_gradient(
        "Window", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result

Window = tf_export("raw_ops.Window")(_ops.to_raw_op(window))
_dispatcher_for_window = window._tf_type_based_dispatcher.Dispatch


def window_eager_fallback(input, weights, shift, name, ctx):
  shift = _execute.make_int(shift, "shift")
  input = _ops.convert_to_tensor(input, _dtypes.int16)
  weights = _ops.convert_to_tensor(weights, _dtypes.int16)
  _inputs_flat = [input, weights]
  _attrs = ("shift", shift)
  _result = _execute.execute(b"Window", 1, inputs=_inputs_flat, attrs=_attrs,
                             ctx=ctx, name=name)
  if _execute.must_record_gradient():
    _execute.record_gradient(
        "Window", _inputs_flat, _attrs, _result)
  _result, = _result
  return _result
