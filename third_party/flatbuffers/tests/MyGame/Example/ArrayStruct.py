# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Example

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class ArrayStruct(object):
  __slots__ = ['_tab']

  @classmethod
  def SizeOf(cls):
    return 160

  # ArrayStruct
  def Init(self, buf, pos):
    self._tab = flatbuffers.table.Table(buf, pos)

  # ArrayStruct
  def A(self):
    return self._tab.Get(
        flatbuffers.number_types.Float32Flags,
        self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

  # ArrayStruct
  def B(self):
    return [
        self._tab.Get(
            flatbuffers.number_types.Int32Flags, self._tab.Pos +
            flatbuffers.number_types.UOffsetTFlags.py_type(4 + i * 4))
        for i in range(15)
    ]

  # ArrayStruct
  def BLength(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    if o != 0:
      return self._tab.VectorLen(o)
    return 0

  # ArrayStruct
  def BIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
    return o == 0

  # ArrayStruct
  def C(self):
    return self._tab.Get(
        flatbuffers.number_types.Int8Flags,
        self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(64))

  # ArrayStruct
  def D(self, obj, i):
    obj.Init(self._tab.Bytes, self._tab.Pos + 72 + i * 32)
    return obj

  # ArrayStruct
  def DLength(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(72))
    if o != 0:
      return self._tab.VectorLen(o)
    return 0

  # ArrayStruct
  def DIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(72))
    return o == 0

  # ArrayStruct
  def E(self):
    return self._tab.Get(
        flatbuffers.number_types.Int32Flags,
        self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(136))

  # ArrayStruct
  def F(self):
    return [
        self._tab.Get(
            flatbuffers.number_types.Int64Flags, self._tab.Pos +
            flatbuffers.number_types.UOffsetTFlags.py_type(144 + i * 8))
        for i in range(2)
    ]

  # ArrayStruct
  def FLength(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(144))
    if o != 0:
      return self._tab.VectorLen(o)
    return 0

  # ArrayStruct
  def FIsNone(self):
    o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(144))
    return o == 0


def CreateArrayStruct(builder, a, b, c, d_a, d_b, d_c, d_d, e, f):
  builder.Prep(8, 160)
  for _idx0 in range(2, 0, -1):
    builder.PrependInt64(f[_idx0 - 1])
  builder.Pad(4)
  builder.PrependInt32(e)
  for _idx0 in range(2, 0, -1):
    builder.Prep(8, 32)
    for _idx1 in range(2, 0, -1):
      builder.PrependInt64(d_d[_idx0 - 1][_idx1 - 1])
    builder.Pad(5)
    for _idx1 in range(2, 0, -1):
      builder.PrependInt8(d_c[_idx0 - 1][_idx1 - 1])
    builder.PrependInt8(d_b[_idx0 - 1])
    for _idx1 in range(2, 0, -1):
      builder.PrependInt32(d_a[_idx0 - 1][_idx1 - 1])
  builder.Pad(7)
  builder.PrependInt8(c)
  for _idx0 in range(15, 0, -1):
    builder.PrependInt32(b[_idx0 - 1])
  builder.PrependFloat32(a)
  return builder.Offset()


import MyGame.Example.NestedStruct
try:
  from typing import List
except:
  pass


class ArrayStructT(object):

  # ArrayStructT
  def __init__(self):
    self.a = 0.0  # type: float
    self.b = None  # type: List[int]
    self.c = 0  # type: int
    self.d = None  # type: List[MyGame.Example.NestedStruct.NestedStructT]
    self.e = 0  # type: int
    self.f = None  # type: List[int]

  @classmethod
  def InitFromBuf(cls, buf, pos):
    arrayStruct = ArrayStruct()
    arrayStruct.Init(buf, pos)
    return cls.InitFromObj(arrayStruct)

  @classmethod
  def InitFromObj(cls, arrayStruct):
    x = ArrayStructT()
    x._UnPack(arrayStruct)
    return x

  # ArrayStructT
  def _UnPack(self, arrayStruct):
    if arrayStruct is None:
      return
    self.a = arrayStruct.A()
    if not arrayStruct.BIsNone():
      if np is None:
        self.b = []
        for i in range(arrayStruct.BLength()):
          self.b.append(arrayStruct.B(i))
      else:
        self.b = arrayStruct.BAsNumpy()
    self.c = arrayStruct.C()
    if not arrayStruct.DIsNone():
      self.d = []
      for i in range(arrayStruct.DLength()):
        self.d.append(arrayStruct.D(i))
    self.e = arrayStruct.E()
    if not arrayStruct.FIsNone():
      if np is None:
        self.f = []
        for i in range(arrayStruct.FLength()):
          self.f.append(arrayStruct.F(i))
      else:
        self.f = arrayStruct.FAsNumpy()

  # ArrayStructT
  def Pack(self, builder):
    return CreateArrayStruct(builder, self.a, self.b, self.c, self.d.a,
                             self.d.b, self.d.c, self.d.d, self.e, self.f)
