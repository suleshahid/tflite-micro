# automatically generated by the FlatBuffers compiler, do not modify

# namespace: NamespaceB

import flatbuffers
from flatbuffers.compat import import_numpy

np = import_numpy()


class StructInNestedNS(object):
  __slots__ = ['_tab']

  @classmethod
  def SizeOf(cls):
    return 8

  # StructInNestedNS
  def Init(self, buf, pos):
    self._tab = flatbuffers.table.Table(buf, pos)

  # StructInNestedNS
  def A(self):
    return self._tab.Get(
        flatbuffers.number_types.Int32Flags,
        self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(0))

  # StructInNestedNS
  def B(self):
    return self._tab.Get(
        flatbuffers.number_types.Int32Flags,
        self._tab.Pos + flatbuffers.number_types.UOffsetTFlags.py_type(4))


def CreateStructInNestedNS(builder, a, b):
  builder.Prep(4, 8)
  builder.PrependInt32(b)
  builder.PrependInt32(a)
  return builder.Offset()


class StructInNestedNST(object):

  # StructInNestedNST
  def __init__(self):
    self.a = 0  # type: int
    self.b = 0  # type: int

  @classmethod
  def InitFromBuf(cls, buf, pos):
    structInNestedNS = StructInNestedNS()
    structInNestedNS.Init(buf, pos)
    return cls.InitFromObj(structInNestedNS)

  @classmethod
  def InitFromObj(cls, structInNestedNS):
    x = StructInNestedNST()
    x._UnPack(structInNestedNS)
    return x

  # StructInNestedNST
  def _UnPack(self, structInNestedNS):
    if structInNestedNS is None:
      return
    self.a = structInNestedNS.A()
    self.b = structInNestedNS.B()

  # StructInNestedNST
  def Pack(self, builder):
    return CreateStructInNestedNS(builder, self.a, self.b)
