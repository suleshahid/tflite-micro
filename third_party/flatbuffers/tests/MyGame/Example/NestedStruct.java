// automatically generated by the FlatBuffers compiler, do not modify

package MyGame.Example;

import com.google.flatbuffers.*;
import java.lang.*;
import java.nio.*;
import java.util.*;

@SuppressWarnings("unused")
public final class NestedStruct extends Struct {
  public void __init(int _i, ByteBuffer _bb) {
    __reset(_i, _bb);
  }
  public NestedStruct __assign(int _i, ByteBuffer _bb) {
    __init(_i, _bb);
    return this;
  }

  public int a(int j) {
    return bb.getInt(bb_pos + 0 + j * 4);
  }
  public void mutateA(int j, int a) {
    bb.putInt(bb_pos + 0 + j * 4, a);
  }
  public byte b() {
    return bb.get(bb_pos + 8);
  }
  public void mutateB(byte b) {
    bb.put(bb_pos + 8, b);
  }
  public byte c(int j) {
    return bb.get(bb_pos + 9 + j * 1);
  }
  public void mutateC(int j, byte c) {
    bb.put(bb_pos + 9 + j * 1, c);
  }
  public long d(int j) {
    return bb.getLong(bb_pos + 16 + j * 8);
  }
  public void mutateD(int j, long d) {
    bb.putLong(bb_pos + 16 + j * 8, d);
  }

  public static int createNestedStruct(
      FlatBufferBuilder builder, int[] a, byte b, byte[] c, long[] d) {
    builder.prep(8, 32);
    for (int _idx0 = 2; _idx0 > 0; _idx0--) {
      builder.putLong(d[_idx0 - 1]);
    }
    builder.pad(5);
    for (int _idx0 = 2; _idx0 > 0; _idx0--) {
      builder.putByte(c[_idx0 - 1]);
    }
    builder.putByte(b);
    for (int _idx0 = 2; _idx0 > 0; _idx0--) {
      builder.putInt(a[_idx0 - 1]);
    }
    return builder.offset();
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) {
      __reset(_vector, _element_size, _bb);
      return this;
    }

    public NestedStruct get(int j) {
      return get(new NestedStruct(), j);
    }
    public NestedStruct get(NestedStruct obj, int j) {
      return obj.__assign(__element(j), bb);
    }
  }
  public NestedStructT unpack() {
    NestedStructT _o = new NestedStructT();
    unpackTo(_o);
    return _o;
  }
  public void unpackTo(NestedStructT _o) {
    int[] _oA = _o.getA();
    for (int _j = 0; _j < 2; ++_j) {
      _oA[_j] = a(_j);
    }
    byte _oB = b();
    _o.setB(_oB);
    byte[] _oC = _o.getC();
    for (int _j = 0; _j < 2; ++_j) {
      _oC[_j] = c(_j);
    }
    long[] _oD = _o.getD();
    for (int _j = 0; _j < 2; ++_j) {
      _oD[_j] = d(_j);
    }
  }
  public static int pack(FlatBufferBuilder builder, NestedStructT _o) {
    if (_o == null)
      return 0;
    int[] _a = _o.getA();
    byte[] _c = _o.getC();
    long[] _d = _o.getD();
    return createNestedStruct(builder, _a, _o.getB(), _c, _d);
  }
}
