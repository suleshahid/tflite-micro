// automatically generated by the FlatBuffers compiler, do not modify

import com.google.flatbuffers.*;
import java.lang.*;
import java.nio.*;
import java.util.*;

public class MovieT {
  private CharacterUnion mainCharacter;
  private CharacterUnion[] characters;

  public CharacterUnion getMainCharacter() {
    return mainCharacter;
  }

  public void setMainCharacter(CharacterUnion mainCharacter) {
    this.mainCharacter = mainCharacter;
  }

  public CharacterUnion[] getCharacters() {
    return characters;
  }

  public void setCharacters(CharacterUnion[] characters) {
    this.characters = characters;
  }

  public MovieT() {
    this.mainCharacter = null;
    this.characters = null;
  }
  public static MovieT deserializeFromBinary(byte[] fbBuffer) {
    return Movie.getRootAsMovie(ByteBuffer.wrap(fbBuffer)).unpack();
  }
  public byte[] serializeToBinary() {
    FlatBufferBuilder fbb = new FlatBufferBuilder();
    Movie.finishMovieBuffer(fbb, Movie.pack(fbb, this));
    return fbb.sizedByteArray();
  }
}
