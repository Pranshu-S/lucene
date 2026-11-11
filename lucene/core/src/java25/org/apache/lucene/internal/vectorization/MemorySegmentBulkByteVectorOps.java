/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.internal.vectorization;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static jdk.incubator.vector.VectorOperators.ADD;
import static jdk.incubator.vector.VectorOperators.ASHR;
import static jdk.incubator.vector.VectorOperators.B2S;
import static jdk.incubator.vector.VectorOperators.LSHL;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Bulk comparison operations for signed byte (int8) vectors stored in a {@link MemorySegment}.
 *
 * <p>Each operation compares one query vector, either on heap or in the same segment as the
 * documents, against four document vectors at once, so that each query chunk is loaded once per
 * four documents and the four independent accumulator chains can overlap in the CPU pipeline. All
 * arithmetic is integer, so results are exactly equal to the corresponding single-vector functions
 * in {@link PanamaVectorUtilSupport}.
 *
 * <p>Every vector width uses the same scheme: half a register of bytes is sign-extended to a full
 * register of shorts and multiplied (or subtracted) in 16 bits. The result is then reinterpreted as
 * ints, each holding two adjacent shorts, which are sign-extended with shifts for accumulation in
 * 32 bits. This is much cheaper than widening the upper half of the shorts with a conversion, which
 * is slow on some platforms (e.g. ARM NEON). The vector code is deliberately written out in full,
 * rather than factored into helper methods, since a helper that is not inlined forces its vectors
 * to be boxed.
 */
final class MemorySegmentBulkByteVectorOps {

  private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;
  private static final VectorSpecies<Byte> BYTE_SPECIES =
      ByteVector.SPECIES_MAX.withShape(
          VectorShape.forBitSize(PanamaVectorConstants.PREFERRED_VECTOR_BITSIZE >> 1));
  private static final VectorSpecies<Short> SHORT_SPECIES =
      ShortVector.SPECIES_MAX.withShape(
          VectorShape.forBitSize(PanamaVectorConstants.PREFERRED_VECTOR_BITSIZE));
  private static final VectorSpecies<Integer> INT_SPECIES =
      PanamaVectorConstants.PRERERRED_INT_SPECIES;

  private MemorySegmentBulkByteVectorOps() {}

  /**
   * Computes the dot product of the query {@code q} with each of the four documents at offsets
   * {@code d1..d4} in {@code seg}, storing the results in {@code scores[0..3]}.
   */
  static void dotProductBulk(
      float[] scores, byte[] q, MemorySegment seg, long d1, long d2, long d3, long d4) {
    dotProductBulkImpl(scores, q, -1L, seg, d1, d2, d3, d4, q.length);
  }

  /**
   * Computes the dot product of the query at offset {@code q} in {@code seg} with each of the four
   * documents at offsets {@code d1..d4} in {@code seg}, storing the results in {@code
   * scores[0..3]}.
   */
  static void dotProductBulk(
      float[] scores, MemorySegment seg, long q, long d1, long d2, long d3, long d4, int length) {
    dotProductBulkImpl(scores, null, q, seg, d1, d2, d3, d4, length);
  }

  private static void dotProductBulkImpl(
      float[] scores,
      byte[] qArray,
      long qOffset,
      MemorySegment seg,
      long d1,
      long d2,
      long d3,
      long d4,
      int length) {
    IntVector acc1 = IntVector.zero(INT_SPECIES);
    IntVector acc2 = IntVector.zero(INT_SPECIES);
    IntVector acc3 = IntVector.zero(INT_SPECIES);
    IntVector acc4 = IntVector.zero(INT_SPECIES);
    int i = 0;
    final int limit = BYTE_SPECIES.loopBound(length);
    for (; i < limit; i += BYTE_SPECIES.length()) {
      // sign-extend half a register of bytes to a full register of shorts
      ByteVector qv8;
      if (qArray != null) {
        qv8 = ByteVector.fromArray(BYTE_SPECIES, qArray, i);
      } else {
        qv8 = ByteVector.fromMemorySegment(BYTE_SPECIES, seg, qOffset + i, LE);
      }
      ShortVector qv = (ShortVector) qv8.convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv1 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d1 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv2 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d2 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv3 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d3 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv4 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d4 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      // 16-bit multiply, then add both shorts of each int lane into the accumulators
      IntVector prod1 = qv.mul(dv1).reinterpretAsInts();
      IntVector prod2 = qv.mul(dv2).reinterpretAsInts();
      IntVector prod3 = qv.mul(dv3).reinterpretAsInts();
      IntVector prod4 = qv.mul(dv4).reinterpretAsInts();
      acc1 = acc1.add(prod1.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod1.lanewise(ASHR, 16));
      acc2 = acc2.add(prod2.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod2.lanewise(ASHR, 16));
      acc3 = acc3.add(prod3.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod3.lanewise(ASHR, 16));
      acc4 = acc4.add(prod4.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod4.lanewise(ASHR, 16));
    }
    int sum1 = acc1.reduceLanes(ADD);
    int sum2 = acc2.reduceLanes(ADD);
    int sum3 = acc3.reduceLanes(ADD);
    int sum4 = acc4.reduceLanes(ADD);
    for (; i < length; i++) {
      final byte qValue = qArray != null ? qArray[i] : seg.get(JAVA_BYTE, qOffset + i);
      sum1 += qValue * seg.get(JAVA_BYTE, d1 + i);
      sum2 += qValue * seg.get(JAVA_BYTE, d2 + i);
      sum3 += qValue * seg.get(JAVA_BYTE, d3 + i);
      sum4 += qValue * seg.get(JAVA_BYTE, d4 + i);
    }
    scores[0] = sum1;
    scores[1] = sum2;
    scores[2] = sum3;
    scores[3] = sum4;
  }

  /**
   * Computes the cosine similarity of the query {@code q} with each of the four documents at
   * offsets {@code d1..d4} in {@code seg}, storing the results in {@code scores[0..3]}.
   */
  static void cosineBulk(
      float[] scores, byte[] q, MemorySegment seg, long d1, long d2, long d3, long d4) {
    cosineBulkImpl(scores, q, -1L, seg, d1, d2, d3, d4, q.length);
  }

  /**
   * Computes the cosine similarity of the query at offset {@code q} in {@code seg} with each of the
   * four documents at offsets {@code d1..d4} in {@code seg}, storing the results in {@code
   * scores[0..3]}.
   */
  static void cosineBulk(
      float[] scores, MemorySegment seg, long q, long d1, long d2, long d3, long d4, int length) {
    cosineBulkImpl(scores, null, q, seg, d1, d2, d3, d4, length);
  }

  private static void cosineBulkImpl(
      float[] scores,
      byte[] qArray,
      long qOffset,
      MemorySegment seg,
      long d1,
      long d2,
      long d3,
      long d4,
      int length) {
    IntVector acc1 = IntVector.zero(INT_SPECIES);
    IntVector acc2 = IntVector.zero(INT_SPECIES);
    IntVector acc3 = IntVector.zero(INT_SPECIES);
    IntVector acc4 = IntVector.zero(INT_SPECIES);
    IntVector qAccNorm = IntVector.zero(INT_SPECIES);
    IntVector accNorm1 = IntVector.zero(INT_SPECIES);
    IntVector accNorm2 = IntVector.zero(INT_SPECIES);
    IntVector accNorm3 = IntVector.zero(INT_SPECIES);
    IntVector accNorm4 = IntVector.zero(INT_SPECIES);
    int i = 0;
    final int limit = BYTE_SPECIES.loopBound(length);
    for (; i < limit; i += BYTE_SPECIES.length()) {
      // sign-extend half a register of bytes to a full register of shorts
      ByteVector qv8;
      if (qArray != null) {
        qv8 = ByteVector.fromArray(BYTE_SPECIES, qArray, i);
      } else {
        qv8 = ByteVector.fromMemorySegment(BYTE_SPECIES, seg, qOffset + i, LE);
      }
      ShortVector qv = (ShortVector) qv8.convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv1 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d1 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv2 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d2 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv3 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d3 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv4 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d4 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      // 16-bit multiply, then add both shorts of each int lane into the accumulators
      IntVector qNorm32 = qv.mul(qv).reinterpretAsInts();
      IntVector prod1 = qv.mul(dv1).reinterpretAsInts();
      IntVector norm1_32 = dv1.mul(dv1).reinterpretAsInts();
      IntVector prod2 = qv.mul(dv2).reinterpretAsInts();
      IntVector norm2_32 = dv2.mul(dv2).reinterpretAsInts();
      IntVector prod3 = qv.mul(dv3).reinterpretAsInts();
      IntVector norm3_32 = dv3.mul(dv3).reinterpretAsInts();
      IntVector prod4 = qv.mul(dv4).reinterpretAsInts();
      IntVector norm4_32 = dv4.mul(dv4).reinterpretAsInts();
      qAccNorm =
          qAccNorm
              .add(qNorm32.lanewise(LSHL, 16).lanewise(ASHR, 16))
              .add(qNorm32.lanewise(ASHR, 16));
      acc1 = acc1.add(prod1.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod1.lanewise(ASHR, 16));
      accNorm1 =
          accNorm1
              .add(norm1_32.lanewise(LSHL, 16).lanewise(ASHR, 16))
              .add(norm1_32.lanewise(ASHR, 16));
      acc2 = acc2.add(prod2.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod2.lanewise(ASHR, 16));
      accNorm2 =
          accNorm2
              .add(norm2_32.lanewise(LSHL, 16).lanewise(ASHR, 16))
              .add(norm2_32.lanewise(ASHR, 16));
      acc3 = acc3.add(prod3.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod3.lanewise(ASHR, 16));
      accNorm3 =
          accNorm3
              .add(norm3_32.lanewise(LSHL, 16).lanewise(ASHR, 16))
              .add(norm3_32.lanewise(ASHR, 16));
      acc4 = acc4.add(prod4.lanewise(LSHL, 16).lanewise(ASHR, 16)).add(prod4.lanewise(ASHR, 16));
      accNorm4 =
          accNorm4
              .add(norm4_32.lanewise(LSHL, 16).lanewise(ASHR, 16))
              .add(norm4_32.lanewise(ASHR, 16));
    }
    int sum1 = acc1.reduceLanes(ADD);
    int sum2 = acc2.reduceLanes(ADD);
    int sum3 = acc3.reduceLanes(ADD);
    int sum4 = acc4.reduceLanes(ADD);
    int qNorm = qAccNorm.reduceLanes(ADD);
    int norm1 = accNorm1.reduceLanes(ADD);
    int norm2 = accNorm2.reduceLanes(ADD);
    int norm3 = accNorm3.reduceLanes(ADD);
    int norm4 = accNorm4.reduceLanes(ADD);
    for (; i < length; i++) {
      final byte qValue = qArray != null ? qArray[i] : seg.get(JAVA_BYTE, qOffset + i);
      final byte dValue1 = seg.get(JAVA_BYTE, d1 + i);
      final byte dValue2 = seg.get(JAVA_BYTE, d2 + i);
      final byte dValue3 = seg.get(JAVA_BYTE, d3 + i);
      final byte dValue4 = seg.get(JAVA_BYTE, d4 + i);
      qNorm += qValue * qValue;
      sum1 += qValue * dValue1;
      norm1 += dValue1 * dValue1;
      sum2 += qValue * dValue2;
      norm2 += dValue2 * dValue2;
      sum3 += qValue * dValue3;
      norm3 += dValue3 * dValue3;
      sum4 += qValue * dValue4;
      norm4 += dValue4 * dValue4;
    }
    scores[0] = (float) (sum1 / Math.sqrt((double) qNorm * (double) norm1));
    scores[1] = (float) (sum2 / Math.sqrt((double) qNorm * (double) norm2));
    scores[2] = (float) (sum3 / Math.sqrt((double) qNorm * (double) norm3));
    scores[3] = (float) (sum4 / Math.sqrt((double) qNorm * (double) norm4));
  }

  /**
   * Computes the squared euclidean distance between the query {@code q} and each of the four
   * documents at offsets {@code d1..d4} in {@code seg}, storing the results in {@code
   * scores[0..3]}.
   */
  static void squareDistanceBulk(
      float[] scores, byte[] q, MemorySegment seg, long d1, long d2, long d3, long d4) {
    squareDistanceBulkImpl(scores, q, -1L, seg, d1, d2, d3, d4, q.length);
  }

  /**
   * Computes the squared euclidean distance between the query at offset {@code q} in {@code seg}
   * and each of the four documents at offsets {@code d1..d4} in {@code seg}, storing the results in
   * {@code scores[0..3]}.
   */
  static void squareDistanceBulk(
      float[] scores, MemorySegment seg, long q, long d1, long d2, long d3, long d4, int length) {
    squareDistanceBulkImpl(scores, null, q, seg, d1, d2, d3, d4, length);
  }

  private static void squareDistanceBulkImpl(
      float[] scores,
      byte[] qArray,
      long qOffset,
      MemorySegment seg,
      long d1,
      long d2,
      long d3,
      long d4,
      int length) {
    IntVector acc1 = IntVector.zero(INT_SPECIES);
    IntVector acc2 = IntVector.zero(INT_SPECIES);
    IntVector acc3 = IntVector.zero(INT_SPECIES);
    IntVector acc4 = IntVector.zero(INT_SPECIES);
    int i = 0;
    final int limit = BYTE_SPECIES.loopBound(length);
    for (; i < limit; i += BYTE_SPECIES.length()) {
      // sign-extend half a register of bytes to a full register of shorts
      ByteVector qv8;
      if (qArray != null) {
        qv8 = ByteVector.fromArray(BYTE_SPECIES, qArray, i);
      } else {
        qv8 = ByteVector.fromMemorySegment(BYTE_SPECIES, seg, qOffset + i, LE);
      }
      ShortVector qv = (ShortVector) qv8.convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv1 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d1 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv2 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d2 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv3 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d3 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      ShortVector dv4 =
          (ShortVector)
              ByteVector.fromMemorySegment(BYTE_SPECIES, seg, d4 + i, LE)
                  .convertShape(B2S, SHORT_SPECIES, 0);
      // the difference of two signed bytes fits in a short, but its square does not: 16-bit
      // subtract, then extract both shorts of each int lane to multiply in 32 bits
      IntVector diff1 = qv.sub(dv1).reinterpretAsInts();
      IntVector diff2 = qv.sub(dv2).reinterpretAsInts();
      IntVector diff3 = qv.sub(dv3).reinterpretAsInts();
      IntVector diff4 = qv.sub(dv4).reinterpretAsInts();
      IntVector lo1 = diff1.lanewise(LSHL, 16).lanewise(ASHR, 16);
      IntVector lo2 = diff2.lanewise(LSHL, 16).lanewise(ASHR, 16);
      IntVector lo3 = diff3.lanewise(LSHL, 16).lanewise(ASHR, 16);
      IntVector lo4 = diff4.lanewise(LSHL, 16).lanewise(ASHR, 16);
      IntVector hi1 = diff1.lanewise(ASHR, 16);
      IntVector hi2 = diff2.lanewise(ASHR, 16);
      IntVector hi3 = diff3.lanewise(ASHR, 16);
      IntVector hi4 = diff4.lanewise(ASHR, 16);
      acc1 = acc1.add(lo1.mul(lo1)).add(hi1.mul(hi1));
      acc2 = acc2.add(lo2.mul(lo2)).add(hi2.mul(hi2));
      acc3 = acc3.add(lo3.mul(lo3)).add(hi3.mul(hi3));
      acc4 = acc4.add(lo4.mul(lo4)).add(hi4.mul(hi4));
    }
    int sum1 = acc1.reduceLanes(ADD);
    int sum2 = acc2.reduceLanes(ADD);
    int sum3 = acc3.reduceLanes(ADD);
    int sum4 = acc4.reduceLanes(ADD);
    for (; i < length; i++) {
      final byte qValue = qArray != null ? qArray[i] : seg.get(JAVA_BYTE, qOffset + i);
      final int diff1 = qValue - seg.get(JAVA_BYTE, d1 + i);
      final int diff2 = qValue - seg.get(JAVA_BYTE, d2 + i);
      final int diff3 = qValue - seg.get(JAVA_BYTE, d3 + i);
      final int diff4 = qValue - seg.get(JAVA_BYTE, d4 + i);
      sum1 += diff1 * diff1;
      sum2 += diff2 * diff2;
      sum3 += diff3 * diff3;
      sum4 += diff4 * diff4;
    }
    scores[0] = sum1;
    scores[1] = sum2;
    scores[2] = sum3;
    scores[3] = sum4;
  }
}
