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

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.util.Optional;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

abstract sealed class Lucene99MemorySegmentByteVectorScorer
    extends RandomVectorScorer.AbstractRandomVectorScorer {

  final int vectorByteSize;
  final MemorySegmentAccessInput input;
  // the whole input as a single segment, or null if it is not available
  final MemorySegment seg;
  final byte[] query;
  final float[] scratchScores = new float[4];
  byte[] scratch;

  /**
   * Return an optional whose value, if present, is the scorer. Otherwise, an empty optional is
   * returned.
   */
  public static Optional<Lucene99MemorySegmentByteVectorScorer> create(
      VectorSimilarityFunction type, IndexInput input, KnnVectorValues values, byte[] queryVector)
      throws IOException {
    assert values instanceof ByteVectorValues;
    input = FilterIndexInput.unwrapOnlyTest(input);
    if (!(input instanceof MemorySegmentAccessInput msInput)) {
      return Optional.empty();
    }
    checkInvariants(values.size(), values.getVectorByteLength(), input);
    return switch (type) {
      case COSINE -> Optional.of(new CosineScorer(msInput, values, queryVector));
      case DOT_PRODUCT -> Optional.of(new DotProductScorer(msInput, values, queryVector));
      case EUCLIDEAN -> Optional.of(new EuclideanScorer(msInput, values, queryVector));
      case MAXIMUM_INNER_PRODUCT ->
          Optional.of(new MaxInnerProductScorer(msInput, values, queryVector));
    };
  }

  Lucene99MemorySegmentByteVectorScorer(
      MemorySegmentAccessInput input, KnnVectorValues values, byte[] queryVector)
      throws IOException {
    super(values);
    this.input = input;
    this.seg = input.segmentSliceOrNull(0L, input.length());
    this.vectorByteSize = values.getVectorByteLength();
    this.query = queryVector;
  }

  final MemorySegment getSegment(int ord) throws IOException {
    checkOrdinal(ord);
    long byteOffset = (long) ord * vectorByteSize;
    MemorySegment seg = input.segmentSliceOrNull(byteOffset, vectorByteSize);
    if (seg == null) {
      if (scratch == null) {
        scratch = new byte[vectorByteSize];
      }
      input.readBytes(byteOffset, scratch, 0, vectorByteSize);
      seg = MemorySegment.ofArray(scratch);
    }
    return seg;
  }

  static void checkInvariants(int maxOrd, int vectorByteLength, IndexInput input) {
    if (input.length() < (long) vectorByteLength * maxOrd) {
      throw new IllegalArgumentException("input length is less than expected vector data");
    }
  }

  final void checkOrdinal(int ord) {
    if (ord < 0 || ord >= maxOrd()) {
      throw new IllegalArgumentException("illegal ordinal: " + ord);
    }
  }

  @Override
  public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
    if (seg == null || Constants.NATIVE_DOT_PRODUCT_ENABLED) {
      return super.bulkScore(nodes, scores, numNodes);
    }
    int i = 0;
    final int limit = numNodes & ~3;
    float maxScore = Float.NEGATIVE_INFINITY;
    for (; i < limit; i += 4) {
      long offset1 = offset(nodes[i]);
      long offset2 = offset(nodes[i + 1]);
      long offset3 = offset(nodes[i + 2]);
      long offset4 = offset(nodes[i + 3]);
      vectorOp(scratchScores, offset1, offset2, offset3, offset4);
      scores[i + 0] = normalizeRawScore(scratchScores[0]);
      maxScore = Math.max(maxScore, scores[i + 0]);
      scores[i + 1] = normalizeRawScore(scratchScores[1]);
      maxScore = Math.max(maxScore, scores[i + 1]);
      scores[i + 2] = normalizeRawScore(scratchScores[2]);
      maxScore = Math.max(maxScore, scores[i + 2]);
      scores[i + 3] = normalizeRawScore(scratchScores[3]);
      maxScore = Math.max(maxScore, scores[i + 3]);
    }
    // Handle remaining 1–3 nodes in bulk (if any)
    int remaining = numNodes - i;
    if (remaining > 0) {
      long offset1 = offset(nodes[i]);
      long offset2 = (remaining > 1) ? offset(nodes[i + 1]) : offset1;
      long offset3 = (remaining > 2) ? offset(nodes[i + 2]) : offset1;
      vectorOp(scratchScores, offset1, offset2, offset3, offset1);
      scores[i] = normalizeRawScore(scratchScores[0]);
      maxScore = Math.max(maxScore, scores[i]);
      if (remaining > 1) {
        scores[i + 1] = normalizeRawScore(scratchScores[1]);
        maxScore = Math.max(maxScore, scores[i + 1]);
      }
      if (remaining > 2) {
        scores[i + 2] = normalizeRawScore(scratchScores[2]);
        maxScore = Math.max(maxScore, scores[i + 2]);
      }
    }
    return maxScore;
  }

  private long offset(int ord) {
    checkOrdinal(ord);
    return (long) ord * vectorByteSize;
  }

  /**
   * Computes the raw similarity between the query and the four vectors at the given offsets in
   * {@link #seg}, storing the results in {@code scores[0..3]}.
   */
  abstract void vectorOp(float[] scores, long offset1, long offset2, long offset3, long offset4);

  /** Converts a raw similarity to a score. */
  abstract float normalizeRawScore(float rawScore);

  static final class CosineScorer extends Lucene99MemorySegmentByteVectorScorer {
    CosineScorer(MemorySegmentAccessInput input, KnnVectorValues values, byte[] query)
        throws IOException {
      super(input, values, query);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw;
      if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
        raw = NativeVectorUtilSupport.cosine(query, getSegment(node));
      } else {
        raw = PanamaVectorUtilSupport.cosine(query, getSegment(node));
      }
      return normalizeRawScore(raw);
    }

    @Override
    void vectorOp(float[] scores, long offset1, long offset2, long offset3, long offset4) {
      MemorySegmentBulkByteVectorOps.cosineBulk(
          scores, query, seg, offset1, offset2, offset3, offset4);
    }

    @Override
    float normalizeRawScore(float rawScore) {
      return (1 + rawScore) / 2;
    }
  }

  static final class DotProductScorer extends Lucene99MemorySegmentByteVectorScorer {
    DotProductScorer(MemorySegmentAccessInput input, KnnVectorValues values, byte[] query)
        throws IOException {
      super(input, values, query);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw;
      if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
        raw = NativeVectorUtilSupport.dotProduct(query, getSegment(node));
      } else {
        raw = PanamaVectorUtilSupport.dotProduct(query, getSegment(node));
      }
      return normalizeRawScore(raw);
    }

    @Override
    void vectorOp(float[] scores, long offset1, long offset2, long offset3, long offset4) {
      MemorySegmentBulkByteVectorOps.dotProductBulk(
          scores, query, seg, offset1, offset2, offset3, offset4);
    }

    @Override
    float normalizeRawScore(float rawScore) {
      // divide by 2 * 2^14 (maximum absolute value of product of 2 signed bytes) * len
      return 0.5f + rawScore / (float) (query.length * (1 << 15));
    }
  }

  static final class EuclideanScorer extends Lucene99MemorySegmentByteVectorScorer {
    EuclideanScorer(MemorySegmentAccessInput input, KnnVectorValues values, byte[] query)
        throws IOException {
      super(input, values, query);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw;
      if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
        raw = NativeVectorUtilSupport.squareDistance(query, getSegment(node));
      } else {
        raw = PanamaVectorUtilSupport.squareDistance(query, getSegment(node));
      }
      return normalizeRawScore(raw);
    }

    @Override
    void vectorOp(float[] scores, long offset1, long offset2, long offset3, long offset4) {
      MemorySegmentBulkByteVectorOps.squareDistanceBulk(
          scores, query, seg, offset1, offset2, offset3, offset4);
    }

    @Override
    float normalizeRawScore(float rawScore) {
      return 1 / (1f + rawScore);
    }
  }

  static final class MaxInnerProductScorer extends Lucene99MemorySegmentByteVectorScorer {
    MaxInnerProductScorer(MemorySegmentAccessInput input, KnnVectorValues values, byte[] query)
        throws IOException {
      super(input, values, query);
    }

    @Override
    public float score(int node) throws IOException {
      checkOrdinal(node);
      float raw;
      if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
        raw = NativeVectorUtilSupport.dotProduct(query, getSegment(node));
      } else {
        raw = PanamaVectorUtilSupport.dotProduct(query, getSegment(node));
      }
      return normalizeRawScore(raw);
    }

    @Override
    void vectorOp(float[] scores, long offset1, long offset2, long offset3, long offset4) {
      MemorySegmentBulkByteVectorOps.dotProductBulk(
          scores, query, seg, offset1, offset2, offset3, offset4);
    }

    @Override
    float normalizeRawScore(float rawScore) {
      if (rawScore < 0) {
        return 1 / (1 + -1 * rawScore);
      }
      return rawScore + 1;
    }
  }
}
