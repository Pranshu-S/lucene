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
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

/** A score supplier of vectors whose element size is byte. */
public abstract sealed class Lucene99MemorySegmentByteVectorScorerSupplier
    implements RandomVectorScorerSupplier {
  final int vectorByteSize;
  final int maxOrd;
  final MemorySegmentAccessInput input;
  // the whole input as a single segment, or null if it is not available
  final MemorySegment seg;
  final KnnVectorValues values; // to support ordToDoc/getAcceptOrds
  byte[] scratch1, scratch2;

  /**
   * Return an optional whose value, if present, is the scorer supplier. Otherwise, an empty
   * optional is returned.
   */
  static Optional<RandomVectorScorerSupplier> create(
      VectorSimilarityFunction type, IndexInput input, KnnVectorValues values) throws IOException {
    assert values instanceof ByteVectorValues;
    input = FilterIndexInput.unwrapOnlyTest(input);
    if (!(input instanceof MemorySegmentAccessInput msInput)) {
      return Optional.empty();
    }
    checkInvariants(values.size(), values.getVectorByteLength(), input);
    return switch (type) {
      case COSINE -> Optional.of(new CosineSupplier(msInput, values));
      case DOT_PRODUCT -> Optional.of(new DotProductSupplier(msInput, values));
      case EUCLIDEAN -> Optional.of(new EuclideanSupplier(msInput, values));
      case MAXIMUM_INNER_PRODUCT -> Optional.of(new MaxInnerProductSupplier(msInput, values));
    };
  }

  Lucene99MemorySegmentByteVectorScorerSupplier(
      MemorySegmentAccessInput input, KnnVectorValues values) throws IOException {
    this.input = input;
    this.seg = input.segmentSliceOrNull(0L, input.length());
    this.values = values;
    this.vectorByteSize = values.getVectorByteLength();
    this.maxOrd = values.size();
  }

  static void checkInvariants(int maxOrd, int vectorByteLength, IndexInput input) {
    if (input.length() < (long) vectorByteLength * maxOrd) {
      throw new IllegalArgumentException("input length is less than expected vector data");
    }
  }

  final void checkOrdinal(int ord) {
    if (ord < 0 || ord >= maxOrd) {
      throw new IllegalArgumentException("illegal ordinal: " + ord);
    }
  }

  final MemorySegment getFirstSegment(int ord) throws IOException {
    long byteOffset = (long) ord * vectorByteSize;
    MemorySegment seg = input.segmentSliceOrNull(byteOffset, vectorByteSize);
    if (seg == null) {
      if (scratch1 == null) {
        scratch1 = new byte[vectorByteSize];
      }
      input.readBytes(byteOffset, scratch1, 0, vectorByteSize);
      seg = MemorySegment.ofArray(scratch1);
    }
    return seg;
  }

  final MemorySegment getSecondSegment(int ord) throws IOException {
    long byteOffset = (long) ord * vectorByteSize;
    MemorySegment seg = input.segmentSliceOrNull(byteOffset, vectorByteSize);
    if (seg == null) {
      if (scratch2 == null) {
        scratch2 = new byte[vectorByteSize];
      }
      input.readBytes(byteOffset, scratch2, 0, vectorByteSize);
      seg = MemorySegment.ofArray(scratch2);
    }
    return seg;
  }

  static final class CosineSupplier extends Lucene99MemorySegmentByteVectorScorerSupplier {

    CosineSupplier(MemorySegmentAccessInput input, KnnVectorValues values) throws IOException {
      super(input, values);
    }

    @Override
    public UpdateableRandomVectorScorer scorer() {
      return new BulkScorer(values) {
        @Override
        public float score(int node) throws IOException {
          checkOrdinal(node);
          float raw;
          if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
            raw = NativeVectorUtilSupport.cosine(getFirstSegment(queryOrd), getSecondSegment(node));
          } else {
            raw = PanamaVectorUtilSupport.cosine(getFirstSegment(queryOrd), getSecondSegment(node));
          }
          return normalizeRawScore(raw);
        }

        @Override
        void vectorOp(
            float[] scores,
            long queryOffset,
            long offset1,
            long offset2,
            long offset3,
            long offset4) {
          MemorySegmentBulkByteVectorOps.cosineBulk(
              scores, seg, queryOffset, offset1, offset2, offset3, offset4, vectorByteSize);
        }

        @Override
        float normalizeRawScore(float rawScore) {
          return (1 + rawScore) / 2;
        }
      };
    }

    @Override
    public CosineSupplier copy() throws IOException {
      return new CosineSupplier(input.clone(), values);
    }
  }

  static final class DotProductSupplier extends Lucene99MemorySegmentByteVectorScorerSupplier {

    DotProductSupplier(MemorySegmentAccessInput input, KnnVectorValues values) throws IOException {
      super(input, values);
    }

    @Override
    public UpdateableRandomVectorScorer scorer() {
      return new BulkScorer(values) {
        @Override
        public float score(int node) throws IOException {
          checkOrdinal(node);
          float raw;
          if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
            raw =
                NativeVectorUtilSupport.dotProduct(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          } else {
            raw =
                PanamaVectorUtilSupport.dotProduct(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          }
          return normalizeRawScore(raw);
        }

        @Override
        void vectorOp(
            float[] scores,
            long queryOffset,
            long offset1,
            long offset2,
            long offset3,
            long offset4) {
          MemorySegmentBulkByteVectorOps.dotProductBulk(
              scores, seg, queryOffset, offset1, offset2, offset3, offset4, vectorByteSize);
        }

        @Override
        float normalizeRawScore(float rawScore) {
          // divide by 2 * 2^14 (maximum absolute value of product of 2 signed bytes) * len
          return 0.5f + rawScore / (float) (values.dimension() * (1 << 15));
        }
      };
    }

    @Override
    public DotProductSupplier copy() throws IOException {
      return new DotProductSupplier(input.clone(), values);
    }
  }

  static final class EuclideanSupplier extends Lucene99MemorySegmentByteVectorScorerSupplier {

    EuclideanSupplier(MemorySegmentAccessInput input, KnnVectorValues values) throws IOException {
      super(input, values);
    }

    @Override
    public UpdateableRandomVectorScorer scorer() {
      return new BulkScorer(values) {
        @Override
        public float score(int node) throws IOException {
          checkOrdinal(node);
          float raw;
          if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
            raw =
                NativeVectorUtilSupport.squareDistance(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          } else {
            raw =
                PanamaVectorUtilSupport.squareDistance(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          }
          return normalizeRawScore(raw);
        }

        @Override
        void vectorOp(
            float[] scores,
            long queryOffset,
            long offset1,
            long offset2,
            long offset3,
            long offset4) {
          MemorySegmentBulkByteVectorOps.squareDistanceBulk(
              scores, seg, queryOffset, offset1, offset2, offset3, offset4, vectorByteSize);
        }

        @Override
        float normalizeRawScore(float rawScore) {
          return 1 / (1f + rawScore);
        }
      };
    }

    @Override
    public EuclideanSupplier copy() throws IOException {
      return new EuclideanSupplier(input.clone(), values);
    }
  }

  static final class MaxInnerProductSupplier extends Lucene99MemorySegmentByteVectorScorerSupplier {

    MaxInnerProductSupplier(MemorySegmentAccessInput input, KnnVectorValues values)
        throws IOException {
      super(input, values);
    }

    @Override
    public UpdateableRandomVectorScorer scorer() {
      return new BulkScorer(values) {
        @Override
        public float score(int node) throws IOException {
          checkOrdinal(node);
          float raw;
          if (Constants.NATIVE_DOT_PRODUCT_ENABLED) {
            raw =
                NativeVectorUtilSupport.dotProduct(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          } else {
            raw =
                PanamaVectorUtilSupport.dotProduct(
                    getFirstSegment(queryOrd), getSecondSegment(node));
          }
          return normalizeRawScore(raw);
        }

        @Override
        void vectorOp(
            float[] scores,
            long queryOffset,
            long offset1,
            long offset2,
            long offset3,
            long offset4) {
          MemorySegmentBulkByteVectorOps.dotProductBulk(
              scores, seg, queryOffset, offset1, offset2, offset3, offset4, vectorByteSize);
        }

        @Override
        float normalizeRawScore(float rawScore) {
          if (rawScore < 0) {
            return 1 / (1 + -1 * rawScore);
          }
          return rawScore + 1;
        }
      };
    }

    @Override
    public MaxInnerProductSupplier copy() throws IOException {
      return new MaxInnerProductSupplier(input.clone(), values);
    }
  }

  /**
   * Base class of the scorers returned by {@link #scorer()}, which scores four vectors at a time
   * against the scoring ordinal in {@link #bulkScore} when the vector data is a single segment.
   */
  abstract class BulkScorer
      extends UpdateableRandomVectorScorer.AbstractUpdateableRandomVectorScorer {
    int queryOrd = 0;
    final float[] scratchScores = new float[4];

    BulkScorer(KnnVectorValues values) {
      super(values);
    }

    /**
     * Computes the raw similarity between the vector at {@code queryOffset} and the four vectors at
     * the given offsets in {@link #seg}, storing the results in {@code scores[0..3]}.
     */
    abstract void vectorOp(
        float[] scores, long queryOffset, long offset1, long offset2, long offset3, long offset4);

    /** Converts a raw similarity to a score. */
    abstract float normalizeRawScore(float rawScore);

    @Override
    public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
      if (seg == null || Constants.NATIVE_DOT_PRODUCT_ENABLED) {
        return super.bulkScore(nodes, scores, numNodes);
      }
      int i = 0;
      final long queryOffset = (long) queryOrd * vectorByteSize;
      final int limit = numNodes & ~3;
      float maxScore = Float.NEGATIVE_INFINITY;
      for (; i < limit; i += 4) {
        long offset1 = offset(nodes[i]);
        long offset2 = offset(nodes[i + 1]);
        long offset3 = offset(nodes[i + 2]);
        long offset4 = offset(nodes[i + 3]);
        vectorOp(scratchScores, queryOffset, offset1, offset2, offset3, offset4);
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
        vectorOp(scratchScores, queryOffset, offset1, offset2, offset3, offset1);
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

    @Override
    public void setScoringOrdinal(int node) {
      checkOrdinal(node);
      queryOrd = node;
    }
  }
}
