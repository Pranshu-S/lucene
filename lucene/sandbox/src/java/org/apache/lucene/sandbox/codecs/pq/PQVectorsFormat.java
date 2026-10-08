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
package org.apache.lucene.sandbox.codecs.pq;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.NoReuseHint;

/**
 * A flat vectors format that stores FLOAT32 vectors compressed with 8-bit product quantization
 * (PQ8), next to the full-precision vectors.
 *
 * <p>Each segment trains its own codebooks, per field, with k-means on a bounded sample of the
 * segment's vectors, and encodes every vector as {@code m} one-byte codes, one per subspace. Merges
 * retrain on the merged vectors and re-encode them from full precision, so codes never depend on
 * the codebooks of the segments being merged.
 *
 * <p>Queries score the codes with asymmetric distance computation (ADC): the query is kept at full
 * precision and a per-query table of its distances to every centroid turns each distance into
 * {@code m} lookups. Full-precision vectors are kept in a {@link Lucene99FlatVectorsFormat} for
 * rescoring, for merging, and for building an HNSW graph when this format is wrapped by {@link
 * HnswPQVectorsFormat}: the graph is built on full-precision distances, only searched with ADC.
 *
 * <p>Only {@link org.apache.lucene.index.VectorEncoding#FLOAT32} fields with {@link
 * org.apache.lucene.index.VectorSimilarityFunction#EUCLIDEAN} similarity are supported. Indexing
 * any other field with this format throws {@link IllegalArgumentException}.
 *
 * <h2>Files</h2>
 *
 * <ul>
 *   <li><b>.vpqm</b> (metadata), per field: field number, encoding, similarity, dimension, vector
 *       count and, when there are vectors, the offset and length of the codes, followed by the
 *       quantizer: number of subspaces, number of centroids and the codebooks as little-endian
 *       floats ({@code dimension * numCentroids} values). Ends with field number -1.
 *   <li><b>.vpq</b> (codes): for each field, {@code count * numSubspaces} bytes, in vector ordinal
 *       order, the same order as the full-precision vectors.
 *   <li>The {@link Lucene99FlatVectorsFormat} files (.vec, .vemf) for the full-precision vectors.
 * </ul>
 *
 * @lucene.experimental
 */
public class PQVectorsFormat extends FlatVectorsFormat {

  public static final String NAME = "PQVectorsFormat";

  /** The infoStream component this format logs to. */
  public static final String PQ_COMPONENT = "PQVEC";

  /** Default number of dimensions per subspace, so codes are {@code dimension / 4} bytes. */
  public static final int DEFAULT_DIMENSIONS_PER_SUBSPACE = 4;

  /**
   * Default upper bound on the vectors k-means trains on. This is about 39 vectors per centroid, a
   * common rule of thumb for 256 centroids.
   */
  public static final int DEFAULT_SAMPLE_SIZE = 10_000;

  /** Default maximum number of k-means iterations per subspace. */
  public static final int DEFAULT_TRAINING_ITERATIONS = 20;

  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;
  static final String META_CODEC_NAME = "PQVectorsFormatMeta";
  static final String VECTOR_DATA_CODEC_NAME = "PQVectorsFormatData";
  static final String META_EXTENSION = "vpqm";
  static final String VECTOR_DATA_EXTENSION = "vpq";

  /** Seed for sampling and k-means++, fixed so that indexing is reproducible. */
  static final long TRAINING_SEED = 0x5EEDL;

  private static final FlatVectorsFormat rawVectorFormat =
      new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  private static final PQVectorScorer scorer =
      new PQVectorScorer(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());

  private final int dimensionsPerSubspace;
  private final int sampleSize;
  private final int trainingIterations;

  /** Creates a format with default parameters. */
  public PQVectorsFormat() {
    this(DEFAULT_DIMENSIONS_PER_SUBSPACE, DEFAULT_SAMPLE_SIZE, DEFAULT_TRAINING_ITERATIONS);
  }

  /**
   * Creates a format with the given parameters.
   *
   * @param dimensionsPerSubspace how many dimensions each one-byte code covers. A field of
   *     dimension {@code d} gets {@code ceil(d / dimensionsPerSubspace)} subspaces, so this sets
   *     the compression: smaller values store more bytes per vector and estimate distances better.
   * @param sampleSize the maximum number of vectors k-means trains on, per segment and field
   * @param trainingIterations the maximum number of k-means iterations per subspace
   */
  public PQVectorsFormat(int dimensionsPerSubspace, int sampleSize, int trainingIterations) {
    super(NAME);
    if (dimensionsPerSubspace < 1) {
      throw new IllegalArgumentException(
          "dimensionsPerSubspace must be positive; got " + dimensionsPerSubspace);
    }
    if (sampleSize < ProductQuantizer.MAX_CENTROIDS) {
      throw new IllegalArgumentException(
          "sampleSize must be at least " + ProductQuantizer.MAX_CENTROIDS + "; got " + sampleSize);
    }
    if (trainingIterations < 1) {
      throw new IllegalArgumentException(
          "trainingIterations must be positive; got " + trainingIterations);
    }
    this.dimensionsPerSubspace = dimensionsPerSubspace;
    this.sampleSize = sampleSize;
    this.trainingIterations = trainingIterations;
  }

  /** Returns the number of subspaces, which is the code size in bytes, for a dimension. */
  public int numSubspaces(int dimension) {
    return Math.ceilDiv(dimension, dimensionsPerSubspace);
  }

  @Override
  public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new PQVectorsWriter(
        state, this, rawVectorFormat.fieldsWriter(rescoreOnly(state)), scorer);
  }

  @Override
  public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new PQVectorsReader(state, rawVectorFormat.fieldsReader(rescoreOnly(state)), scorer);
  }

  int sampleSize() {
    return sampleSize;
  }

  int trainingIterations() {
    return trainingIterations;
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return NAME
        + "(dimensionsPerSubspace="
        + dimensionsPerSubspace
        + ", sampleSize="
        + sampleSize
        + ", trainingIterations="
        + trainingIterations
        + ", rawVectorFormat="
        + rawVectorFormat
        + ")";
  }

  /** The full-precision vectors are read back only to rescore, merge, or build graphs. */
  private static SegmentReadState rescoreOnly(SegmentReadState state) {
    return new SegmentReadState(state, state.context.union(NoReuseHint.INSTANCE));
  }

  private static SegmentWriteState rescoreOnly(SegmentWriteState state) {
    return new SegmentWriteState(state, state.context.union(NoReuseHint.INSTANCE));
  }
}
