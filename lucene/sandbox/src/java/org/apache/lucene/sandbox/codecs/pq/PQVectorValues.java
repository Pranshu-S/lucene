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
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * A field's full-precision vectors together with their product-quantized codes. Reads return the
 * full-precision vectors, while {@link #scorer} and {@link #adcScorer} estimate distances from the
 * codes alone; {@link #rescorer} uses full precision.
 *
 * @lucene.experimental
 */
public final class PQVectorValues extends FloatVectorValues {

  private final FloatVectorValues fullPrecision;
  private final ProductQuantizer quantizer;
  private final IndexInput codes;

  /**
   * @param fullPrecision the full-precision vectors, which also define ordinals and documents
   * @param quantizer the quantizer the codes were encoded with
   * @param codes {@code size() * quantizer.numSubspaces()} bytes of codes, in ordinal order
   */
  PQVectorValues(FloatVectorValues fullPrecision, ProductQuantizer quantizer, IndexInput codes) {
    assert codes.length() == (long) fullPrecision.size() * quantizer.numSubspaces();
    this.fullPrecision = fullPrecision;
    this.quantizer = quantizer;
    this.codes = codes;
  }

  /** The full-precision vectors. */
  public FloatVectorValues fullPrecisionValues() {
    return fullPrecision;
  }

  /** The quantizer the codes were encoded with. */
  public ProductQuantizer quantizer() {
    return quantizer;
  }

  /** Reads the code of {@code ord} into {@code code}. */
  public void codeValue(int ord, byte[] code) throws IOException {
    readCode(codes, ord, code, quantizer.numSubspaces());
  }

  @Override
  public int dimension() {
    return fullPrecision.dimension();
  }

  @Override
  public int size() {
    return fullPrecision.size();
  }

  @Override
  public float[] vectorValue(int ord) throws IOException {
    return fullPrecision.vectorValue(ord);
  }

  @Override
  public boolean prefetch(int ord, int count) throws IOException {
    // vectorValue() and rescorer() read the full-precision vectors
    return fullPrecision.prefetch(ord, count);
  }

  @Override
  public PQVectorValues copy() throws IOException {
    return new PQVectorValues(fullPrecision.copy(), quantizer, codes.clone());
  }

  @Override
  public int ordToDoc(int ord) {
    return fullPrecision.ordToDoc(ord);
  }

  @Override
  public Bits getAcceptOrds(Bits acceptDocs) {
    return fullPrecision.getAcceptOrds(acceptDocs);
  }

  @Override
  public DocIndexIterator iterator() {
    return fullPrecision.iterator();
  }

  /** Returns a scorer that estimates the similarity of {@code target} to vectors by ordinal. */
  public RandomVectorScorer adcScorer(float[] target) throws IOException {
    return new ADCScorer(this, quantizer, codes.clone(), target);
  }

  @Override
  public VectorScorer scorer(float[] target) throws IOException {
    if (size() == 0) {
      return null;
    }
    ADCScorer adc = new ADCScorer(this, quantizer, codes.clone(), target);
    DocIndexIterator iterator = fullPrecision.iterator();
    return new VectorScorer() {
      @Override
      public float score() throws IOException {
        return adc.score(iterator.index());
      }

      @Override
      public DocIdSetIterator iterator() {
        return iterator;
      }
    };
  }

  @Override
  public VectorScorer rescorer(float[] target) throws IOException {
    return fullPrecision.rescorer(target);
  }

  private static void readCode(IndexInput codes, int ord, byte[] code, int codeLength)
      throws IOException {
    codes.seek((long) ord * codeLength);
    codes.readBytes(code, 0, codeLength);
  }

  /** Converts a squared Euclidean distance to a score, as {@code EUCLIDEAN} similarity does. */
  static float euclideanScore(float squareDistance) {
    return 1 / (1 + squareDistance);
  }

  /**
   * Asymmetric distance computation: the query's distance to every centroid is computed once, and
   * the distance to an encoded vector is the sum of its subspaces' table entries.
   */
  private static final class ADCScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
    private final IndexInput codes;
    private final float[] table;
    private final byte[] code;
    private final int numCentroids;
    private final int numSubspaces;

    ADCScorer(PQVectorValues values, ProductQuantizer quantizer, IndexInput codes, float[] target) {
      super(values);
      this.codes = codes;
      this.numCentroids = quantizer.numCentroids();
      this.numSubspaces = quantizer.numSubspaces();
      this.table = new float[numSubspaces * numCentroids];
      this.code = new byte[numSubspaces];
      quantizer.distanceTable(target, table);
    }

    @Override
    public float score(int node) throws IOException {
      readCode(codes, node, code, numSubspaces);
      return euclideanScore(
          ProductQuantizer.squareDistance(table, numCentroids, code, numSubspaces));
    }
  }
}
