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
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/**
 * Scores {@link PQVectorValues} against a query with asymmetric distance computation, and anything
 * else with the full-precision delegate.
 *
 * <p>Vector-to-vector scoring, which HNSW graph construction uses, always runs on the
 * full-precision vectors, including when it is handed {@link PQVectorValues} during a merge.
 *
 * @lucene.experimental
 */
public class PQVectorScorer implements FlatVectorsScorer {

  private final FlatVectorsScorer fullPrecisionScorer;

  /** Creates a scorer that delegates full-precision scoring to {@code fullPrecisionScorer}. */
  public PQVectorScorer(FlatVectorsScorer fullPrecisionScorer) {
    this.fullPrecisionScorer = fullPrecisionScorer;
  }

  @Override
  public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
      VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues)
      throws IOException {
    if (vectorValues instanceof PQVectorValues pqValues) {
      vectorValues = pqValues.fullPrecisionValues();
    }
    return fullPrecisionScorer.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues, float[] target)
      throws IOException {
    if (vectorValues instanceof PQVectorValues pqValues) {
      FlatVectorsScorer.checkDimensions(target.length, pqValues.dimension());
      return pqValues.adcScorer(target);
    }
    return fullPrecisionScorer.getRandomVectorScorer(similarityFunction, vectorValues, target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues, byte[] target)
      throws IOException {
    return fullPrecisionScorer.getRandomVectorScorer(similarityFunction, vectorValues, target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues, short[] target)
      throws IOException {
    return fullPrecisionScorer.getRandomVectorScorer(similarityFunction, vectorValues, target);
  }

  @Override
  public String toString() {
    return "PQVectorScorer(fullPrecisionScorer=" + fullPrecisionScorer + ")";
  }
}
