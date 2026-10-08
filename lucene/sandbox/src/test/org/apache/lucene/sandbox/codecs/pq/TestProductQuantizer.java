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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.ByteBuffersDataInput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;

public class TestProductQuantizer extends LuceneTestCase {

  private static float[][] randomVectors(Random random, int n, int dim) {
    float[][] vectors = new float[n][dim];
    for (float[] v : vectors) {
      for (int j = 0; j < dim; j++) {
        v[j] = random.nextFloat() * 2 - 1;
      }
    }
    return vectors;
  }

  /** Vectors drawn around a few cluster centers, so k-means has structure to find. */
  private static float[][] clusteredVectors(Random random, int n, int dim, int numClusters) {
    float[][] centers = randomVectors(random, numClusters, dim);
    float[][] vectors = new float[n][dim];
    for (int i = 0; i < n; i++) {
      float[] center = centers[random.nextInt(numClusters)];
      for (int j = 0; j < dim; j++) {
        vectors[i][j] = center[j] + (float) random.nextGaussian() * 0.05f;
      }
    }
    return vectors;
  }

  public void testSubspaceOffsetsCoverDimension() {
    int[] offsets = ProductQuantizer.subspaceOffsets(10, 4);
    assertArrayEquals(new int[] {0, 2, 5, 7, 10}, offsets);
    for (int trial = 0; trial < 100; trial++) {
      int dim = 1 + random().nextInt(1024);
      int m = 1 + random().nextInt(dim);
      offsets = ProductQuantizer.subspaceOffsets(dim, m);
      assertEquals(0, offsets[0]);
      assertEquals(dim, offsets[m]);
      for (int s = 0; s < m; s++) {
        int width = offsets[s + 1] - offsets[s];
        assertTrue(width == dim / m || width == dim / m + 1);
      }
    }
  }

  /** With no more vectors than centroids, every training vector is its own centroid. */
  public void testSmallSampleIsExact() {
    int dim = 2 + random().nextInt(30);
    int n = 1 + random().nextInt(ProductQuantizer.MAX_CENTROIDS);
    int m = 1 + random().nextInt(dim);
    float[][] vectors = randomVectors(random(), n, dim);
    ProductQuantizer pq = ProductQuantizer.train(vectors, dim, m, 5, random().nextLong());
    assertEquals(n, pq.numCentroids());
    byte[] code = new byte[m];
    float[] scratch = new float[pq.numCentroids()];
    float[] decoded = new float[dim];
    for (float[] v : vectors) {
      pq.encode(v, code, scratch);
      pq.decode(code, decoded);
      assertArrayEquals(v, decoded, 0f);
    }
  }

  /** The ADC estimate is the exact squared distance from the query to the decoded vector. */
  public void testAdcMatchesDistanceToReconstruction() {
    int dim = 16 + random().nextInt(48);
    int m = 1 + random().nextInt(dim);
    float[][] vectors = randomVectors(random(), 1000, dim);
    ProductQuantizer pq = ProductQuantizer.train(vectors, dim, m, 5, random().nextLong());
    float[] table = new float[m * pq.numCentroids()];
    float[] scratch = new float[pq.numCentroids()];
    byte[] code = new byte[m];
    float[] decoded = new float[dim];
    for (int trial = 0; trial < 50; trial++) {
      float[] query = randomVectors(random(), 1, dim)[0];
      pq.distanceTable(query, table);
      float[] v = vectors[random().nextInt(vectors.length)];
      pq.encode(v, code, scratch);
      pq.decode(code, decoded);
      float expected = VectorUtil.squareDistance(query, decoded);
      float actual = ProductQuantizer.squareDistance(table, pq.numCentroids(), code, m);
      assertEquals(expected, actual, 1e-4f * Math.max(1f, expected));
    }
  }

  /** Each sub-code is the nearest centroid of its subspace. */
  public void testEncodePicksNearestCentroid() {
    int dim = 24;
    int m = 6;
    float[][] vectors = randomVectors(random(), 600, dim);
    ProductQuantizer pq = ProductQuantizer.train(vectors, dim, m, 5, random().nextLong());
    int[] offsets = ProductQuantizer.subspaceOffsets(dim, m);
    float[] codebooks = pq.codebooks();
    int k = pq.numCentroids();
    byte[] code = new byte[m];
    float[] scratch = new float[k];
    float[] v = randomVectors(random(), 1, dim)[0];
    pq.encode(v, code, scratch);
    for (int s = 0; s < m; s++) {
      int width = offsets[s + 1] - offsets[s];
      int base = offsets[s] * k;
      float best = Float.MAX_VALUE;
      for (int c = 0; c < k; c++) {
        float d = 0;
        for (int j = 0; j < width; j++) {
          float diff = v[offsets[s] + j] - codebooks[base + c * width + j];
          d += diff * diff;
        }
        best = Math.min(best, d);
      }
      int chosen = Byte.toUnsignedInt(code[s]);
      float chosenDistance = 0;
      for (int j = 0; j < width; j++) {
        float diff = v[offsets[s] + j] - codebooks[base + chosen * width + j];
        chosenDistance += diff * diff;
      }
      assertEquals(best, chosenDistance, 1e-5f);
    }
  }

  /** Training on clustered data reconstructs far better than a random codebook would. */
  public void testTrainingReducesError() {
    int dim = 32;
    int m = 8;
    float[][] vectors = clusteredVectors(random(), 4000, dim, 20);
    ProductQuantizer trained = ProductQuantizer.train(vectors, dim, m, 20, 42);
    // a "codebook" of 256 random training vectors, no iterations of refinement
    float[][] seeds = Arrays.copyOf(vectors, 256);
    ProductQuantizer untrained = ProductQuantizer.train(seeds, dim, m, 1, 42);
    assertTrue(
        reconstructionError(trained, vectors) < 0.8 * reconstructionError(untrained, vectors));
  }

  private static double reconstructionError(ProductQuantizer pq, float[][] vectors) {
    byte[] code = new byte[pq.numSubspaces()];
    float[] scratch = new float[pq.numCentroids()];
    float[] decoded = new float[pq.dimension()];
    double error = 0;
    for (float[] v : vectors) {
      pq.encode(v, code, scratch);
      pq.decode(code, decoded);
      error += VectorUtil.squareDistance(v, decoded);
    }
    return error;
  }

  public void testTrainingIsDeterministic() {
    float[][] vectors = randomVectors(random(), 2000, 20);
    long seed = random().nextLong();
    ProductQuantizer a = ProductQuantizer.train(vectors, 20, 5, 10, seed);
    ProductQuantizer b = ProductQuantizer.train(vectors, 20, 5, 10, seed);
    assertArrayEquals(a.codebooks(), b.codebooks(), 0f);
  }

  public void testSerializationRoundTrip() throws Exception {
    int dim = 3 + random().nextInt(40);
    int m = 1 + random().nextInt(dim);
    float[][] vectors = randomVectors(random(), 300 + random().nextInt(300), dim);
    ProductQuantizer pq = ProductQuantizer.train(vectors, dim, m, 5, random().nextLong());
    ByteBuffersDataOutput out = new ByteBuffersDataOutput();
    pq.write(out);
    ProductQuantizer read = ProductQuantizer.read(new ByteBuffersDataInput(out.toBufferList()));
    assertEquals(pq.dimension(), read.dimension());
    assertEquals(pq.numSubspaces(), read.numSubspaces());
    assertEquals(pq.numCentroids(), read.numCentroids());
    assertArrayEquals(pq.codebooks(), read.codebooks(), 0f);
  }

  public void testReservoirSample() throws Exception {
    int dim = 4;
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < 1000; i++) {
      vectors.add(new float[] {i, i, i, i});
    }
    FloatVectorValues values = FloatVectorValues.fromFloats(vectors, dim);
    ProductQuantizer.Sample sample = ProductQuantizer.reservoirSample(values, 100, 7);
    assertEquals(1000, sample.numVectors());
    assertEquals(100, sample.vectors().length);
    boolean[] seen = new boolean[1000];
    for (float[] v : sample.vectors()) {
      int i = (int) v[0];
      assertFalse("sampled twice: " + i, seen[i]);
      seen[i] = true;
    }
    // fewer vectors than the sample size: take them all
    sample =
        ProductQuantizer.reservoirSample(
            FloatVectorValues.fromFloats(vectors.subList(0, 10), dim), 100, 7);
    assertEquals(10, sample.numVectors());
    assertEquals(10, sample.vectors().length);
  }

  public void testIllegalArguments() {
    float[][] vectors = randomVectors(random(), 10, 4);
    expectThrows(
        IllegalArgumentException.class, () -> ProductQuantizer.train(new float[0][], 4, 2, 5, 0));
    expectThrows(IllegalArgumentException.class, () -> ProductQuantizer.train(vectors, 4, 5, 5, 0));
    expectThrows(IllegalArgumentException.class, () -> ProductQuantizer.train(vectors, 4, 2, 0, 0));
    expectThrows(IllegalArgumentException.class, () -> new PQVectorsFormat(0, 10_000, 10));
    expectThrows(IllegalArgumentException.class, () -> new PQVectorsFormat(4, 100, 10));
    expectThrows(IllegalArgumentException.class, () -> new PQVectorsFormat(4, 10_000, 0));
  }
}
