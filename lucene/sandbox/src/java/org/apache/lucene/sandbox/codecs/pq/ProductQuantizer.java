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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * An 8-bit product quantizer (PQ8). A vector is split into {@code m} contiguous subspaces, and each
 * subspace is encoded as the index of its nearest centroid among at most {@value #MAX_CENTROIDS}
 * centroids learned for that subspace by k-means. A code is therefore {@code m} bytes.
 *
 * <p>Distances are estimated asymmetrically (ADC): the query stays at full precision, and {@link
 * #distanceTable} computes its squared distance to every centroid of every subspace once, so that
 * the squared distance to an encoded vector costs {@code m} table lookups, see {@link
 * #squareDistance}.
 *
 * <p>When {@code dimension} is not a multiple of {@code m}, subspace widths differ by at most one.
 *
 * @lucene.experimental
 */
public final class ProductQuantizer implements Accountable {

  /** The maximum number of centroids per subspace, so that a sub-code fits in one byte. */
  public static final int MAX_CENTROIDS = 256;

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(ProductQuantizer.class);

  /** Training stops once an iteration improves the k-means objective by less than this ratio. */
  private static final double CONVERGENCE_TOLERANCE = 1e-4;

  private final int dimension;
  private final int numSubspaces;
  private final int numCentroids;
  private final int[] offsets;

  /** Subspace {@code s} starts at {@code offsets[s] * numCentroids}, one row per centroid. */
  private final float[] codebooks;

  /**
   * The codebooks transposed per subspace: component {@code j} of centroid {@code k} of subspace
   * {@code s} is at {@code offsets[s] * numCentroids + j * numCentroids + k}. Scoring every
   * centroid then runs over contiguous memory, which the JIT vectorizes.
   */
  private final float[] transposed;

  /**
   * Creates a quantizer from trained codebooks.
   *
   * @param dimension the vector dimension
   * @param numSubspaces the number of subspaces, which is the code length in bytes
   * @param numCentroids the number of centroids per subspace
   * @param codebooks the centroids, laid out as described on {@link #codebooks()}
   */
  public ProductQuantizer(int dimension, int numSubspaces, int numCentroids, float[] codebooks) {
    if (numSubspaces < 1 || numSubspaces > dimension) {
      throw new IllegalArgumentException(
          "numSubspaces must be in [1, " + dimension + "]; got " + numSubspaces);
    }
    if (numCentroids < 1 || numCentroids > MAX_CENTROIDS) {
      throw new IllegalArgumentException(
          "numCentroids must be in [1, " + MAX_CENTROIDS + "]; got " + numCentroids);
    }
    if (codebooks.length != dimension * numCentroids) {
      throw new IllegalArgumentException(
          "expected " + dimension * numCentroids + " codebook values; got " + codebooks.length);
    }
    this.dimension = dimension;
    this.numSubspaces = numSubspaces;
    this.numCentroids = numCentroids;
    this.offsets = subspaceOffsets(dimension, numSubspaces);
    this.codebooks = codebooks;
    this.transposed = new float[codebooks.length];
    for (int s = 0; s < numSubspaces; s++) {
      int width = offsets[s + 1] - offsets[s];
      int base = offsets[s] * numCentroids;
      transpose(codebooks, base, transposed, base, numCentroids, width);
    }
  }

  /** The vector dimension. */
  public int dimension() {
    return dimension;
  }

  /** The number of subspaces, which is also the length of a code in bytes. */
  public int numSubspaces() {
    return numSubspaces;
  }

  /** The number of centroids per subspace. */
  public int numCentroids() {
    return numCentroids;
  }

  /**
   * The centroids of all subspaces. Those of subspace {@code s} start at {@code offset(s) *
   * numCentroids()}, one row of {@code width(s)} values per centroid.
   */
  float[] codebooks() {
    return codebooks;
  }

  /**
   * Returns the subspace boundaries for the given dimension: subspace {@code s} spans dimensions
   * {@code [offsets[s], offsets[s + 1])}.
   */
  static int[] subspaceOffsets(int dimension, int numSubspaces) {
    int[] offsets = new int[numSubspaces + 1];
    for (int s = 0; s <= numSubspaces; s++) {
      offsets[s] = (int) ((long) s * dimension / numSubspaces);
    }
    return offsets;
  }

  /**
   * Encodes {@code vector} into {@code code}, one byte per subspace holding the index of its
   * nearest centroid.
   *
   * @param scratch a buffer of at least {@link #numCentroids()} floats
   */
  public void encode(float[] vector, byte[] code, float[] scratch) {
    assert vector.length == dimension;
    assert code.length >= numSubspaces;
    for (int s = 0; s < numSubspaces; s++) {
      int start = offsets[s];
      int width = offsets[s + 1] - start;
      code[s] =
          (byte)
              nearestCentroid(
                  vector, start, width, transposed, start * numCentroids, numCentroids, scratch);
    }
  }

  /** Writes into {@code vector} the reconstruction of {@code code}. */
  public void decode(byte[] code, float[] vector) {
    for (int s = 0; s < numSubspaces; s++) {
      int start = offsets[s];
      int width = offsets[s + 1] - start;
      int centroid = Byte.toUnsignedInt(code[s]);
      System.arraycopy(codebooks, start * numCentroids + centroid * width, vector, start, width);
    }
  }

  /**
   * Fills {@code table} with the squared distances from {@code query} to every centroid: the
   * distance from subspace {@code s} of the query to centroid {@code k} is at {@code s *
   * numCentroids() + k}.
   *
   * @param table a buffer of at least {@code numSubspaces() * numCentroids()} floats
   */
  public void distanceTable(float[] query, float[] table) {
    assert query.length == dimension;
    for (int s = 0; s < numSubspaces; s++) {
      int start = offsets[s];
      int width = offsets[s + 1] - start;
      squareDistances(
          query,
          start,
          width,
          transposed,
          start * numCentroids,
          numCentroids,
          table,
          s * numCentroids);
    }
  }

  /**
   * Returns the estimated squared distance between the query that {@code table} was built for and
   * the vector encoded by the {@code numSubspaces} bytes of {@code code}.
   */
  public static float squareDistance(
      float[] table, int numCentroids, byte[] code, int numSubspaces) {
    float distance = 0;
    for (int s = 0, base = 0; s < numSubspaces; s++, base += numCentroids) {
      distance += table[base + Byte.toUnsignedInt(code[s])];
    }
    return distance;
  }

  /** Writes this quantizer to {@code out}; {@link #read} reads it back. */
  public void write(DataOutput out) throws IOException {
    out.writeVInt(dimension);
    out.writeVInt(numSubspaces);
    out.writeVInt(numCentroids);
    for (float value : codebooks) {
      out.writeInt(Float.floatToIntBits(value));
    }
  }

  /** Reads a quantizer written by {@link #write}. */
  public static ProductQuantizer read(DataInput in) throws IOException {
    int dimension = in.readVInt();
    int numSubspaces = in.readVInt();
    int numCentroids = in.readVInt();
    if (numSubspaces < 1
        || numSubspaces > dimension
        || numCentroids < 1
        || numCentroids > MAX_CENTROIDS) {
      throw new CorruptIndexException(
          "invalid product quantizer: dimension="
              + dimension
              + ", numSubspaces="
              + numSubspaces
              + ", numCentroids="
              + numCentroids,
          in);
    }
    float[] codebooks = new float[dimension * numCentroids];
    in.readFloats(codebooks, 0, codebooks.length);
    return new ProductQuantizer(dimension, numSubspaces, numCentroids, codebooks);
  }

  @Override
  public long ramBytesUsed() {
    return SHALLOW_SIZE
        + RamUsageEstimator.sizeOf(offsets)
        + RamUsageEstimator.sizeOf(codebooks)
        + RamUsageEstimator.sizeOf(transposed);
  }

  @Override
  public String toString() {
    return "ProductQuantizer(dimension="
        + dimension
        + ", numSubspaces="
        + numSubspaces
        + ", numCentroids="
        + numCentroids
        + ")";
  }

  /**
   * Trains a quantizer with k-means, run independently on each subspace of {@code sample}.
   * Initialization is k-means++. Each subspace gets {@code min(sample.length, 256)} centroids, so a
   * sample of at most 256 vectors is reproduced exactly.
   *
   * @param sample the training vectors, which are not modified
   * @param dimension the vector dimension
   * @param numSubspaces the number of subspaces
   * @param maxIterations the maximum number of Lloyd iterations per subspace
   * @param seed the seed for k-means++ initialization
   */
  public static ProductQuantizer train(
      float[][] sample, int dimension, int numSubspaces, int maxIterations, long seed) {
    if (sample.length == 0) {
      throw new IllegalArgumentException("cannot train on an empty sample");
    }
    if (maxIterations < 1) {
      throw new IllegalArgumentException("maxIterations must be positive; got " + maxIterations);
    }
    int numVectors = sample.length;
    int numCentroids = Math.min(MAX_CENTROIDS, numVectors);
    int[] offsets = subspaceOffsets(dimension, numSubspaces);
    float[] codebooks = new float[dimension * numCentroids];
    for (int s = 0; s < numSubspaces; s++) {
      int start = offsets[s];
      int width = offsets[s + 1] - start;
      float[] points = new float[numVectors * width];
      for (int i = 0; i < numVectors; i++) {
        System.arraycopy(sample[i], start, points, i * width, width);
      }
      float[] centroids =
          numCentroids == numVectors
              ? points
              : kMeans(
                  points, numVectors, width, numCentroids, maxIterations, new Random(seed + s));
      System.arraycopy(centroids, 0, codebooks, start * numCentroids, numCentroids * width);
    }
    return new ProductQuantizer(dimension, numSubspaces, numCentroids, codebooks);
  }

  /**
   * Draws a uniform sample of at most {@code sampleSize} vectors in a single forward pass over
   * {@code values}, so it works on merged values that can only be iterated.
   *
   * @return the sampled vectors (copies), and the number of vectors seen
   */
  static Sample reservoirSample(FloatVectorValues values, int sampleSize, long seed)
      throws IOException {
    Random random = new Random(seed);
    float[][] reservoir = new float[sampleSize][];
    int count = 0;
    KnnVectorValues.DocIndexIterator iterator = values.iterator();
    for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
      int slot = count < sampleSize ? count : random.nextInt(count + 1);
      if (slot < sampleSize) {
        float[] vector = values.vectorValue(iterator.index());
        if (reservoir[slot] == null) {
          reservoir[slot] = vector.clone();
        } else {
          System.arraycopy(vector, 0, reservoir[slot], 0, vector.length);
        }
      }
      count++;
    }
    return new Sample(
        count < sampleSize ? ArrayUtil.copyOfSubArray(reservoir, 0, count) : reservoir, count);
  }

  /** A training sample and the number of vectors it was drawn from. */
  record Sample(float[][] vectors, int numVectors) {}

  /** Lloyd's k-means over {@code n} points of {@code width} values, from a k-means++ seeding. */
  private static float[] kMeans(
      float[] points, int n, int width, int k, int maxIterations, Random random) {
    float[] centroids = kMeansPlusPlus(points, n, width, k, random);
    float[] transposed = new float[k * width];
    float[] scratch = new float[k];
    int[] assignments = new int[n];
    float[] distances = new float[n];
    double[] sums = new double[k * width];
    int[] counts = new int[k];
    double previousCost = Double.POSITIVE_INFINITY;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      transpose(centroids, 0, transposed, 0, k, width);
      double cost = 0;
      for (int i = 0; i < n; i++) {
        int best = nearestCentroid(points, i * width, width, transposed, 0, k, scratch);
        assignments[i] = best;
        distances[i] = scratch[best];
        cost += distances[i];
      }
      Arrays.fill(sums, 0);
      Arrays.fill(counts, 0);
      for (int i = 0; i < n; i++) {
        int c = assignments[i];
        counts[c]++;
        for (int j = 0; j < width; j++) {
          sums[c * width + j] += points[i * width + j];
        }
      }
      for (int c = 0; c < k; c++) {
        if (counts[c] > 0) {
          for (int j = 0; j < width; j++) {
            centroids[c * width + j] = (float) (sums[c * width + j] / counts[c]);
          }
        } else {
          // Re-seed an empty cluster at the point worst served by the current centroids.
          int farthest = 0;
          for (int i = 1; i < n; i++) {
            if (distances[i] > distances[farthest]) {
              farthest = i;
            }
          }
          System.arraycopy(points, farthest * width, centroids, c * width, width);
          distances[farthest] = 0;
        }
      }
      if (previousCost - cost <= CONVERGENCE_TOLERANCE * cost) {
        break;
      }
      previousCost = cost;
    }
    return centroids;
  }

  private static float[] kMeansPlusPlus(float[] points, int n, int width, int k, Random random) {
    float[] centroids = new float[k * width];
    float[] minDistances = new float[n];
    int first = random.nextInt(n);
    System.arraycopy(points, first * width, centroids, 0, width);
    for (int i = 0; i < n; i++) {
      minDistances[i] = squareDistance(points, i * width, centroids, 0, width);
    }
    for (int c = 1; c < k; c++) {
      double total = 0;
      for (int i = 0; i < n; i++) {
        total += minDistances[i];
      }
      int next;
      if (total == 0) {
        // every point coincides with a centroid already; any choice is as good
        next = random.nextInt(n);
      } else {
        double target = random.nextDouble() * total;
        double cumulative = 0;
        next = n - 1;
        for (int i = 0; i < n; i++) {
          cumulative += minDistances[i];
          if (cumulative >= target && minDistances[i] > 0) {
            next = i;
            break;
          }
        }
      }
      System.arraycopy(points, next * width, centroids, c * width, width);
      for (int i = 0; i < n; i++) {
        float distance = squareDistance(points, i * width, centroids, c * width, width);
        if (distance < minDistances[i]) {
          minDistances[i] = distance;
        }
      }
    }
    return centroids;
  }

  /**
   * Writes into {@code out[outOffset + c]} the squared distance from {@code width} values of {@code
   * x} at {@code xOffset} to each centroid {@code c} of a transposed codebook. Differences are
   * squared directly rather than expanded as {@code ||x||^2 + ||c||^2 - 2 x.c}, which loses the
   * precision needed to tell close centroids apart. The inner loop runs over contiguous centroids,
   * which the JIT vectorizes.
   */
  private static void squareDistances(
      float[] x,
      int xOffset,
      int width,
      float[] transposed,
      int transposedOffset,
      int numCentroids,
      float[] out,
      int outOffset) {
    Arrays.fill(out, outOffset, outOffset + numCentroids, 0f);
    for (int j = 0; j < width; j++) {
      float xj = x[xOffset + j];
      int row = transposedOffset + j * numCentroids;
      for (int c = 0; c < numCentroids; c++) {
        float diff = xj - transposed[row + c];
        out[outOffset + c] += diff * diff;
      }
    }
  }

  /**
   * Returns the centroid nearest to {@code width} values of {@code x} at {@code xOffset}, leaving
   * the squared distance to every centroid in {@code scratch}.
   */
  private static int nearestCentroid(
      float[] x,
      int xOffset,
      int width,
      float[] transposed,
      int transposedOffset,
      int numCentroids,
      float[] scratch) {
    squareDistances(x, xOffset, width, transposed, transposedOffset, numCentroids, scratch, 0);
    int best = 0;
    for (int c = 1; c < numCentroids; c++) {
      if (scratch[c] < scratch[best]) {
        best = c;
      }
    }
    return best;
  }

  private static void transpose(
      float[] rows, int rowsOffset, float[] out, int outOffset, int numRows, int width) {
    for (int r = 0; r < numRows; r++) {
      for (int j = 0; j < width; j++) {
        out[outOffset + j * numRows + r] = rows[rowsOffset + r * width + j];
      }
    }
  }

  private static float squareDistance(float[] a, int aOffset, float[] b, int bOffset, int width) {
    float distance = 0;
    for (int j = 0; j < width; j++) {
      float diff = a[aOffset + j] - b[bOffset + j];
      distance += diff * diff;
    }
    return distance;
  }
}
