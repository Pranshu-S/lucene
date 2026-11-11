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
package org.apache.lucene.benchmark.jmh;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.lucene.benchmark.jmh.VectorScorerFloat32Benchmark.ThrowingFlatVectorScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene95.OffHeapByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
// first iteration is complete garbage, so make sure we really warmup
@Warmup(iterations = 4, time = 1)
// real iterations. not useful to spend tons of time here, better to fork more
@Measurement(iterations = 5, time = 1)
// engage some noise reduction
@Fork(
    value = 3,
    jvmArgsAppend = {
      "-Xmx2g",
      "-Xms2g",
      "-XX:+AlwaysPreTouch",
      "--add-modules=jdk.incubator.vector"
    })
/*
 * Benchmark to compare scoring int8 vectors one at a time to bulk scoring them, with the optimized
 * scorers, against both a query vector (search) and an indexed vector (graph building).
 */
public class VectorScorerInt8Benchmark {

  @Param({"1024"})
  public int size;

  @Param({"DOT_PRODUCT", "COSINE", "EUCLIDEAN", "MAXIMUM_INNER_PRODUCT"})
  public VectorSimilarityFunction sim;

  @Param({"10000"})
  public int numVectors;

  float[] scores;
  int[] indices;
  Path path;
  Directory dir;
  IndexInput in;
  RandomVectorScorer queryScorer;
  UpdateableRandomVectorScorer supplierScorer;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var random = ThreadLocalRandom.current();
    path = Files.createTempDirectory("VectorScorerInt8Benchmark");
    dir = new MMapDirectory(path);
    try (IndexOutput out = dir.createOutput("vector.data", IOContext.DEFAULT)) {
      var ba = new byte[size];
      for (int v = 0; v < numVectors; v++) {
        random.nextBytes(ba);
        out.writeBytes(ba, 0, ba.length);
      }
    }
  }

  @Setup(Level.Iteration)
  public void perIterationInit() throws IOException {
    var random = ThreadLocalRandom.current();
    scores = new float[numVectors];
    in = dir.openInput("vector.data", IOContext.DEFAULT);
    var values = vectorValues(size, numVectors, in, sim);
    var scorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();

    byte[] query = new byte[size];
    random.nextBytes(query);
    queryScorer = scorer.getRandomVectorScorer(sim, values.copy(), query);
    supplierScorer = scorer.getRandomVectorScorerSupplier(sim, values.copy()).scorer();
    supplierScorer.setScoringOrdinal(random.nextInt(numVectors));

    List<Integer> list = IntStream.range(0, numVectors).boxed().collect(Collectors.toList());
    Collections.shuffle(list, random);
    indices = list.stream().mapToInt(i -> i).toArray();
  }

  @TearDown(Level.Iteration)
  public void perIterationTeardown() throws IOException {
    IOUtils.close(in);
  }

  @TearDown(Level.Trial)
  public void teardown() throws IOException {
    dir.deleteFile("vector.data");
    IOUtils.close(dir);
    Files.delete(path);
  }

  @Benchmark
  public float[] queryScore() throws IOException {
    for (int v = 0; v < numVectors; v++) {
      scores[v] = queryScorer.score(indices[v]);
    }
    return scores;
  }

  @Benchmark
  public float[] queryBulkScore() throws IOException {
    queryScorer.bulkScore(indices, scores, numVectors);
    return scores;
  }

  @Benchmark
  public float[] supplierScore() throws IOException {
    for (int v = 0; v < numVectors; v++) {
      scores[v] = supplierScorer.score(indices[v]);
    }
    return scores;
  }

  @Benchmark
  public float[] supplierBulkScore() throws IOException {
    supplierScorer.bulkScore(indices, scores, numVectors);
    return scores;
  }

  static KnnVectorValues vectorValues(
      int dims, int size, IndexInput in, VectorSimilarityFunction sim) throws IOException {
    return new OffHeapByteVectorValues.DenseOffHeapVectorValues(
        dims, size, in.slice("test", 0, in.length()), dims, new ThrowingFlatVectorScorer(), sim);
  }
}
