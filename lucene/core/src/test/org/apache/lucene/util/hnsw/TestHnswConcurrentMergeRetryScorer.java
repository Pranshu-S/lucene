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
package org.apache.lucene.util.hnsw;

import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.tests.util.LuceneTestCase;

/**
 * Tests that a concurrent merge worker that loses the entry node promotion to another worker
 * retries the insertion of its node with a scorer that scores against that node.
 */
public class TestHnswConcurrentMergeRetryScorer extends LuceneTestCase {

  // one-dimensional vectors, scored with 1 / (1 + d^2)
  private static final float[] X = {0, 1, 3, 7};

  // worker 1's task, run inline by worker 0 to force the interleaving under test
  private Runnable forkedWorker1;

  private static float sim(int a, int b) {
    float d = X[a] - X[b];
    return 1 / (1 + d * d);
  }

  /**
   * A scorer that runs worker 1 the first time worker 0 points it away from node 1, i.e. while
   * worker 0 links node 1 and before it tries to promote node 1 to entry node.
   */
  private final class Scorer implements UpdateableRandomVectorScorer {
    private int ord = -1;

    @Override
    public void setScoringOrdinal(int node) {
      if (ord == 1 && node != 1 && forkedWorker1 != null) {
        Runnable worker1 = forkedWorker1;
        forkedWorker1 = null;
        worker1.run();
      }
      ord = node;
    }

    @Override
    public float score(int node) {
      return sim(ord, node);
    }

    @Override
    public int maxOrd() {
      return X.length;
    }
  }

  public void testLostPromotionRetryScoresAgainstInsertedNode() throws Exception {
    RandomVectorScorerSupplier supplier =
        new RandomVectorScorerSupplier() {
          @Override
          public UpdateableRandomVectorScorer scorer() {
            return new Scorer();
          }

          @Override
          public RandomVectorScorerSupplier copy() {
            return this;
          }
        };
    OnHeapHnswGraph graph = new OnHeapHnswGraph(2, X.length);
    long savedRandSeed = HnswGraphBuilder.randSeed;
    try {
      // with this seed node 1 (worker 0) and node 3 (worker 1) both open a new top level
      HnswGraphBuilder.randSeed = 42;
      HnswConcurrentMergeBuilder builder =
          new HnswConcurrentMergeBuilder(
              new TaskExecutor(r -> forkedWorker1 = r), 2, supplier, 10, graph, null);
      builder.setBatchSize(2);
      builder.build(X.length);
    } finally {
      HnswGraphBuilder.randSeed = savedRandSeed;
    }

    // make sure the interleaving under test happened: node 3 won the promotion node 1 lost
    assertEquals(3, graph.entryNode());
    for (int level = 0; level < graph.numLevels(); level++) {
      for (int node = 0; node < X.length; node++) {
        if (graph.nodeExistAtLevel(level, node)) {
          NeighborArray neighbors = graph.getNeighbors(level, node);
          for (int i = 0; i < neighbors.size(); i++) {
            int neighbor = neighbors.nodes()[i];
            String edge = "level " + level + ": " + node + " -> " + neighbor;
            assertEquals(edge, sim(node, neighbor), neighbors.getScores(i), 0f);
          }
        }
      }
    }
  }
}
