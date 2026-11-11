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

import static org.apache.lucene.tests.util.RamUsageTester.ramUsed;

import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.tests.util.LuceneTestCase;

/**
 * Test OnHeapHnswGraph's behavior specifically, for more complex test, see {@link
 * HnswGraphTestCase}
 */
public class TestOnHeapHnswGraph extends LuceneTestCase {

  /* assert exception will be thrown when we add out of bound node to a fixed size graph */
  public void testNoGrowth() {
    OnHeapHnswGraph graph = new OnHeapHnswGraph(10, 100);
    expectThrows(IllegalStateException.class, () -> graph.addNode(1, 100));
  }

  /* assert exception will be thrown when we call getNodeOnLevel for an incomplete graph */
  public void testIncompleteGraphThrow() {
    OnHeapHnswGraph graph = new OnHeapHnswGraph(10, -1);
    graph.addNode(1, 0);
    graph.addNode(0, 0);
    assertEquals(1, graph.getNodesOnLevel(1).size());
    graph.addNode(0, 5);
    expectThrows(IllegalStateException.class, () -> graph.getNodesOnLevel(0));
  }

  public void testGraphGrowth() {
    OnHeapHnswGraph graph = new OnHeapHnswGraph(10, -1);
    List<List<Integer>> levelToNodes = new ArrayList<>();
    int maxLevel = 5;
    for (int i = 0; i < maxLevel; i++) {
      levelToNodes.add(new ArrayList<>());
    }
    for (int i = 0; i < 101; i++) {
      int level = random().nextInt(maxLevel);
      for (int l = level; l >= 0; l--) {
        graph.addNode(l, i);
        graph.trySetNewEntryNode(i, l);
        if (l > graph.numLevels() - 1) {
          graph.tryPromoteNewEntryNode(i, l, graph.numLevels() - 1);
        }
        levelToNodes.get(l).add(i);
      }
    }
    assertGraphEquals(graph, levelToNodes);
  }

  public void testGraphBuildOutOfOrder() {
    OnHeapHnswGraph graph = new OnHeapHnswGraph(10, -1);
    List<List<Integer>> levelToNodes = new ArrayList<>();
    int maxLevel = 5;
    int numNodes = 100;
    for (int i = 0; i < maxLevel; i++) {
      levelToNodes.add(new ArrayList<>());
    }
    int[] insertions = new int[numNodes];
    for (int i = 0; i < numNodes; i++) {
      insertions[i] = i;
    }
    // shuffle the insertion order
    for (int i = 0; i < 40; i++) {
      int pos1 = random().nextInt(numNodes);
      int pos2 = random().nextInt(numNodes);
      int tmp = insertions[pos1];
      insertions[pos1] = insertions[pos2];
      insertions[pos2] = tmp;
    }

    for (int i : insertions) {
      int level = random().nextInt(maxLevel);
      for (int l = level; l >= 0; l--) {
        graph.addNode(l, i);
        graph.trySetNewEntryNode(i, l);
        if (l > graph.numLevels() - 1) {
          graph.tryPromoteNewEntryNode(i, l, graph.numLevels() - 1);
        }
        levelToNodes.get(l).add(i);
      }
    }

    for (int i = 0; i < maxLevel; i++) {
      levelToNodes.get(i).sort(Integer::compare);
    }

    assertGraphEquals(graph, levelToNodes);
  }

  /* nodes may be promoted to upper levels after they were added to level 0 */
  public void testPromoteNodeToUpperLevels() {
    OnHeapHnswGraph graph = new OnHeapHnswGraph(10, 3);
    graph.addNode(0, 0);
    graph.addNode(0, 1);
    // node 2 is added from its top level down, it already counts as a node of level 0
    graph.addNode(1, 2);
    assertTrue(graph.nodeExistAtLevel(0, 2));
    assertTrue(graph.nodeExistAtLevel(1, 2));
    assertFalse(graph.nodeExistAtLevel(2, 2));
    graph.addNode(0, 2);
    graph.trySetNewEntryNode(2, 1);
    assertEquals(3, graph.size());

    assertFalse(graph.nodeExistAtLevel(1, 0));
    graph.addNode(1, 0);
    graph.addNode(2, 0);
    graph.tryPromoteNewEntryNode(0, 2, 1);
    assertEquals(3, graph.size());
    assertTrue(graph.nodeExistAtLevel(2, 0));
    assertFalse(graph.nodeExistAtLevel(1, 1));

    graph.getNeighbors(0, 0).addInOrder(1, 0.5f);
    graph.getNeighbors(1, 0).addInOrder(2, 0.5f);
    assertEquals(1, graph.getNeighbors(0, 0).size());
    assertEquals(1, graph.getNeighbors(1, 0).size());
    assertEquals(0, graph.getNeighbors(2, 0).size());
    assertNotSame(graph.getNeighbors(0, 0), graph.getNeighbors(1, 0));

    assertGraphEquals(graph, List.of(List.of(0, 1, 2), List.of(0, 2), List.of(0)));
  }

  public void testRamBytesUsed() {
    int maxConn = random().nextInt(2, 32);
    int numNodes = atLeast(500);
    boolean noGrowth = random().nextBoolean();
    OnHeapHnswGraph graph = new OnHeapHnswGraph(maxConn, noGrowth ? numNodes : -1);
    for (int node = 0; node < numNodes; node++) {
      int nodeLevel = 0;
      while (nodeLevel < 4 && random().nextInt(maxConn) == 0) {
        nodeLevel++;
      }
      for (int level = nodeLevel; level >= 0; level--) {
        graph.addNode(level, node);
        NeighborArray neighbors = graph.getNeighbors(level, node);
        int numNeighbors = random().nextInt(neighbors.maxSize());
        for (int i = 0; i < numNeighbors; i++) {
          neighbors.addOutOfOrder(random().nextInt(numNodes), random().nextFloat());
        }
      }
    }
    long actual = ramUsed(graph);
    // only a few small fixed-size objects (atomics, entry node) are not accounted for
    assertEquals(actual, graph.ramBytesUsed(), 256);
  }

  private static void assertGraphEquals(OnHeapHnswGraph graph, List<List<Integer>> levelToNodes) {
    for (int l = 0; l < graph.numLevels(); l++) {
      HnswGraph.NodesIterator nodesIterator = graph.getNodesOnLevel(l);
      assertEquals(levelToNodes.get(l).size(), nodesIterator.size());
      int idx = 0;
      while (nodesIterator.hasNext()) {
        assertEquals(levelToNodes.get(l).get(idx++), nodesIterator.next());
      }
    }
  }
}
