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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.LongConsumer;
import org.apache.lucene.internal.hppc.IntArrayList;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.RamUsageEstimator;

/**
 * An {@link HnswGraph} where all nodes and connections are held in memory. This class is used to
 * construct the HNSW graph before it's written to the index.
 */
public final class OnHeapHnswGraph extends HnswGraph implements Accountable {

  // shallow estimate of the statically used on-heap memory.
  private static final long RAM_BYTES_USED =
      RamUsageEstimator.shallowSizeOfInstance(OnHeapHnswGraph.class);

  private static final int INIT_SIZE = 128;

  private final AtomicReference<EntryNode> entryNode;

  // the neighbours of every node on level 0, e.g. level0[1] is all the neighbours of node 1 at
  // level 0. Most nodes (about 1 - 1/M of them) only exist on level 0, so their neighbours are kept
  // in this flat array rather than in a per-node array of levels.
  private NeighborArray[] level0;
  // the neighbours on upper levels, where the first dimension is node id and second dimension is
  // level - 1, e.g. upperLevels[1][1] is all the neighbours of node 1 at level 2. This is null for
  // nodes that only exist on level 0.
  private NeighborArray[][] upperLevels;
  // essentially another 2d map in which the first dimension is level and second dimension is node
  // id, this is only generated on demand when there's someone calling getNodeOnLevel on a
  // non-zero level
  private IntArrayList[] levelToNodes;
  private int
      lastFreezeSize; // remember the size we are at last time to freeze the graph and generate
  // levelToNodes
  private final AtomicInteger size =
      new AtomicInteger(0); // graph size, which is number of nodes in level 0
  private final AtomicInteger maxNodeId = new AtomicInteger(-1);
  private final int nsize; // neighbour array size at non-zero level
  private final int nsize0; // neighbour array size at zero level
  private final boolean
      noGrowth; // if an initial size is passed in, we don't expect the graph to grow itself

  // KnnGraphValues iterator members
  private int upto;
  private NeighborArray cur;

  private final LongAdder graphRamBytesUsed = new LongAdder();
  // shared by all the NeighborArrays of this graph to report their memory usage
  private final LongConsumer neighborArrayRamUsageListener =
      l -> {
        assert l >= 0;
        graphRamBytesUsed.add(l);
      };

  /**
   * ctor
   *
   * @param numNodes number of nodes that will be added to this graph, passing in -1 means unbounded
   *     while passing in a non-negative value will lock the whole graph and disable the graph from
   *     growing itself (you cannot add a node with id >= numNodes)
   */
  OnHeapHnswGraph(int M, int numNodes) {
    this.entryNode = new AtomicReference<>(new EntryNode(-1, 1));
    // Neighbours' size on upper levels (nsize) and level 0 (nsize0)
    // We allocate extra space for neighbours, but then prune them to keep allowed maximum
    this.nsize = M + 1;
    this.nsize0 = (M * 2 + 1);
    noGrowth = numNodes != -1;
    if (noGrowth == false) {
      numNodes = INIT_SIZE;
    }
    this.level0 = new NeighborArray[numNodes];
    this.upperLevels = new NeighborArray[numNodes][];
    this.graphRamBytesUsed.add(
        RAM_BYTES_USED
            + RamUsageEstimator.shallowSizeOf(level0)
            + RamUsageEstimator.shallowSizeOf(upperLevels));
  }

  /**
   * Returns the {@link NeighborArray} connected to the given node.
   *
   * @param level level of the graph
   * @param node the node whose neighbors are returned, represented as an ordinal on the level 0.
   */
  public NeighborArray getNeighbors(int level, int node) {
    assert node < level0.length;
    assert nodeExistAtLevel(level, node)
        : "level="
            + level
            + ", node "
            + node
            + " has only "
            + numLevelsOfNode(node)
            + " levels for graph "
            + this;
    NeighborArray neighbors = level == 0 ? level0[node] : upperLevels[node][level - 1];
    assert neighbors != null : "node=" + node + ", level=" + level;
    return neighbors;
  }

  @Override
  public int size() {
    return size.get();
  }

  /**
   * When we initialize from another graph, the max node id is different from {@link #size()},
   * because we will add nodes out of order, such that we need two method for each
   *
   * @return max node id (inclusive)
   */
  @Override
  public int maxNodeId() {
    if (noGrowth) {
      // we know the eventual graph size and the graph can possibly
      // being concurrently modified
      return level0.length - 1;
    } else {
      // The graph cannot be concurrently modified (and searched) if
      // we don't know the size beforehand, so it's safe to return the
      // actual maxNodeId
      return maxNodeId.get();
    }
  }

  /**
   * Add node on the given level. Nodes can be inserted out of order, but it requires that the nodes
   * preceded by the node inserted out of order are eventually added.
   *
   * <p>NOTE: You must add a node starting from the node's top level
   *
   * @param level level to add a node on
   * @param node the node to add, represented as an ordinal on the level 0.
   */
  public void addNode(int level, int node) {

    if (node >= level0.length) {
      if (noGrowth) {
        throw new IllegalStateException(
            "The graph does not expect to grow when an initial size is given");
      }
      long oldBytes =
          RamUsageEstimator.shallowSizeOf(level0) + RamUsageEstimator.shallowSizeOf(upperLevels);
      level0 = ArrayUtil.grow(level0, node + 1);
      upperLevels = ArrayUtil.growExact(upperLevels, level0.length);
      graphRamBytesUsed.add(
          RamUsageEstimator.shallowSizeOf(level0)
              + RamUsageEstimator.shallowSizeOf(upperLevels)
              - oldBytes);
    }

    int nodeLevels = numLevelsOfNode(node);
    assert nodeLevels == 0 || nodeLevels >= level : "node must be inserted from the top level: ";
    if (nodeLevels == 0) {
      size.incrementAndGet();
    }
    NeighborArray neighbors =
        new NeighborArray(level == 0 ? nsize0 : nsize, true, neighborArrayRamUsageListener);
    if (level == 0) {
      level0[node] = neighbors;
    } else {
      NeighborArray[] levels = upperLevels[node];
      if (levels == null || levels.length < level) {
        long oldBytes = levels == null ? 0 : RamUsageEstimator.shallowSizeOf(levels);
        levels = levels == null ? new NeighborArray[level] : ArrayUtil.growExact(levels, level);
        graphRamBytesUsed.add(RamUsageEstimator.shallowSizeOf(levels) - oldBytes);
      }
      levels[level - 1] = neighbors;
      upperLevels[node] = levels;
    }
    maxNodeId.accumulateAndGet(node, Math::max);
  }

  @Override
  public void seek(int level, int targetNode) {
    cur = getNeighbors(level, targetNode);
    upto = -1;
  }

  @Override
  public int neighborCount() {
    return cur.size();
  }

  @Override
  public int nextNeighbor() {
    if (++upto < cur.size()) {
      return cur.nodes()[upto];
    }
    return NO_MORE_DOCS;
  }

  /**
   * Returns the current number of levels in the graph
   *
   * @return the current number of levels in the graph
   */
  @Override
  public int numLevels() {
    return entryNode.get().level + 1;
  }

  @Override
  public int maxConn() {
    return nsize - 1;
  }

  public boolean nodeExistAtLevel(int level, int node) {
    return numLevelsOfNode(node) > level;
  }

  /**
   * Returns the number of levels the given node has been added to, counting level 0 even if the
   * node was so far only added to upper levels (nodes are added from their top level down).
   */
  private int numLevelsOfNode(int node) {
    NeighborArray[] levels = upperLevels[node];
    if (levels != null) {
      return levels.length + 1;
    }
    return level0[node] == null ? 0 : 1;
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public int entryNode() {
    return entryNode.get().node;
  }

  /**
   * Try to set the entry node if the graph does not have one
   *
   * @return True if the entry node is set to the provided node. False if the entry node already
   *     exists
   */
  public boolean trySetNewEntryNode(int node, int level) {
    EntryNode current = entryNode.get();
    if (current.node == -1) {
      return entryNode.compareAndSet(current, new EntryNode(node, level));
    }
    return false;
  }

  /**
   * Try to promote the provided node to the entry node
   *
   * @param level should be larger than expectedOldLevel
   * @param expectOldLevel is the old entry node level the caller expect to be, the actual graph
   *     level can be different due to concurrent modification
   * @return True if the entry node is set to the provided node. False if expectOldLevel is not the
   *     same as the current entry node level. Even if the provided node's level is still higher
   *     than the current entry node level, the new entry node will not be set and false will be
   *     returned.
   */
  public boolean tryPromoteNewEntryNode(int node, int level, int expectOldLevel) {
    assert level > expectOldLevel;
    EntryNode currentEntry = entryNode.get();
    if (currentEntry.level == expectOldLevel) {
      return entryNode.compareAndSet(currentEntry, new EntryNode(node, level));
    }
    return false;
  }

  /**
   * WARN: calling this method will essentially iterate through all nodes at level 0 (even if you're
   * not getting node at level 0), we have built some caching mechanism such that if graph is not
   * changed only the first non-zero level call will pay the cost. So it is highly NOT recommended
   * to call this method while the graph is still building.
   *
   * <p>NOTE: calling this method while the graph is still building is prohibited
   */
  @Override
  public NodesIterator getNodesOnLevel(int level) {
    if (size() != maxNodeId() + 1) {
      throw new IllegalStateException(
          "graph build not complete, size=" + size() + " maxNodeId=" + maxNodeId());
    }
    if (level == 0) {
      return new DenseNodesIterator(size());
    } else {
      generateLevelToNodes();
      return new CollectionNodesIterator(levelToNodes[level]);
    }
  }

  private void generateLevelToNodes() {
    if (lastFreezeSize == size()) {
      return;
    }
    int maxLevels = numLevels();
    levelToNodes = new IntArrayList[maxLevels];
    for (int i = 1; i < maxLevels; i++) {
      levelToNodes[i] = new IntArrayList();
    }
    int nonNullNode = 0;
    for (int node = 0; node < level0.length; node++) {
      // when we init from another graph, we could have holes where some slot is null
      int nodeLevels = numLevelsOfNode(node);
      if (nodeLevels == 0) {
        continue;
      }
      nonNullNode++;
      for (int i = 1; i < nodeLevels; i++) {
        levelToNodes[i].add(node);
      }
      if (nonNullNode == size()) {
        break;
      }
    }
    lastFreezeSize = size();
  }

  /**
   * Provides an estimate of the current on-heap memory usage of the graph. This is not threadsafe,
   * meaning the heap utilization if building the graph concurrently may be inaccurate. The main
   * purpose of this method is during initial document indexing and flush.
   */
  @Override
  public long ramBytesUsed() {
    return graphRamBytesUsed.sum();
  }

  @Override
  public String toString() {
    return "OnHeapHnswGraph(size="
        + size()
        + ", numLevels="
        + numLevels()
        + ", entryNode="
        + entryNode()
        + ")";
  }

  private record EntryNode(int node, int level) {}
}
