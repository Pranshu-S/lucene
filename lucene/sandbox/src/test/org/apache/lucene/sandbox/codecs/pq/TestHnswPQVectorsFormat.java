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

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.VectorUtil;

public class TestHnswPQVectorsFormat extends LuceneTestCase {

  private static final String FIELD = "v";

  private static KnnVectorsFormat format() {
    // a small sample and few iterations keep the tests fast, and still exercise training
    return new HnswPQVectorsFormat(
        new PQVectorsFormat(2, 1000, 5), 16, 100, 1, null, random().nextInt(3) == 0 ? 0 : 100);
  }

  private static IndexWriterConfig config(KnnVectorsFormat format) {
    return newIndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format));
  }

  private static float[] randomVector(int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = random().nextFloat() * 2 - 1;
    }
    return v;
  }

  private static Document doc(int id, float[] vector) {
    Document doc = new Document();
    doc.add(new StringField("id", Integer.toString(id), Field.Store.YES));
    doc.add(new NumericDocValuesField("sort", random().nextInt(1000)));
    if (vector != null) {
      doc.add(new KnnFloatVectorField(FIELD, vector, EUCLIDEAN));
      doc.add(new StoredField("vec", floatsToBytes(vector)));
    }
    return doc;
  }

  private static byte[] floatsToBytes(float[] v) {
    ByteBuffer buffer = ByteBuffer.allocate(v.length * Float.BYTES);
    buffer.asFloatBuffer().put(v);
    return buffer.array();
  }

  private static float[] bytesToFloats(byte[] bytes) {
    float[] v = new float[bytes.length / Float.BYTES];
    ByteBuffer.wrap(bytes).asFloatBuffer().get(v);
    return v;
  }

  /**
   * Checks every segment: full-precision vectors match what each document indexed, and each code is
   * exactly what the segment's quantizer encodes that document's vector to, so codes and
   * full-precision vectors share their ordinals.
   */
  private static int assertSegmentsConsistent(DirectoryReader reader) throws IOException {
    int total = 0;
    for (LeafReaderContext ctx : reader.leaves()) {
      LeafReader leaf = ctx.reader();
      FloatVectorValues values = leaf.getFloatVectorValues(FIELD);
      if (values == null) {
        continue;
      }
      assertTrue(values.getClass().getName(), values instanceof PQVectorValues);
      PQVectorValues pqValues = (PQVectorValues) values;
      ProductQuantizer pq = pqValues.quantizer();
      byte[] code = new byte[pq.numSubspaces()];
      byte[] expected = new byte[pq.numSubspaces()];
      float[] scratch = new float[pq.numCentroids()];
      int leafCount = 0;
      KnnVectorValues.DocIndexIterator it = values.iterator();
      for (int doc = it.nextDoc(); doc != NO_MORE_DOCS; doc = it.nextDoc()) {
        float[] stored =
            bytesToFloats(leaf.storedFields().document(doc).getBinaryValue("vec").bytes);
        float[] vector = values.vectorValue(it.index());
        assertArrayEquals(stored, vector, 0f);
        pqValues.codeValue(it.index(), code);
        pq.encode(vector, expected, scratch);
        assertArrayEquals(expected, code);
        leafCount++;
      }
      assertEquals(values.size(), leafCount);
      total += leafCount;
    }
    return total;
  }

  public void testFlushMergeAndDeletes() throws Exception {
    int dim = 4 + random().nextInt(30);
    int numDocs = 300 + random().nextInt(700);
    try (Directory dir = newDirectory()) {
      Set<Integer> deleted = new HashSet<>();
      try (IndexWriter w = new IndexWriter(dir, config(format()).setMaxBufferedDocs(100))) {
        for (int i = 0; i < numDocs; i++) {
          // some documents have no vector, to exercise sparse fields
          w.addDocument(doc(i, random().nextInt(10) == 0 ? null : randomVector(dim)));
          if (i > 0 && random().nextInt(20) == 0) {
            int victim = random().nextInt(i);
            w.deleteDocuments(new Term("id", Integer.toString(victim)));
            deleted.add(victim);
          }
        }
        w.commit();
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          assertSegmentsConsistent(reader);
        }
        w.forceMerge(1);
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          LeafReader leaf = getOnlyLeafReader(reader);
          assertEquals(0, leaf.numDeletedDocs());
          int count = assertSegmentsConsistent(reader);
          assertEquals(count, leaf.getFloatVectorValues(FIELD).size());
          // deleted documents are gone from the merged codes
          for (int victim : deleted) {
            assertEquals(
                0,
                new IndexSearcher(reader)
                    .count(new TermQuery(new Term("id", Integer.toString(victim)))));
          }
          KnnVectorsReader vectorsReader = ((CodecReader) leaf).getVectorReader();
          Map<String, Long> offHeap =
              vectorsReader.getOffHeapByteSize(leaf.getFieldInfos().fieldInfo(FIELD));
          int m = ((PQVectorValues) leaf.getFloatVectorValues(FIELD)).quantizer().numSubspaces();
          assertEquals((long) count * m, (long) offHeap.get(PQVectorsFormat.VECTOR_DATA_EXTENSION));
          assertEquals((long) count * dim * Float.BYTES, (long) offHeap.get("vec"));
        }
      }
      // reopen from disk
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        assertSegmentsConsistent(reader);
      }
    }
  }

  public void testIndexSorting() throws Exception {
    int dim = 8;
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc =
          config(format())
              .setIndexSort(new Sort(new SortField("sort", SortField.Type.LONG)))
              .setMaxBufferedDocs(200)
              .setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < 1000; i++) {
          w.addDocument(doc(i, random().nextInt(5) == 0 ? null : randomVector(dim)));
        }
        w.commit();
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          assertTrue(reader.leaves().size() > 1);
          assertSegmentsConsistent(reader);
        }
        w.forceMerge(1);
        try (DirectoryReader reader = DirectoryReader.open(w)) {
          assertSegmentsConsistent(reader);
        }
      }
    }
  }

  /** Segments of at most 256 vectors are encoded exactly, so ADC scores are exact. */
  public void testSmallSegmentScoresExactly() throws Exception {
    int dim = 16;
    float[][] vectors = new float[50][];
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, config(format()))) {
      for (int i = 0; i < vectors.length; i++) {
        vectors[i] = randomVector(dim);
        w.addDocument(doc(i, vectors[i]));
      }
      w.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] query = randomVector(dim);
        TopDocs top = searcher.search(new KnnFloatVectorQuery(FIELD, query, 10), 10);
        assertEquals(10, top.scoreDocs.length);
        for (ScoreDoc sd : top.scoreDocs) {
          int id = Integer.parseInt(searcher.storedFields().document(sd.doc).get("id"));
          assertEquals(EUCLIDEAN.compare(query, vectors[id]), sd.score, 1e-5f);
        }
      }
    }
  }

  public void testRecall() throws Exception {
    int dim = 32;
    int numDocs = 3000;
    // clustered data, as real embeddings are
    float[][] centers = new float[30][];
    for (int i = 0; i < centers.length; i++) {
      centers[i] = randomVector(dim);
    }
    float[][] vectors = new float[numDocs][dim];
    for (int i = 0; i < numDocs; i++) {
      float[] c = centers[random().nextInt(centers.length)];
      for (int j = 0; j < dim; j++) {
        vectors[i][j] = c[j] + (float) random().nextGaussian() * 0.2f;
      }
    }
    try (Directory dir = newDirectory();
        IndexWriter w =
            new IndexWriter(
                dir,
                newIndexWriterConfig()
                    .setCodec(
                        TestUtil.alwaysKnnVectorsFormat(
                            new HnswPQVectorsFormat(
                                new PQVectorsFormat(2, 2000, 10), 16, 100, 1, null, 0)))
                    .setMaxBufferedDocs(1000))) {
      for (int i = 0; i < numDocs; i++) {
        w.addDocument(doc(i, vectors[i]));
      }
      w.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        int k = 10;
        int numQueries = 50;
        int hits = 0;
        for (int q = 0; q < numQueries; q++) {
          float[] query = vectors[random().nextInt(numDocs)].clone();
          for (int j = 0; j < dim; j++) {
            query[j] += (float) random().nextGaussian() * 0.1f;
          }
          Set<Integer> truth = bruteForce(vectors, query, k);
          TopDocs top = searcher.search(new KnnFloatVectorQuery(FIELD, query, k), k);
          for (ScoreDoc sd : top.scoreDocs) {
            if (truth.contains(
                Integer.parseInt(searcher.storedFields().document(sd.doc).get("id")))) {
              hits++;
            }
          }
        }
        double recall = hits / (double) (numQueries * k);
        assertTrue("recall=" + recall, recall > 0.7);
      }
    }
  }

  private static Set<Integer> bruteForce(float[][] vectors, float[] query, int k) {
    Integer[] ids = new Integer[vectors.length];
    float[] distances = new float[vectors.length];
    for (int i = 0; i < vectors.length; i++) {
      ids[i] = i;
      distances[i] = VectorUtil.squareDistance(query, vectors[i]);
    }
    Arrays.sort(ids, (a, b) -> Float.compare(distances[a], distances[b]));
    return new HashSet<>(Arrays.asList(ids).subList(0, k));
  }

  /** The flat format alone searches exhaustively with ADC. */
  public void testFlatFormat() throws Exception {
    int dim = 12;
    float[][] vectors = new float[400][];
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, config(new PQVectorsFormat(3, 1000, 5)))) {
      for (int i = 0; i < vectors.length; i++) {
        vectors[i] = randomVector(dim);
        w.addDocument(doc(i, vectors[i]));
      }
      w.forceMerge(1);
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        assertSegmentsConsistent(reader);
        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs top = searcher.search(new KnnFloatVectorQuery(FIELD, randomVector(dim), 5), 5);
        assertEquals(5, top.scoreDocs.length);
      }
    }
  }

  public void testUnsupportedFieldsAreRejected() throws Exception {
    for (VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
      if (similarity == EUCLIDEAN) {
        continue;
      }
      try (Directory dir = newDirectory();
          IndexWriter w = new IndexWriter(dir, config(format()))) {
        Document doc = new Document();
        doc.add(new KnnFloatVectorField(FIELD, new float[] {1, 2, 3}, similarity));
        IllegalArgumentException e =
            expectThrows(IllegalArgumentException.class, () -> w.addDocument(doc));
        assertTrue(e.getMessage(), e.getMessage().contains("EUCLIDEAN"));
      }
    }
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, config(format()))) {
      Document doc = new Document();
      doc.add(new KnnByteVectorField(FIELD, new byte[] {1, 2, 3}, EUCLIDEAN));
      expectThrows(IllegalArgumentException.class, () -> w.addDocument(doc));
    }
  }

  public void testFormatIsLoadedBySPI() {
    assertEquals(
        HnswPQVectorsFormat.class, KnnVectorsFormat.forName(HnswPQVectorsFormat.NAME).getClass());
    assertEquals(PQVectorsFormat.class, KnnVectorsFormat.forName(PQVectorsFormat.NAME).getClass());
  }
}
