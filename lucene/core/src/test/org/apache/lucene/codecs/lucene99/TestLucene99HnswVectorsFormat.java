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
package org.apache.lucene.codecs.lucene99;

import static java.lang.String.format;
import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.oneOf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SameThreadExecutorService;

public class TestLucene99HnswVectorsFormat extends BaseKnnVectorsFormatTestCase {
  @Override
  protected Codec getCodec() {
    return TestUtil.getDefaultCodec();
  }

  public void testToString() {
    FilterCodec customCodec =
        new FilterCodec("foo", Codec.getDefault()) {
          @Override
          public KnnVectorsFormat knnVectorsFormat() {
            return new Lucene99HnswVectorsFormat(10, 20);
          }
        };
    String expectedPattern =
        "Lucene99HnswVectorsFormat(name=Lucene99HnswVectorsFormat, maxConn=10, beamWidth=20, tinySegmentsThreshold=100, flatVectorFormat=Lucene99FlatVectorsFormat(vectorsScorer=%s()))";
    var defaultScorer = format(Locale.ROOT, expectedPattern, "DefaultFlatVectorScorer");
    var memSegScorer =
        format(Locale.ROOT, expectedPattern, "Lucene99MemorySegmentFlatVectorsScorer");
    assertThat(customCodec.knnVectorsFormat().toString(), is(oneOf(defaultScorer, memSegScorer)));
  }

  public void testLimits() {
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(-1, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(0, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, 0));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, -1));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(512 + 1, 20));
    expectThrows(IllegalArgumentException.class, () -> new Lucene99HnswVectorsFormat(20, 3201));
    expectThrows(
        IllegalArgumentException.class,
        () -> new Lucene99HnswVectorsFormat(20, 100, 1, new SameThreadExecutorService()));
  }

  public void testSimpleOffHeapSize() throws IOException {
    float[] vector = randomVector(random().nextInt(12, 500));
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {
      Document doc = new Document();
      doc.add(new KnnFloatVectorField("f", vector, DOT_PRODUCT));
      w.addDocument(doc);
      w.commit();
      try (IndexReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        if (r instanceof CodecReader codecReader) {
          KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
          knnVectorsReader = knnVectorsReader.unwrapReaderForField("f");
          var fieldInfo = r.getFieldInfos().fieldInfo("f");
          var offHeap = knnVectorsReader.getOffHeapByteSize(fieldInfo);
          assertEquals(vector.length * Float.BYTES, (long) offHeap.get("vec"));
          assertNotNull(offHeap.get("vex"));
          assertEquals(2, offHeap.size());
        }
      }
    }
  }

  /**
   * A selective filter takes the exhaustive (non-HNSW) path, which must score exactly the accepted
   * docs that have a vector, on both dense and sparse fields.
   */
  public void testExhaustiveSearchWithSelectiveFilter() throws IOException {
    boolean sparse = random().nextBoolean();
    int numDocs = 2000;
    int dim = 4;
    VectorSimilarityFunction similarity = VectorSimilarityFunction.EUCLIDEAN;
    Codec codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99HnswVectorsFormat());
    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig().setCodec(codec))) {
      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        if (sparse == false || i % 3 == 0) {
          doc.add(new KnnFloatVectorField("f", randomVector(dim), similarity));
        }
        w.addDocument(doc);
      }
      w.forceMerge(1);
      try (IndexReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FixedBitSet filter = new FixedBitSet(r.maxDoc());
        for (int i = 0; i < 30; i++) {
          filter.set(random().nextInt(r.maxDoc()));
        }
        float[] target = randomVector(dim);
        int k = 10;

        List<ScoreDoc> expected = new ArrayList<>();
        FloatVectorValues values = r.getFloatVectorValues("f");
        KnnVectorValues.DocIndexIterator it = values.iterator();
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
          if (filter.get(doc)) {
            expected.add(
                new ScoreDoc(doc, similarity.compare(target, values.vectorValue(it.index()))));
          }
        }
        expected.sort(Comparator.comparingDouble((ScoreDoc sd) -> -sd.score));

        TopDocs actual =
            r.searchNearestVectors(
                "f", target, k, AcceptDocs.fromLiveDocs(filter, r.maxDoc()), Integer.MAX_VALUE);
        assertEquals(expected.size(), actual.totalHits.value());
        assertEquals(Math.min(k, expected.size()), actual.scoreDocs.length);
        for (int i = 0; i < actual.scoreDocs.length; i++) {
          assertEquals(expected.get(i).doc, actual.scoreDocs[i].doc);
          assertEquals(expected.get(i).score, actual.scoreDocs[i].score, 1e-5f);
        }
      }
    }
  }

  @Override
  protected boolean supportsFloatVectorFallback() {
    return false;
  }
}
