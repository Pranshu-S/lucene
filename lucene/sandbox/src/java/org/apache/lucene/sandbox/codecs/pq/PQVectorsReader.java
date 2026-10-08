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

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readSimilarityFunction;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader.readVectorEncoding;
import static org.apache.lucene.sandbox.codecs.pq.PQVectorsFormat.VECTOR_DATA_EXTENSION;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.Float16VectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Reads product-quantized vectors in the format described on {@link PQVectorsFormat}.
 *
 * @lucene.experimental
 */
public class PQVectorsReader extends FlatVectorsReader {

  private static final long SHALLOW_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(PQVectorsReader.class);

  private static final int EXHAUSTIVE_BULK_SCORE_ORDS = 64;

  private final Map<String, FieldEntry> fields;
  private final IndexInput vectorData;
  private final FlatVectorsReader rawVectorsReader;
  private final PQVectorScorer vectorScorer;

  /** Sole public constructor */
  public PQVectorsReader(
      SegmentReadState state, FlatVectorsReader rawVectorsReader, PQVectorScorer vectorScorer)
      throws IOException {
    this.fields = new HashMap<>();
    this.rawVectorsReader = rawVectorsReader;
    this.vectorScorer = vectorScorer;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, PQVectorsFormat.META_EXTENSION);
    try {
      int versionMeta = -1;
      try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaFileName)) {
        Throwable priorE = null;
        try {
          versionMeta =
              CodecUtil.checkIndexHeader(
                  meta,
                  PQVectorsFormat.META_CODEC_NAME,
                  PQVectorsFormat.VERSION_START,
                  PQVectorsFormat.VERSION_CURRENT,
                  state.segmentInfo.getId(),
                  state.segmentSuffix);
          readFields(meta, state.fieldInfos);
        } catch (Throwable exception) {
          priorE = exception;
        } finally {
          CodecUtil.checkFooter(meta, priorE);
        }
      }
      String dataFileName =
          IndexFileNames.segmentFileName(
              state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
      // how the codes are read is up to whoever wraps this format
      vectorData =
          state.directory.openInput(
              dataFileName, state.context.union(FileTypeHint.DATA, FileDataHint.KNN_VECTORS));
      int versionData =
          CodecUtil.checkIndexHeader(
              vectorData,
              PQVectorsFormat.VECTOR_DATA_CODEC_NAME,
              PQVectorsFormat.VERSION_START,
              PQVectorsFormat.VERSION_CURRENT,
              state.segmentInfo.getId(),
              state.segmentSuffix);
      if (versionMeta != versionData) {
        throw new CorruptIndexException(
            "Format versions mismatch: meta=" + versionMeta + ", data=" + versionData, vectorData);
      }
      CodecUtil.retrieveChecksum(vectorData);
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  /**
   * Copy constructor for {@link #getMergeInstance()}: shares {@code reader}'s open state and reads
   * full-precision vectors through {@code rawVectorsReader}. Never closed; {@link #finishMerge()}
   * releases the raw merge instance.
   */
  private PQVectorsReader(PQVectorsReader reader, FlatVectorsReader rawVectorsReader) {
    this.fields = reader.fields;
    this.vectorData = reader.vectorData;
    this.rawVectorsReader = rawVectorsReader;
    this.vectorScorer = reader.vectorScorer;
  }

  @Override
  public FlatVectorsReader getMergeInstance() throws IOException {
    return new PQVectorsReader(this, rawVectorsReader.getMergeInstance());
  }

  @Override
  public void finishMerge() throws IOException {
    rawVectorsReader.finishMerge();
  }

  private void readFields(ChecksumIndexInput meta, FieldInfos infos) throws IOException {
    for (int fieldNumber = meta.readInt(); fieldNumber != -1; fieldNumber = meta.readInt()) {
      FieldInfo info = infos.fieldInfo(fieldNumber);
      if (info == null) {
        throw new CorruptIndexException("Invalid field number: " + fieldNumber, meta);
      }
      fields.put(info.name, readField(meta, info));
    }
  }

  private static FieldEntry readField(IndexInput meta, FieldInfo info) throws IOException {
    VectorEncoding encoding = readVectorEncoding(meta);
    VectorSimilarityFunction similarity = readSimilarityFunction(meta);
    if (encoding != info.getVectorEncoding() || similarity != info.getVectorSimilarityFunction()) {
      throw new CorruptIndexException(
          "Inconsistent vector encoding or similarity for field=\""
              + info.name
              + "\"; "
              + encoding
              + "/"
              + similarity
              + " != "
              + info.getVectorEncoding()
              + "/"
              + info.getVectorSimilarityFunction(),
          meta);
    }
    int dimension = meta.readVInt();
    if (dimension != info.getVectorDimension()) {
      throw new CorruptIndexException(
          "Inconsistent vector dimension for field=\""
              + info.name
              + "\"; "
              + dimension
              + " != "
              + info.getVectorDimension(),
          meta);
    }
    int size = meta.readVInt();
    if (size == 0) {
      return new FieldEntry(size, 0, 0, null);
    }
    long codesOffset = meta.readVLong();
    long codesLength = meta.readVLong();
    ProductQuantizer quantizer = ProductQuantizer.read(meta);
    if (quantizer.dimension() != dimension) {
      throw new CorruptIndexException(
          "quantizer dimension " + quantizer.dimension() + " != field dimension " + dimension,
          meta);
    }
    if (codesLength != (long) size * quantizer.numSubspaces()) {
      throw new CorruptIndexException(
          "codes length "
              + codesLength
              + " != size="
              + size
              + " * numSubspaces="
              + quantizer.numSubspaces(),
          meta);
    }
    return new FieldEntry(size, codesOffset, codesLength, quantizer);
  }

  @Override
  public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
    return vectorScorer;
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
    FloatVectorValues values = getFloatVectorValues(field);
    if (values == null) {
      return null;
    }
    return vectorScorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
    return rawVectorsReader.getRandomVectorScorer(field, target);
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(String field, short[] target) throws IOException {
    return rawVectorsReader.getRandomVectorScorer(field, target);
  }

  @Override
  public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
    rawVectorsReader.checkIntegrity(merge);
    CodecUtil.checksumEntireFile(vectorData, merge);
  }

  @Override
  public FloatVectorValues getFloatVectorValues(String field) throws IOException {
    FieldEntry entry = fields.get(field);
    FloatVectorValues raw = rawVectorsReader.getFloatVectorValues(field);
    if (entry == null || entry.size == 0) {
      return raw;
    }
    if (raw.size() != entry.size) {
      throw new CorruptIndexException(
          "field=\""
              + field
              + "\" has "
              + entry.size
              + " codes but "
              + raw.size()
              + " full-precision vectors",
          vectorData);
    }
    IndexInput codes = vectorData.slice("pq-codes", entry.codesOffset, entry.codesLength);
    return new PQVectorValues(raw, entry.quantizer, codes);
  }

  @Override
  public ByteVectorValues getByteVectorValues(String field) throws IOException {
    return rawVectorsReader.getByteVectorValues(field);
  }

  @Override
  public Float16VectorValues getFloat16VectorValues(String field) throws IOException {
    return rawVectorsReader.getFloat16VectorValues(field);
  }

  @Override
  public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs)
      throws IOException {
    if (knnCollector.k() == 0) {
      return;
    }
    RandomVectorScorer scorer = getRandomVectorScorer(field, target);
    if (scorer == null) {
      return;
    }
    // no graph: score every accepted vector with ADC
    Bits acceptedOrds = scorer.getAcceptOrds(acceptDocs.bits());
    int[] ords = new int[EXHAUSTIVE_BULK_SCORE_ORDS];
    float[] scores = new float[EXHAUSTIVE_BULK_SCORE_ORDS];
    int numOrds = 0;
    for (int ord = 0; ord < scorer.maxOrd(); ord++) {
      if (acceptedOrds == null || acceptedOrds.get(ord)) {
        if (knnCollector.earlyTerminated()) {
          break;
        }
        ords[numOrds++] = ord;
        if (numOrds == ords.length) {
          collect(scorer, ords, scores, numOrds, knnCollector);
          numOrds = 0;
        }
      }
    }
    if (numOrds > 0) {
      collect(scorer, ords, scores, numOrds, knnCollector);
    }
  }

  private static void collect(
      RandomVectorScorer scorer, int[] ords, float[] scores, int numOrds, KnnCollector collector)
      throws IOException {
    collector.incVisitedCount(numOrds);
    if (scorer.bulkScore(ords, scores, numOrds) > collector.minCompetitiveSimilarity()) {
      for (int i = 0; i < numOrds; i++) {
        collector.collect(scorer.ordToDoc(ords[i]), scores[i]);
      }
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(vectorData, rawVectorsReader);
  }

  @Override
  public long ramBytesUsed() {
    long size = SHALLOW_SIZE;
    size +=
        RamUsageEstimator.sizeOfMap(
            fields, RamUsageEstimator.shallowSizeOfInstance(FieldEntry.class));
    for (FieldEntry entry : fields.values()) {
      if (entry.quantizer != null) {
        size += entry.quantizer.ramBytesUsed();
      }
    }
    size += rawVectorsReader.ramBytesUsed();
    return size;
  }

  @Override
  public Map<String, Long> getOffHeapByteSize(FieldInfo fieldInfo) {
    Objects.requireNonNull(fieldInfo);
    var raw = rawVectorsReader.getOffHeapByteSize(fieldInfo);
    FieldEntry entry = fields.get(fieldInfo.name);
    if (entry == null) {
      return raw;
    }
    return KnnVectorsReader.mergeOffHeapByteSizeMaps(
        raw, Map.of(VECTOR_DATA_EXTENSION, entry.codesLength));
  }

  @Override
  public int getVectorCount(FieldInfo fieldInfo) throws IOException {
    Objects.requireNonNull(fieldInfo);
    FieldEntry entry = fields.get(fieldInfo.name);
    if (entry == null) {
      return rawVectorsReader.getVectorCount(fieldInfo);
    }
    return entry.size;
  }

  /** Returns the quantizer of a field, or null if the field has no vectors in this segment. */
  public ProductQuantizer getQuantizer(String field) {
    FieldEntry entry = fields.get(field);
    return entry == null ? null : entry.quantizer;
  }

  private record FieldEntry(
      int size, long codesOffset, long codesLength, ProductQuantizer quantizer) {}
}
