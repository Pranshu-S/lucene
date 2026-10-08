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

import static org.apache.lucene.sandbox.codecs.pq.PQVectorsFormat.PQ_COMPONENT;
import static org.apache.lucene.sandbox.codecs.pq.PQVectorsFormat.TRAINING_SEED;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.RamUsageEstimator.shallowSizeOfInstance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.InfoStream;

/**
 * Writes product-quantized vectors in the format described on {@link PQVectorsFormat}.
 *
 * @lucene.experimental
 */
public class PQVectorsWriter extends FlatVectorsWriter {
  private static final long SHALLOW_RAM_BYTES_USED = shallowSizeOfInstance(PQVectorsWriter.class);

  private final SegmentWriteState segmentWriteState;
  private final PQVectorsFormat format;
  private final List<FieldWriter> fields = new ArrayList<>();
  private final IndexOutput meta, vectorData;
  private final FlatVectorsWriter rawVectorDelegate;
  private boolean finished;

  /** Sole constructor */
  public PQVectorsWriter(
      SegmentWriteState state,
      PQVectorsFormat format,
      FlatVectorsWriter rawVectorDelegate,
      PQVectorScorer vectorScorer)
      throws IOException {
    super(vectorScorer);
    this.segmentWriteState = state;
    this.format = format;
    this.rawVectorDelegate = rawVectorDelegate;
    String metaFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, PQVectorsFormat.META_EXTENSION);
    String vectorDataFileName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, PQVectorsFormat.VECTOR_DATA_EXTENSION);
    try {
      meta = state.directory.createOutput(metaFileName, state.context);
      vectorData =
          state.directory.createOutput(
              vectorDataFileName, state.context.union(FileTypeHint.DATA, FileDataHint.KNN_VECTORS));
      CodecUtil.writeIndexHeader(
          meta,
          PQVectorsFormat.META_CODEC_NAME,
          PQVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      CodecUtil.writeIndexHeader(
          vectorData,
          PQVectorsFormat.VECTOR_DATA_CODEC_NAME,
          PQVectorsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
    } catch (Throwable t) {
      IOUtils.closeWhileSuppressingExceptions(t, this);
      throw t;
    }
  }

  /** Throws if this format cannot quantize the field. */
  static void checkSupported(FieldInfo fieldInfo) {
    if (fieldInfo.getVectorEncoding() != VectorEncoding.FLOAT32
        || fieldInfo.getVectorSimilarityFunction() != VectorSimilarityFunction.EUCLIDEAN) {
      throw new IllegalArgumentException(
          PQVectorsFormat.NAME
              + " only supports FLOAT32 vectors with EUCLIDEAN similarity; field=\""
              + fieldInfo.name
              + "\" has encoding="
              + fieldInfo.getVectorEncoding()
              + " and similarity="
              + fieldInfo.getVectorSimilarityFunction());
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public FlatFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
    checkSupported(fieldInfo);
    FieldWriter fieldWriter =
        new FieldWriter(
            fieldInfo, (FlatFieldVectorsWriter<float[]>) rawVectorDelegate.addField(fieldInfo));
    fields.add(fieldWriter);
    return fieldWriter;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    rawVectorDelegate.flush(maxDoc, sortMap);
    for (FieldWriter field : fields) {
      List<float[]> vectors = field.getVectors();
      if (sortMap != null) {
        // write codes in the order the full-precision delegate wrote the vectors
        int[] newToOldOrd = new int[vectors.size()];
        mapOldOrdToNewOrd(field.getDocsWithFieldSet(), sortMap, null, newToOldOrd, null);
        List<float[]> sorted = new ArrayList<>(vectors.size());
        for (int oldOrd : newToOldOrd) {
          sorted.add(vectors.get(oldOrd));
        }
        vectors = sorted;
      }
      FloatVectorValues values =
          FloatVectorValues.fromFloats(vectors, field.fieldInfo.getVectorDimension());
      writeField(field.fieldInfo, () -> values);
      field.finish();
    }
  }

  @Override
  public void mergeOneFlatVectorField(FieldInfo fieldInfo, MergeState mergeState)
      throws IOException {
    rawVectorDelegate.mergeOneFlatVectorField(fieldInfo, mergeState);
    checkSupported(fieldInfo);
    // Codebooks are retrained on the merged vectors, which are re-encoded from full precision.
    writeField(fieldInfo, () -> MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState));
  }

  /**
   * Trains a quantizer for the field and writes its metadata and codes. {@code values} must yield
   * the vectors in the order the full-precision delegate wrote them, each time it is called: once
   * to sample, once to encode, both in a single forward pass.
   */
  private void writeField(FieldInfo fieldInfo, IOSupplier<FloatVectorValues> values)
      throws IOException {
    InfoStream infoStream = segmentWriteState.infoStream;
    int dimension = fieldInfo.getVectorDimension();
    long startNS = System.nanoTime();
    ProductQuantizer.Sample sample =
        ProductQuantizer.reservoirSample(values.get(), format.sampleSize(), TRAINING_SEED);
    int count = sample.numVectors();

    meta.writeInt(fieldInfo.number);
    meta.writeInt(fieldInfo.getVectorEncoding().ordinal());
    meta.writeInt(fieldInfo.getVectorSimilarityFunction().ordinal());
    meta.writeVInt(dimension);
    meta.writeVInt(count);
    if (count == 0) {
      return;
    }

    int numSubspaces = Math.min(dimension, format.numSubspaces(dimension));
    ProductQuantizer quantizer =
        ProductQuantizer.train(
            sample.vectors(), dimension, numSubspaces, format.trainingIterations(), TRAINING_SEED);
    long trainedNS = System.nanoTime();

    long codesOffset = vectorData.getFilePointer();
    byte[] code = new byte[numSubspaces];
    float[] scratch = new float[quantizer.numCentroids()];
    FloatVectorValues vectorValues = values.get();
    KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
    int encoded = 0;
    for (int doc = iterator.nextDoc(); doc != NO_MORE_DOCS; doc = iterator.nextDoc()) {
      quantizer.encode(vectorValues.vectorValue(iterator.index()), code, scratch);
      vectorData.writeBytes(code, numSubspaces);
      encoded++;
    }
    if (encoded != count) {
      throw new IllegalStateException(
          "field=\"" + fieldInfo.name + "\" yielded " + encoded + " vectors, expected " + count);
    }
    long codesLength = vectorData.getFilePointer() - codesOffset;

    meta.writeVLong(codesOffset);
    meta.writeVLong(codesLength);
    quantizer.write(meta);

    if (infoStream.isEnabled(PQ_COMPONENT)) {
      long doneNS = System.nanoTime();
      infoStream.message(
          PQ_COMPONENT,
          "field=\""
              + fieldInfo.name
              + "\" vectors="
              + count
              + " sample="
              + sample.vectors().length
              + " "
              + quantizer
              + " sampleAndTrainMs="
              + (trainedNS - startNS) / 1_000_000
              + " encodeMs="
              + (doneNS - trainedNS) / 1_000_000);
    }
  }

  @Override
  public void finish() throws IOException {
    if (finished) {
      throw new IllegalStateException("already finished");
    }
    finished = true;
    rawVectorDelegate.finish();
    if (meta != null) {
      // write end of fields marker
      meta.writeInt(-1);
      CodecUtil.writeFooter(meta);
    }
    if (vectorData != null) {
      CodecUtil.writeFooter(vectorData);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(meta, vectorData, rawVectorDelegate);
  }

  @Override
  public long ramBytesUsed() {
    // the delegate accounts for the buffered vectors of every field
    return SHALLOW_RAM_BYTES_USED + rawVectorDelegate.ramBytesUsed();
  }

  /** Buffers a field's vectors in the full-precision delegate until flush. */
  private static final class FieldWriter extends FlatFieldVectorsWriter<float[]> {
    private final FieldInfo fieldInfo;
    private final FlatFieldVectorsWriter<float[]> delegate;
    private boolean finished;

    FieldWriter(FieldInfo fieldInfo, FlatFieldVectorsWriter<float[]> delegate) {
      this.fieldInfo = fieldInfo;
      this.delegate = delegate;
    }

    @Override
    public void addValue(int docID, float[] vectorValue) throws IOException {
      delegate.addValue(docID, vectorValue);
    }

    @Override
    public float[] copyValue(float[] vectorValue) {
      throw new UnsupportedOperationException();
    }

    @Override
    public List<float[]> getVectors() {
      return delegate.getVectors();
    }

    @Override
    public DocsWithFieldSet getDocsWithFieldSet() {
      return delegate.getDocsWithFieldSet();
    }

    @Override
    public void finish() throws IOException {
      if (finished) {
        return;
      }
      assert delegate.isFinished();
      finished = true;
    }

    @Override
    public boolean isFinished() {
      return finished && delegate.isFinished();
    }

    @Override
    public long ramBytesUsed() {
      return delegate.ramBytesUsed();
    }
  }
}
