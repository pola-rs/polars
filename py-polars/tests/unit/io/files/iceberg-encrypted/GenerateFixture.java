package org.apache.iceberg;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.deletes.BaseDVFileWriter;
import org.apache.iceberg.deletes.PositionDelete;
import org.apache.iceberg.deletes.PositionDeleteWriter;
import org.apache.iceberg.encryption.EncryptedOutputFile;
import org.apache.iceberg.encryption.KeyManagementClient;
import org.apache.iceberg.encryption.StandardEncryptionManager;
import org.apache.iceberg.io.DataWriter;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.io.PositionOutputStream;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.types.Types;

public class GenerateFixture {
  private static final Schema SCHEMA =
      new Schema(Types.NestedField.optional(1, "x", Types.LongType.get()));
  private static final PartitionSpec SPEC = PartitionSpec.unpartitioned();
  private static final StandardEncryptionManager ENCRYPTION =
      new StandardEncryptionManager(List.of(), "test-master", 16, new IdentityKms());

  private static class IdentityKms implements KeyManagementClient {
    public void initialize(Map<String, String> properties) {}
    public ByteBuffer wrapKey(ByteBuffer key, String id) { return key; }
    public ByteBuffer unwrapKey(ByteBuffer key, String id) { return key; }
  }

  private static OutputFile output(String name) {
    OutputFile delegate = Files.localOutput(name);
    return new OutputFile() {
      public PositionOutputStream create() { return delegate.create(); }
      public PositionOutputStream createOrOverwrite() { return delegate.createOrOverwrite(); }
      public String location() { return name; }
      public InputFile toInputFile() { return delegate.toInputFile(); }
    };
  }

  public static void main(String[] args) throws IOException {
    var encryptedData = ENCRYPTION.encrypt(output("data.parquet"));
    DataWriter<Record> data = Parquet.writeData((EncryptedOutputFile) encryptedData)
        .withKeyMetadata(encryptedData.keyMetadata()).schema(SCHEMA).withSpec(SPEC).createWriterFunc(GenericParquetWriter::create).build();
    try (data) {
      IntStream.range(0, 4).mapToObj(i -> {
        GenericRecord record = GenericRecord.create(SCHEMA);
        record.setField("x", (long) i);
        return record;
      }).forEach(data::write);
    }
    ManifestWriter<DataFile> manifest = ManifestFiles.write(
        3, SPEC, (EncryptedOutputFile) ENCRYPTION.encrypt(output("data.avro")), 2L);
    try (manifest) { manifest.add(data.toDataFile(), 1L); }

    var encryptedPositions = ENCRYPTION.encrypt(output("positions.parquet"));
    PositionDeleteWriter<Record> positions = Parquet.writeDeletes(
        (EncryptedOutputFile) encryptedPositions)
        .withKeyMetadata(encryptedPositions.keyMetadata()).withSpec(SPEC).createWriterFunc(GenericParquetWriter::create).buildPositionWriter();
    try (positions) {
      positions.write(PositionDelete.<Record>create().set("data.parquet", 1));
      positions.write(PositionDelete.<Record>create().set("data.parquet", 3));
    }
    writeMetadata("positions", manifest.toManifestFile(), List.of(positions.toDeleteFile()));

    var encryptedVector = ENCRYPTION.encrypt(output("deletes.puffin"));
    BaseDVFileWriter vectors = new BaseDVFileWriter(() -> encryptedVector, path -> null);
    try (vectors) {
      vectors.delete("data.parquet", 1, SPEC, null);
      vectors.delete("data.parquet", 3, SPEC, null);
    }
    long vectorLength = java.nio.file.Files.size(Path.of("deletes.puffin"));
    List<DeleteFile> deletes = vectors.result().deleteFiles().stream()
        .map(file -> FileMetadata.deleteFileBuilder(SPEC).copy(file)
            .withEncryptionKeyMetadata(encryptedVector.keyMetadata().copyWithLength(vectorLength))
            .withFileSizeInBytes(vectorLength).build())
        .toList();
    writeMetadata("vectors", manifest.toManifestFile(), deletes);
  }

  private static void writeMetadata(String name, ManifestFile data, List<DeleteFile> deletes)
      throws IOException {
    ManifestWriter<DeleteFile> manifest = ManifestFiles.writeDeleteManifest(
        3, SPEC, (EncryptedOutputFile) ENCRYPTION.encrypt(output(name + ".avro")), 2L);
    try (manifest) { deletes.forEach(file -> manifest.add(file, 2L)); }
    String listName = name + "-list.avro";
    ManifestListWriter list = ManifestLists.write(3, output(listName), ENCRYPTION, 2L, null, 2L, 0L);
    try (list) { list.addAll(List.of(data, manifest.toManifestFile())); }
    var keys = list.encryptionKeys();
    Snapshot snapshot = new BaseSnapshot(2L, 2L, null, System.currentTimeMillis(), "overwrite",
        Map.of("total-records", "4", "total-position-deletes", "2"),
        0, listName, 0L, 4L, keys.fileKey().keyId());
    TableMetadata metadata = TableMetadata.newTableMetadata(SCHEMA, SPEC, ".",
        Map.of("format-version", "3", "encryption.key-id", "test-master"));
    metadata = TableMetadata.buildFrom(metadata)
        .addEncryptionKey(keys.keyEncryptionKey()).addEncryptionKey(keys.fileKey())
        .setBranchSnapshot(snapshot, "main").build();
    TableMetadataParser.write(metadata, output(name + ".metadata.json"));
  }
}
