use std::io::Write;
use std::sync::Arc;

use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
use polars_parquet_format::{ColumnChunk, ColumnCryptoMetaData, ColumnMetaData, RowGroup};

use super::indexes::{
    write_column_index, write_encrypted_column_index, write_encrypted_offset_index,
    write_offset_index,
};
use super::page::PageWriteSpec;
use super::row_group::write_row_group;
use super::{RowGroupIterColumns, WriteOptions};
use crate::parquet::encryption::encrypt::{
    FileEncryptionProperties, FileEncryptor, column_metadata_aad, encrypted_thrift_object_to_vec,
    write_signed_plaintext_thrift_object,
};
use crate::parquet::encryption::modules::{ModuleType, create_footer_aad, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};
pub use crate::parquet::metadata::KeyValue;
use crate::parquet::metadata::{SchemaDescriptor, ThriftFileMetadata};
use crate::parquet::write::State;
use crate::parquet::{FOOTER_SIZE, PARQUET_ENCRYPTED_MAGIC, PARQUET_MAGIC};

pub(super) fn start_file<W: Write>(writer: &mut W, encrypted_footer: bool) -> ParquetResult<u64> {
    let magic = if encrypted_footer {
        &PARQUET_ENCRYPTED_MAGIC
    } else {
        &PARQUET_MAGIC
    };
    writer.write_all(magic)?;
    Ok(PARQUET_MAGIC.len() as u64)
}

pub(super) fn end_file<W: Write>(
    mut writer: &mut W,
    metadata: &ThriftFileMetadata,
    file_encryptor: Option<&Arc<FileEncryptor>>,
) -> ParquetResult<u64> {
    let (metadata_len, magic): (i32, &[u8; 4]) = match file_encryptor {
        None => {
            // Write metadata
            let mut protocol = TCompactOutputProtocol::new(&mut writer);
            let metadata_len = metadata.write_to_out_protocol(&mut protocol)? as i32;
            (metadata_len, &PARQUET_MAGIC)
        },
        Some(file_encryptor) if file_encryptor.properties().encrypt_footer() => {
            let mut crypto_metadata = vec![];
            {
                let mut protocol = TCompactOutputProtocol::new(&mut crypto_metadata);
                file_encryptor
                    .file_crypto_metadata()
                    .write_to_out_protocol(&mut protocol)?;
            }

            let mut plaintext_metadata = vec![];
            {
                let mut protocol = TCompactOutputProtocol::new(&mut plaintext_metadata);
                metadata.write_to_out_protocol(&mut protocol)?;
            }
            let aad = create_footer_aad(file_encryptor.file_aad())?;
            let mut footer_encryptor = file_encryptor.get_footer_encryptor()?;
            let encrypted_metadata = footer_encryptor.encrypt(&plaintext_metadata, &aad)?;

            writer.write_all(&crypto_metadata)?;
            writer.write_all(&encrypted_metadata)?;
            (
                (crypto_metadata.len() + encrypted_metadata.len()) as i32,
                &PARQUET_ENCRYPTED_MAGIC,
            )
        },
        Some(file_encryptor) => {
            let aad = create_footer_aad(file_encryptor.file_aad())?;
            let mut footer_encryptor = file_encryptor.get_footer_encryptor()?;
            let metadata_len = write_signed_plaintext_thrift_object(
                &mut writer,
                &mut *footer_encryptor,
                &aad,
                |protocol| metadata.write_to_out_protocol(protocol),
            )? as i32;
            (metadata_len, &PARQUET_MAGIC)
        },
    };

    // Write footer
    let metadata_bytes = metadata_len.to_le_bytes();
    let mut footer_buffer = [0u8; FOOTER_SIZE as usize];
    (0..4).for_each(|i| {
        footer_buffer[i] = metadata_bytes[i];
    });

    (&mut footer_buffer[4..]).write_all(magic)?;
    writer.write_all(&footer_buffer)?;
    writer.flush()?;
    Ok(metadata_len as u64 + FOOTER_SIZE)
}

fn create_column_orders(schema_desc: &SchemaDescriptor) -> Vec<polars_parquet_format::ColumnOrder> {
    // We only include ColumnOrder for leaf nodes.
    // Currently only supported ColumnOrder is TypeDefinedOrder so we set this
    // for all leaf nodes.
    // Even if the column has an undefined sort order, such as INTERVAL, this
    // is still technically the defined TYPEORDER so it should still be set.
    (0..schema_desc.columns().len())
        .map(|_| {
            polars_parquet_format::ColumnOrder::TYPEORDER(
                polars_parquet_format::TypeDefinedOrder {},
            )
        })
        .collect()
}

fn encrypt_row_groups(
    row_groups: Vec<RowGroup>,
    file_encryptor: &Arc<FileEncryptor>,
) -> ParquetResult<Vec<RowGroup>> {
    row_groups
        .into_iter()
        .enumerate()
        .map(|(row_group_index, mut row_group)| {
            let columns = row_group
                .columns
                .into_iter()
                .enumerate()
                .map(|(column_index, column_chunk)| {
                    encrypt_column_chunk_metadata(
                        column_chunk,
                        file_encryptor,
                        row_group_index,
                        column_index,
                    )
                })
                .collect::<ParquetResult<Vec<_>>>()?;
            row_group.columns = columns;
            Ok(row_group)
        })
        .collect()
}

fn encrypt_column_chunk_metadata(
    mut column_chunk: ColumnChunk,
    file_encryptor: &Arc<FileEncryptor>,
    row_group_index: usize,
    column_index: usize,
) -> ParquetResult<ColumnChunk> {
    let should_encrypt = match column_chunk.crypto_metadata.as_ref() {
        None => false,
        Some(ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(_)) => {
            !file_encryptor.properties().encrypt_footer()
        },
        Some(ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(_)) => true,
    };

    if !should_encrypt {
        return Ok(column_chunk);
    }

    let metadata = column_chunk
        .meta_data
        .take()
        .ok_or_else(|| ParquetError::oos("ColumnChunk.meta_data missing"))?;
    let encrypted_metadata = encrypt_column_metadata(
        &metadata,
        file_encryptor,
        &column_chunk,
        row_group_index,
        column_index,
    )?;
    column_chunk.encrypted_column_metadata = Some(encrypted_metadata);
    if !file_encryptor.properties().encrypt_footer() {
        let mut plaintext_metadata = metadata;
        plaintext_metadata.statistics = None;
        plaintext_metadata.encoding_stats = None;
        plaintext_metadata.size_statistics = None;
        column_chunk.meta_data = Some(plaintext_metadata);
    }
    Ok(column_chunk)
}

fn encrypt_column_metadata(
    metadata: &ColumnMetaData,
    file_encryptor: &Arc<FileEncryptor>,
    column_chunk: &ColumnChunk,
    row_group_index: usize,
    column_index: usize,
) -> ParquetResult<Vec<u8>> {
    let column_path = match column_chunk.crypto_metadata.as_ref() {
        Some(ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(column_key)) => {
            column_key.path_in_schema.join(".")
        },
        _ => metadata.path_in_schema.join("."),
    };
    let aad = column_metadata_aad(file_encryptor, row_group_index, column_index)?;
    let mut encryptor = file_encryptor.get_column_metadata_encryptor(&column_path)?;
    encrypted_thrift_object_to_vec(&mut *encryptor, &aad, |protocol| {
        metadata.write_to_out_protocol(protocol)
    })
}

fn write_column_index_with_encryption<W: Write>(
    writer: &mut W,
    pages: &[PageWriteSpec],
    file_encryptor: Option<&Arc<FileEncryptor>>,
    row_group_index: usize,
    column_index: usize,
    column_path: &str,
) -> ParquetResult<u64> {
    let Some(file_encryptor) =
        file_encryptor.filter(|encryptor| encryptor.is_column_encrypted(column_path))
    else {
        return write_column_index(writer, pages);
    };
    let mut encryptor = file_encryptor.get_column_metadata_encryptor(column_path)?;
    let aad = create_module_aad(
        file_encryptor.file_aad(),
        ModuleType::ColumnIndex,
        row_group_index,
        column_index,
        None,
    )?;
    write_encrypted_column_index(writer, pages, &mut *encryptor, &aad)
}

fn write_offset_index_with_encryption<W: Write>(
    writer: &mut W,
    pages: &[PageWriteSpec],
    file_encryptor: Option<&Arc<FileEncryptor>>,
    row_group_index: usize,
    column_index: usize,
    column_path: &str,
) -> ParquetResult<u64> {
    let Some(file_encryptor) =
        file_encryptor.filter(|encryptor| encryptor.is_column_encrypted(column_path))
    else {
        return write_offset_index(writer, pages);
    };
    let mut encryptor = file_encryptor.get_column_metadata_encryptor(column_path)?;
    let aad = create_module_aad(
        file_encryptor.file_aad(),
        ModuleType::OffsetIndex,
        row_group_index,
        column_index,
        None,
    )?;
    write_encrypted_offset_index(writer, pages, &mut *encryptor, &aad)
}

/// An interface to write a parquet file.
/// Use `start` to write the header, `write` to write a row group,
/// and `end` to write the footer.
pub struct FileWriter<W: Write> {
    writer: W,
    schema: SchemaDescriptor,
    options: WriteOptions,
    created_by: Option<String>,

    offset: u64,
    row_groups: Vec<RowGroup>,
    page_specs: Vec<Vec<Vec<PageWriteSpec>>>,
    file_encryptor: Option<Arc<FileEncryptor>>,
    /// Used to store the current state for writing the file
    state: State,
    // when the file is written, metadata becomes available
    metadata: Option<ThriftFileMetadata>,
}

/// Writes a parquet file containing only the header and footer
///
/// This is used to write the metadata as a separate Parquet file, usually when data
/// is partitioned across multiple files.
///
/// Note: Recall that when combining row groups from [`ThriftFileMetadata`], the `file_path` on each
/// of their column chunks must be updated with their path relative to where they are written to.
pub fn write_metadata_sidecar<W: Write>(
    writer: &mut W,
    metadata: &ThriftFileMetadata,
) -> ParquetResult<u64> {
    let mut len = start_file(writer, false)?;
    len += end_file(writer, metadata, None)?;
    Ok(len)
}

// Accessors
impl<W: Write> FileWriter<W> {
    /// The options assigned to the file
    pub fn options(&self) -> &WriteOptions {
        &self.options
    }

    /// The [`SchemaDescriptor`] assigned to this file
    pub fn schema(&self) -> &SchemaDescriptor {
        &self.schema
    }

    /// Returns the [`ThriftFileMetadata`]. This is Some iff the [`Self::end`] has been called.
    ///
    /// This is used to write the metadata as a separate Parquet file, usually when data
    /// is partitioned across multiple files
    pub fn metadata(&self) -> Option<&ThriftFileMetadata> {
        self.metadata.as_ref()
    }
}

impl<W: Write> FileWriter<W> {
    /// Returns a new [`FileWriter`].
    pub fn new(
        writer: W,
        schema: SchemaDescriptor,
        options: WriteOptions,
        created_by: Option<String>,
    ) -> Self {
        Self {
            writer,
            schema,
            options,
            created_by,
            offset: 0,
            row_groups: vec![],
            page_specs: vec![],
            file_encryptor: None,
            state: State::Initialised,
            metadata: None,
        }
    }

    pub fn new_with_encryption(
        writer: W,
        schema: SchemaDescriptor,
        options: WriteOptions,
        created_by: Option<String>,
        file_encryption_properties: Arc<FileEncryptionProperties>,
    ) -> ParquetResult<Self> {
        file_encryption_properties.validate_encrypted_column_names(&schema)?;
        Ok(Self {
            writer,
            schema,
            options,
            created_by,
            offset: 0,
            row_groups: vec![],
            page_specs: vec![],
            file_encryptor: Some(Arc::new(FileEncryptor::new(file_encryption_properties)?)),
            state: State::Initialised,
            metadata: None,
        })
    }

    /// Writes the header of the file.
    ///
    /// This is automatically called by [`Self::write`] if not called following [`Self::new`].
    ///
    /// # Errors
    /// Returns an error if data has been written to the file.
    fn start(&mut self) -> ParquetResult<()> {
        if self.offset == 0 {
            let encrypted_footer = self
                .file_encryptor
                .as_ref()
                .is_some_and(|file_encryptor| file_encryptor.properties().encrypt_footer());
            self.offset = start_file(&mut self.writer, encrypted_footer)?;
            self.state = State::Started;
            Ok(())
        } else {
            Err(ParquetError::InvalidParameter(
                "Start cannot be called twice".to_string(),
            ))
        }
    }

    /// Writes a row group to the file.
    ///
    /// This call is IO-bounded
    pub fn write<E>(
        &mut self,
        num_rows: u64,
        row_group: RowGroupIterColumns<'_, E>,
    ) -> ParquetResult<()>
    where
        ParquetError: From<E>,
        E: std::error::Error,
    {
        if self.offset == 0 {
            self.start()?;
        }
        let ordinal = self.row_groups.len();
        let (group, specs, size) = write_row_group(
            &mut self.writer,
            num_rows,
            self.offset,
            self.schema.columns(),
            row_group,
            ordinal,
            self.file_encryptor.as_ref(),
        )?;
        self.offset += size;
        self.row_groups.push(group);
        self.page_specs.push(specs);
        Ok(())
    }

    /// Writes the footer of the parquet file. Returns the total size of the file and the
    /// underlying writer.
    pub fn end(&mut self, key_value_metadata: Option<Vec<KeyValue>>) -> ParquetResult<u64> {
        if self.offset == 0 {
            self.start()?;
        }

        if self.state != State::Started {
            return Err(ParquetError::InvalidParameter(
                "End cannot be called twice".to_string(),
            ));
        }
        // compute file stats
        let num_rows = self.row_groups.iter().map(|group| group.num_rows).sum();
        let column_paths = self
            .schema
            .columns()
            .iter()
            .map(|column| column.path_in_schema.join("."))
            .collect::<Vec<_>>();

        if self.options.write_statistics {
            // write column indexes (require page statistics)
            for (row_group_index, (group, pages)) in self
                .row_groups
                .iter_mut()
                .zip(self.page_specs.iter())
                .enumerate()
            {
                for (column_index, (column, pages)) in
                    group.columns.iter_mut().zip(pages.iter()).enumerate()
                {
                    let offset = self.offset;
                    column.column_index_offset = Some(offset as i64);
                    self.offset += write_column_index_with_encryption(
                        &mut self.writer,
                        pages,
                        self.file_encryptor.as_ref(),
                        row_group_index,
                        column_index,
                        &column_paths[column_index],
                    )?;
                    column.column_index_length = Some((self.offset - offset) as i32);
                }
            }
        };

        // write offset index
        for (row_group_index, (group, pages)) in self
            .row_groups
            .iter_mut()
            .zip(self.page_specs.iter())
            .enumerate()
        {
            for (column_index, (column, pages)) in
                group.columns.iter_mut().zip(pages.iter()).enumerate()
            {
                let offset = self.offset;
                column.offset_index_offset = Some(offset as i64);
                self.offset += write_offset_index_with_encryption(
                    &mut self.writer,
                    pages,
                    self.file_encryptor.as_ref(),
                    row_group_index,
                    column_index,
                    &column_paths[column_index],
                )?;
                column.offset_index_length = Some((self.offset - offset) as i32);
            }
        }

        let row_groups = match self.file_encryptor.as_ref() {
            Some(file_encryptor) => encrypt_row_groups(self.row_groups.clone(), file_encryptor)?,
            None => self.row_groups.clone(),
        };
        let (encryption_algorithm, footer_signing_key_metadata) = match self.file_encryptor.as_ref()
        {
            Some(file_encryptor) if !file_encryptor.properties().encrypt_footer() => (
                Some(file_encryptor.encryption_algorithm()),
                file_encryptor.properties().footer_key_metadata().cloned(),
            ),
            _ => (None, None),
        };

        let metadata = ThriftFileMetadata::new(
            self.options.version.into(),
            self.schema.clone().into_thrift(),
            num_rows,
            row_groups,
            key_value_metadata,
            self.created_by.clone(),
            Some(create_column_orders(&self.schema)),
            encryption_algorithm,
            footer_signing_key_metadata,
        );

        let len = end_file(&mut self.writer, &metadata, self.file_encryptor.as_ref())?;
        self.state = State::Finished;
        self.metadata = Some(metadata);
        Ok(self.offset + len)
    }

    /// Returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Returns the underlying writer and [`ThriftFileMetadata`]
    /// # Panics
    /// This function panics if [`Self::end`] has not yet been called
    pub fn into_inner_and_metadata(self) -> (W, ThriftFileMetadata) {
        (self.writer, self.metadata.expect("File to have ended"))
    }
}
