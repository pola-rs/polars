use std::io::Write;
use std::sync::Arc;

use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
use polars_parquet_format::{ColumnCryptoMetaData, RowGroup};

use super::indexes::{serialize_column_index, serialize_offset_index};
use super::page::PageWriteSpec;
use super::row_group::write_row_group;
use super::{RowGroupIterColumns, WriteOptions, WriteThrift};
use crate::parquet::encryption::encrypt::{FileEncryptionProperties, FileEncryptor};
use crate::parquet::encryption::modules::ModuleType;
use crate::parquet::error::{ParquetError, ParquetResult};
pub use crate::parquet::metadata::KeyValue;
use crate::parquet::metadata::{ColumnDescriptor, SchemaDescriptor, ThriftFileMetadata};
use crate::parquet::write::State;
use crate::parquet::{ENCRYPTED_PARQUET_MAGIC, FOOTER_SIZE, PARQUET_MAGIC};

pub(super) fn start_file<W: Write>(writer: &mut W, magic: &[u8; 4]) -> ParquetResult<u64> {
    writer.write_all(magic)?;
    Ok(magic.len() as u64)
}

fn write_footer_trailer<W: Write>(
    writer: &mut W,
    magic: &[u8; 4],
    metadata_len: usize,
) -> ParquetResult<u64> {
    let metadata_len: i32 = metadata_len.try_into().map_err(|_| {
        ParquetError::oos(format!(
            "The footer can only contain up to i32::MAX bytes. This one contains {}",
            metadata_len
        ))
    })?;

    let mut footer_buffer = [0u8; FOOTER_SIZE as usize];
    footer_buffer[..4].copy_from_slice(&metadata_len.to_le_bytes());
    footer_buffer[4..].copy_from_slice(magic);

    writer.write_all(&footer_buffer)?;
    writer.flush()?;

    Ok(metadata_len as u64 + FOOTER_SIZE)
}

/// Writes the footer of a Parquet file
pub(super) fn end_file<W: Write>(
    mut writer: &mut W,
    metadata: &ThriftFileMetadata,
) -> ParquetResult<u64> {
    // Write metadata
    let mut protocol = TCompactOutputProtocol::new(&mut writer);
    let metadata_len = metadata.write_to_out_protocol(&mut protocol)?;

    write_footer_trailer(&mut writer, &PARQUET_MAGIC, metadata_len)
}

/// Writes the footer of an encrypted Parquet file.
fn end_encrypted_file<W: Write>(
    writer: &mut W,
    metadata: &ThriftFileMetadata,
    encryptor: &mut FileEncryptor,
    schema: &SchemaDescriptor,
) -> ParquetResult<u64> {
    let mut metadata = metadata.clone();
    encrypt_column_metadata(&mut metadata.row_groups, encryptor, schema)?;

    let (footer, magic) = if encryptor.properties().encrypt_footer() {
        // When the footer is encrypted, write the file crypto metadata
        // followed by the encrypted file metadata.
        let mut footer = encryptor.file_crypto_metadata().to_thrift_bytes()?;
        footer.extend(encryptor.encrypt_footer(&metadata.to_thrift_bytes()?)?);
        (footer, ENCRYPTED_PARQUET_MAGIC)
    } else {
        // When the footer is in plaintext, set the encryption related fields
        // in the file metadata before writing it, then write the footer signature.
        metadata.encryption_algorithm = Some(encryptor.encryption_algorithm());
        metadata.footer_signing_key_metadata =
            encryptor.properties().footer_key_metadata().cloned();
        let mut footer = metadata.to_thrift_bytes()?;
        let signature = encryptor.sign_footer(&footer)?;
        footer.extend(signature);
        (footer, PARQUET_MAGIC)
    };

    writer.write_all(&footer)?;
    write_footer_trailer(writer, &magic, footer.len())
}

/// Encrypts the metadata of column chunks that are encrypted to a different key to the footer,
/// or where the footer is stored in plaintext.
/// When the footer is in plaintext, column chunk metadata with redacted statistics is kept.
fn encrypt_column_metadata(
    row_groups: &mut [RowGroup],
    encryptor: &mut FileEncryptor,
    schema: &SchemaDescriptor,
) -> ParquetResult<()> {
    let encrypt_footer = encryptor.properties().encrypt_footer();
    for (row_group_idx, row_group) in row_groups.iter_mut().enumerate() {
        let columns = row_group.columns.iter_mut().zip(schema.columns());
        for (column_ordinal, (column_chunk, descriptor)) in columns.enumerate() {
            let encrypted_with_footer_key = match &column_chunk.crypto_metadata {
                None => continue,
                Some(ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(_)) => true,
                Some(ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(_)) => false,
            };
            if encrypt_footer && encrypted_with_footer_key {
                // No further encryption is needed, the column chunk is already encrypted
                // with the footer key.
                continue;
            }
            let Some(mut metadata) = column_chunk.meta_data.take() else {
                continue;
            };
            column_chunk.encrypted_column_metadata = Some(encryptor.encrypt_column_module(
                descriptor,
                ModuleType::ColumnMetaData,
                row_group_idx,
                column_ordinal,
                &metadata.to_thrift_bytes()?,
            )?);
            if !encrypt_footer {
                // Keep the metadata in the plaintext footer for readers without the key,
                // but without statistics, which must be treated as sensitive data.
                metadata.statistics = None;
                metadata.encoding_stats = None;
                metadata.bloom_filter_offset = None;
                metadata.bloom_filter_length = None;
                metadata.size_statistics = None;
                column_chunk.meta_data = Some(metadata);
            }
        }
    }
    Ok(())
}

/// Write the column index or offset index of a column chunk, encrypting it if the column
/// is encrypted. Returns the number of bytes written.
fn write_index<W: Write>(
    writer: &mut W,
    encryptor: Option<&mut FileEncryptor>,
    index: &impl WriteThrift,
    descriptor: &ColumnDescriptor,
    module_type: ModuleType,
    row_group_idx: usize,
    column_ordinal: usize,
) -> ParquetResult<u64> {
    match column_encryptor(encryptor, descriptor) {
        None => {
            let mut protocol = TCompactOutputProtocol::new(writer);
            Ok(index.write_thrift(&mut protocol)? as u64)
        },
        Some(encryptor) => {
            let encrypted = encryptor.encrypt_column_module(
                descriptor,
                module_type,
                row_group_idx,
                column_ordinal,
                &index.to_thrift_bytes()?,
            )?;
            writer.write_all(&encrypted)?;
            Ok(encrypted.len() as u64)
        },
    }
}

/// Returns the file encryptor to use for a column, or `None` if the column isn't encrypted.
fn column_encryptor<'a>(
    encryptor: Option<&'a mut FileEncryptor>,
    column: &ColumnDescriptor,
) -> Option<&'a mut FileEncryptor> {
    encryptor.filter(|encryptor| encryptor.is_column_descriptor_encrypted(column))
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
    /// Used to store the current state for writing the file
    state: State,
    // when the file is written, metadata becomes available, unless the file is encrypted
    metadata: Option<ThriftFileMetadata>,
    encryptor: Option<FileEncryptor>,
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
    let mut len = start_file(writer, &PARQUET_MAGIC)?;
    len += end_file(writer, metadata)?;
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

    /// Returns the [`ThriftFileMetadata`]. This is Some iff the [`Self::end`] has been called
    /// and the file is not encrypted. Encrypted files return None, as their plaintext metadata
    /// would expose information about encrypted columns, such as statistics.
    ///
    /// This is used to write the metadata as a separate Parquet file, usually when data
    /// is partitioned across multiple files.
    pub fn metadata(&self) -> Option<&ThriftFileMetadata> {
        self.metadata.as_ref()
    }
}

impl<W: Write> FileWriter<W> {
    /// Returns a new [`FileWriter`].
    ///
    /// # Errors
    /// Returns an error if the encryption properties are set and not valid for the file schema.
    pub fn try_new(
        writer: W,
        schema: SchemaDescriptor,
        options: WriteOptions,
        created_by: Option<String>,
        encryption_properties: Option<Arc<FileEncryptionProperties>>,
    ) -> ParquetResult<Self> {
        let encryptor = encryption_properties
            .map(|properties| FileEncryptor::try_new_for_schema(properties, &schema))
            .transpose()?;
        Ok(Self {
            writer,
            schema,
            options,
            created_by,
            offset: 0,
            row_groups: vec![],
            page_specs: vec![],
            state: State::Initialised,
            metadata: None,
            encryptor,
        })
    }

    /// Writes the header of the file.
    ///
    /// This is automatically called by [`Self::write`] if not called following [`Self::try_new`].
    ///
    /// # Errors
    /// Returns an error if data has been written to the file.
    fn start(&mut self) -> ParquetResult<()> {
        if self.offset == 0 {
            let magic = match &self.encryptor {
                Some(encryptor) if encryptor.properties().encrypt_footer() => {
                    &ENCRYPTED_PARQUET_MAGIC
                },
                _ => &PARQUET_MAGIC,
            };
            self.offset = start_file(&mut self.writer, magic)?;
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
            self.encryptor.as_mut(),
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

        let mut encryptor = self.encryptor.as_mut();
        let column_descriptors = self.schema.columns();

        if self.options.write_statistics {
            // write column indexes (require page statistics)
            for (row_group_idx, (group, pages)) in
                self.row_groups.iter_mut().zip(&self.page_specs).enumerate()
            {
                let columns = group.columns.iter_mut().zip(pages).zip(column_descriptors);
                for (column_ordinal, ((column, pages), descriptor)) in columns.enumerate() {
                    // Pages without usable statistics have no column index.
                    let Some(index) = serialize_column_index(pages)? else {
                        continue;
                    };
                    let offset = self.offset;
                    self.offset += write_index(
                        &mut self.writer,
                        encryptor.as_deref_mut(),
                        &index,
                        descriptor,
                        ModuleType::ColumnIndex,
                        row_group_idx,
                        column_ordinal,
                    )?;
                    column.column_index_offset = Some(offset as i64);
                    column.column_index_length = Some((self.offset - offset) as i32);
                }
            }
        };

        // write offset index
        for (row_group_idx, (group, pages)) in
            self.row_groups.iter_mut().zip(&self.page_specs).enumerate()
        {
            let columns = group.columns.iter_mut().zip(pages).zip(column_descriptors);
            for (column_ordinal, ((column, pages), descriptor)) in columns.enumerate() {
                let offset = self.offset;
                self.offset += write_index(
                    &mut self.writer,
                    encryptor.as_deref_mut(),
                    &serialize_offset_index(pages)?,
                    descriptor,
                    ModuleType::OffsetIndex,
                    row_group_idx,
                    column_ordinal,
                )?;
                column.offset_index_offset = Some(offset as i64);
                column.offset_index_length = Some((self.offset - offset) as i32);
            }
        }

        let metadata = ThriftFileMetadata::new(
            self.options.version.into(),
            self.schema.clone().into_thrift(),
            num_rows,
            self.row_groups.clone(),
            key_value_metadata,
            self.created_by.clone(),
            Some(create_column_orders(&self.schema)),
            None,
            None,
        );

        let len = match &mut self.encryptor {
            None => end_file(&mut self.writer, &metadata)?,
            Some(encryptor) => {
                end_encrypted_file(&mut self.writer, &metadata, encryptor, &self.schema)?
            },
        };
        self.state = State::Finished;
        if self.encryptor.is_none() {
            self.metadata = Some(metadata);
        }
        Ok(self.offset + len)
    }

    /// Returns the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Returns the underlying writer and [`ThriftFileMetadata`]
    /// # Panics
    /// This function panics if [`Self::end`] has not yet been called, or if the file is encrypted
    pub fn into_inner_and_metadata(self) -> (W, ThriftFileMetadata) {
        (
            self.writer,
            self.metadata
                .expect("File to have ended and not be encrypted"),
        )
    }
}
