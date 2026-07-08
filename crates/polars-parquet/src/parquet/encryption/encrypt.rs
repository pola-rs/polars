// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use std::collections::{HashMap, HashSet};
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Arc;

use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
use polars_parquet_format::{
    AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
    EncryptionWithFooterKey, FileCryptoMetaData, thrift,
};
use ring::rand::{SecureRandom, SystemRandom};

use super::ciphers::{BlockEncryptor, NONCE_LEN, RingGcmBlockEncryptor, SIZE_LEN, TAG_LEN};
use super::modules::{ModuleType, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::SchemaDescriptor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EncryptionKey {
    key: Vec<u8>,
    key_metadata: Option<Vec<u8>>,
}

impl EncryptionKey {
    fn new(key: Vec<u8>) -> Self {
        Self {
            key,
            key_metadata: None,
        }
    }

    fn with_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(metadata);
        self
    }
}

#[derive(Clone, PartialEq)]
pub struct FileEncryptionProperties {
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: HashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl std::fmt::Debug for FileEncryptionProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileEncryptionProperties")
            .field("encrypt_footer", &self.encrypt_footer)
            .field("footer_key", &"<redacted>")
            .field("column_keys", &self.column_keys.keys().collect::<Vec<_>>())
            .field(
                "aad_prefix",
                &self.aad_prefix.as_ref().map(|_| "<redacted>"),
            )
            .field("store_aad_prefix", &self.store_aad_prefix)
            .finish()
    }
}

impl Eq for FileEncryptionProperties {}

impl Hash for FileEncryptionProperties {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.encrypt_footer.hash(state);
        self.footer_key.hash(state);

        let mut column_keys = self.column_keys.iter().collect::<Vec<_>>();
        column_keys.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        for (column_name, encryption_key) in column_keys {
            column_name.hash(state);
            encryption_key.hash(state);
        }

        self.aad_prefix.hash(state);
        self.store_aad_prefix.hash(state);
    }
}

impl FileEncryptionProperties {
    pub fn builder(footer_key: Vec<u8>) -> EncryptionPropertiesBuilder {
        EncryptionPropertiesBuilder::new(footer_key)
    }

    pub fn encrypt_footer(&self) -> bool {
        self.encrypt_footer
    }

    pub fn footer_key_metadata(&self) -> Option<&Vec<u8>> {
        self.footer_key.key_metadata.as_ref()
    }

    pub fn footer_key(&self) -> &Vec<u8> {
        &self.footer_key.key
    }

    pub fn aad_prefix(&self) -> Option<&Vec<u8>> {
        self.aad_prefix.as_ref()
    }

    pub fn store_aad_prefix(&self) -> bool {
        self.store_aad_prefix && self.aad_prefix.is_some()
    }

    pub(crate) fn validate_encrypted_column_names(
        &self,
        schema: &SchemaDescriptor,
    ) -> ParquetResult<()> {
        let column_paths = schema
            .columns()
            .iter()
            .map(|c| c.path_in_schema.join("."))
            .collect::<HashSet<_>>();
        let encryption_columns = self.column_keys.keys().cloned().collect::<HashSet<_>>();
        if !encryption_columns.is_subset(&column_paths) {
            let mut missing = encryption_columns
                .difference(&column_paths)
                .cloned()
                .collect::<Vec<_>>();
            missing.sort();
            return Err(ParquetError::InvalidParameter(format!(
                "columns with encryption keys were not found in schema: {}",
                missing.join(", ")
            )));
        }
        Ok(())
    }
}

pub struct EncryptionPropertiesBuilder {
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: HashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl EncryptionPropertiesBuilder {
    pub fn new(footer_key: Vec<u8>) -> Self {
        Self {
            footer_key: EncryptionKey::new(footer_key),
            column_keys: HashMap::new(),
            aad_prefix: None,
            encrypt_footer: true,
            store_aad_prefix: false,
        }
    }

    pub fn with_plaintext_footer(mut self, plaintext_footer: bool) -> Self {
        self.encrypt_footer = !plaintext_footer;
        self
    }

    pub fn with_footer_key_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.footer_key = self.footer_key.with_metadata(metadata);
        self
    }

    pub fn with_column_key(mut self, column_name: &str, key: Vec<u8>) -> Self {
        self.column_keys
            .insert(column_name.to_string(), EncryptionKey::new(key));
        self
    }

    pub fn with_column_key_and_metadata(
        mut self,
        column_name: &str,
        key: Vec<u8>,
        metadata: Vec<u8>,
    ) -> Self {
        self.column_keys.insert(
            column_name.to_string(),
            EncryptionKey::new(key).with_metadata(metadata),
        );
        self
    }

    pub fn with_aad_prefix(mut self, aad_prefix: Vec<u8>) -> Self {
        self.aad_prefix = Some(aad_prefix);
        self
    }

    pub fn with_aad_prefix_storage(mut self, store_aad_prefix: bool) -> Self {
        self.store_aad_prefix = store_aad_prefix;
        self
    }

    pub fn build(self) -> ParquetResult<Arc<FileEncryptionProperties>> {
        Ok(Arc::new(FileEncryptionProperties {
            encrypt_footer: self.encrypt_footer,
            footer_key: self.footer_key,
            column_keys: self.column_keys,
            aad_prefix: self.aad_prefix,
            store_aad_prefix: self.store_aad_prefix,
        }))
    }
}

#[derive(Debug)]
pub(crate) struct FileEncryptor {
    properties: Arc<FileEncryptionProperties>,
    aad_file_unique: Vec<u8>,
    file_aad: Vec<u8>,
}

impl FileEncryptor {
    pub(crate) fn new(properties: Arc<FileEncryptionProperties>) -> ParquetResult<Self> {
        let mut aad_file_unique = vec![0u8; 8];
        SystemRandom::new()
            .fill(&mut aad_file_unique)
            .map_err(|_| ParquetError::oos("failed to generate parquet file AAD"))?;

        let file_aad = match properties.aad_prefix.as_ref() {
            None => aad_file_unique.clone(),
            Some(aad_prefix) => [aad_prefix.as_slice(), aad_file_unique.as_slice()].concat(),
        };

        Ok(Self {
            properties,
            aad_file_unique,
            file_aad,
        })
    }

    pub(crate) fn properties(&self) -> &Arc<FileEncryptionProperties> {
        &self.properties
    }

    pub(crate) fn file_aad(&self) -> &[u8] {
        &self.file_aad
    }

    pub(crate) fn aad_file_unique(&self) -> &Vec<u8> {
        &self.aad_file_unique
    }

    pub(crate) fn is_column_encrypted(&self, column_path: &str) -> bool {
        self.properties.column_keys.is_empty()
            || self.properties.column_keys.contains_key(column_path)
    }

    pub(crate) fn get_footer_encryptor(&self) -> ParquetResult<Box<dyn BlockEncryptor>> {
        Ok(Box::new(RingGcmBlockEncryptor::new(
            &self.properties.footer_key.key,
        )?))
    }

    pub(crate) fn get_column_encryptor(
        &self,
        column_path: &str,
    ) -> ParquetResult<Box<dyn BlockEncryptor>> {
        if self.properties.column_keys.is_empty() {
            return self.get_footer_encryptor();
        }

        let column_key = self
            .properties
            .column_keys
            .get(column_path)
            .ok_or_else(|| {
                ParquetError::InvalidParameter(format!("column '{column_path}' is not encrypted"))
            })?;
        Ok(Box::new(RingGcmBlockEncryptor::new(&column_key.key)?))
    }

    pub(crate) fn encryption_algorithm(&self) -> EncryptionAlgorithm {
        let supply_aad_prefix = self
            .properties
            .aad_prefix()
            .map(|_| !self.properties.store_aad_prefix());
        let aad_prefix = if self.properties.store_aad_prefix() {
            self.properties.aad_prefix().cloned()
        } else {
            None
        };
        EncryptionAlgorithm::AESGCMV1(AesGcmV1 {
            aad_prefix,
            aad_file_unique: Some(self.aad_file_unique().clone()),
            supply_aad_prefix,
        })
    }

    pub(crate) fn file_crypto_metadata(&self) -> FileCryptoMetaData {
        FileCryptoMetaData {
            encryption_algorithm: self.encryption_algorithm(),
            key_metadata: self.properties.footer_key_metadata().cloned(),
        }
    }

    pub(crate) fn column_crypto_metadata(
        &self,
        column_path: &str,
        path_in_schema: &[String],
    ) -> Option<ColumnCryptoMetaData> {
        if self.properties.column_keys.is_empty() {
            return Some(ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(
                EncryptionWithFooterKey {},
            ));
        }

        self.properties
            .column_keys
            .get(column_path)
            .map(|encryption_key| {
                ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(EncryptionWithColumnKey {
                    path_in_schema: path_in_schema.to_vec(),
                    key_metadata: encryption_key.key_metadata.clone(),
                })
            })
    }
}

pub(crate) fn encrypt_bytes(
    plaintext: &[u8],
    encryptor: &mut dyn BlockEncryptor,
    module_aad: &[u8],
) -> ParquetResult<Vec<u8>> {
    encryptor.encrypt(plaintext, module_aad)
}

pub(crate) fn write_encrypted_thrift_object<W: Write>(
    sink: &mut W,
    encryptor: &mut dyn BlockEncryptor,
    module_aad: &[u8],
    write_object: impl FnOnce(&mut TCompactOutputProtocol<&mut Vec<u8>>) -> thrift::Result<usize>,
) -> ParquetResult<u64> {
    let encrypted_buffer = encrypted_thrift_object_to_vec(encryptor, module_aad, write_object)?;
    sink.write_all(&encrypted_buffer)?;
    Ok(encrypted_buffer.len() as u64)
}

pub(crate) fn encrypted_thrift_object_to_vec(
    encryptor: &mut dyn BlockEncryptor,
    module_aad: &[u8],
    write_object: impl FnOnce(&mut TCompactOutputProtocol<&mut Vec<u8>>) -> thrift::Result<usize>,
) -> ParquetResult<Vec<u8>> {
    let mut plaintext = vec![];
    {
        let mut protocol = TCompactOutputProtocol::new(&mut plaintext);
        write_object(&mut protocol)?;
    }
    encryptor.encrypt(&plaintext, module_aad)
}

pub(crate) fn write_signed_plaintext_thrift_object<W: Write>(
    sink: &mut W,
    encryptor: &mut dyn BlockEncryptor,
    module_aad: &[u8],
    write_object: impl FnOnce(&mut TCompactOutputProtocol<&mut Vec<u8>>) -> thrift::Result<usize>,
) -> ParquetResult<u64> {
    let mut plaintext = vec![];
    {
        let mut protocol = TCompactOutputProtocol::new(&mut plaintext);
        write_object(&mut protocol)?;
    }
    sink.write_all(&plaintext)?;

    let encrypted = encryptor.encrypt(&plaintext, module_aad)?;
    let nonce = &encrypted[SIZE_LEN..SIZE_LEN + NONCE_LEN];
    let tag = &encrypted[encrypted.len() - TAG_LEN..];
    sink.write_all(nonce)?;
    sink.write_all(tag)?;

    Ok((plaintext.len() + NONCE_LEN + TAG_LEN) as u64)
}

pub(crate) fn column_metadata_aad(
    file_encryptor: &FileEncryptor,
    row_group_index: usize,
    column_index: usize,
) -> ParquetResult<Vec<u8>> {
    create_module_aad(
        file_encryptor.file_aad(),
        ModuleType::ColumnMetaData,
        row_group_index,
        column_index,
        None,
    )
}
