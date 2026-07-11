// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
use polars_parquet_format::{
    AesGcmCtrV1, AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
    EncryptionWithFooterKey, FileCryptoMetaData, thrift,
};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use zeroize::Zeroizing;

use super::ParquetEncryptionAlgorithm;
use super::ciphers::{
    AesCtrBlockEncryptor, AesGcmBlockEncryptor, BlockEncryptor, CipherInvocationCounter, NONCE_LEN,
    SIZE_LEN, TAG_LEN,
};
use super::modules::{ModuleType, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::SchemaDescriptor;

static NEXT_ENCRYPTION_PROPERTIES_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct EncryptionKey {
    key: Zeroizing<Vec<u8>>,
    key_metadata: Option<Vec<u8>>,
}

impl EncryptionKey {
    fn new(key: Vec<u8>) -> Self {
        Self {
            key: Zeroizing::new(key),
            key_metadata: None,
        }
    }

    fn with_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(metadata);
        self
    }
}

#[derive(Clone)]
pub struct FileEncryptionProperties {
    identity: u64,
    algorithm: ParquetEncryptionAlgorithm,
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: PlHashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl std::fmt::Debug for FileEncryptionProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileEncryptionProperties")
            .field("algorithm", &self.algorithm)
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

impl PartialEq for FileEncryptionProperties {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Hash for FileEncryptionProperties {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

impl FileEncryptionProperties {
    pub fn builder(footer_key: Vec<u8>) -> EncryptionPropertiesBuilder {
        EncryptionPropertiesBuilder::new(footer_key)
    }

    pub fn encrypt_footer(&self) -> bool {
        self.encrypt_footer
    }

    pub fn algorithm(&self) -> ParquetEncryptionAlgorithm {
        self.algorithm
    }

    pub fn footer_key_metadata(&self) -> Option<&Vec<u8>> {
        self.footer_key.key_metadata.as_ref()
    }

    pub fn footer_key(&self) -> &[u8] {
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
            .collect::<Vec<_>>();
        let mut missing = self
            .column_keys
            .keys()
            .filter(|configured_path| {
                !column_paths
                    .iter()
                    .any(|path| column_path_matches(configured_path, path))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            missing.sort();
            return Err(ParquetError::InvalidParameter(format!(
                "columns with encryption keys were not found in schema: {}",
                missing.join(", ")
            )));
        }
        Ok(())
    }

    fn column_key_entry(&self, column_path: &str) -> Option<(&str, &EncryptionKey)> {
        self.column_keys
            .iter()
            .filter(|(configured_path, _)| column_path_matches(configured_path, column_path))
            .max_by_key(|(configured_path, _)| configured_path.len())
            .map(|(configured_path, key)| (configured_path.as_str(), key))
    }
}

pub struct EncryptionPropertiesBuilder {
    algorithm: ParquetEncryptionAlgorithm,
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: PlHashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl EncryptionPropertiesBuilder {
    pub fn new(footer_key: Vec<u8>) -> Self {
        Self {
            algorithm: ParquetEncryptionAlgorithm::default(),
            footer_key: EncryptionKey::new(footer_key),
            column_keys: PlHashMap::new(),
            aad_prefix: None,
            encrypt_footer: true,
            store_aad_prefix: false,
        }
    }

    pub fn with_algorithm(mut self, algorithm: ParquetEncryptionAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
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
        validate_key_length(&self.footer_key.key, "footer")?;
        for (column_name, key) in &self.column_keys {
            validate_key_length(&key.key, &format!("column '{column_name}'"))?;
        }
        Ok(Arc::new(FileEncryptionProperties {
            identity: NEXT_ENCRYPTION_PROPERTIES_ID.fetch_add(1, Ordering::Relaxed),
            algorithm: self.algorithm,
            encrypt_footer: self.encrypt_footer,
            footer_key: self.footer_key,
            column_keys: self.column_keys,
            aad_prefix: self.aad_prefix,
            store_aad_prefix: self.store_aad_prefix,
        }))
    }
}

fn column_path_matches(configured_path: &str, column_path: &str) -> bool {
    column_path == configured_path
        || column_path
            .strip_prefix(configured_path)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

fn validate_key_length(key: &[u8], key_name: &str) -> ParquetResult<()> {
    if matches!(key.len(), 16 | 24 | 32) {
        Ok(())
    } else {
        Err(ParquetError::InvalidParameter(format!(
            "{key_name} encryption key must contain 16, 24, or 32 bytes, got {}",
            key.len()
        )))
    }
}

#[derive(Debug)]
pub(crate) struct FileEncryptor {
    properties: Arc<FileEncryptionProperties>,
    aad_file_unique: Vec<u8>,
    file_aad: Vec<u8>,
    footer_invocations: Arc<CipherInvocationCounter>,
    column_invocations: PlHashMap<String, Arc<CipherInvocationCounter>>,
}

impl FileEncryptor {
    pub(crate) fn new(properties: Arc<FileEncryptionProperties>) -> ParquetResult<Self> {
        let mut aad_file_unique = vec![0u8; 8];
        getrandom::fill(&mut aad_file_unique)
            .map_err(|_| ParquetError::oos("failed to generate parquet file AAD"))?;

        let file_aad = match properties.aad_prefix.as_ref() {
            None => aad_file_unique.clone(),
            Some(aad_prefix) => [aad_prefix.as_slice(), aad_file_unique.as_slice()].concat(),
        };
        let footer_invocations = Arc::default();
        let mut column_invocations = PlHashMap::with_capacity(properties.column_keys.len());
        let mut counter_owners = Vec::<(String, Arc<CipherInvocationCounter>)>::new();
        for (column_name, column_key) in &properties.column_keys {
            let invocation_counter =
                if column_key.key.as_slice() == properties.footer_key.key.as_slice() {
                    Arc::clone(&footer_invocations)
                } else if let Some((_, counter)) = counter_owners.iter().find(|(owner, _)| {
                    properties.column_keys[owner].key.as_slice() == column_key.key.as_slice()
                }) {
                    Arc::clone(counter)
                } else {
                    let counter = Arc::new(CipherInvocationCounter::default());
                    counter_owners.push((column_name.clone(), Arc::clone(&counter)));
                    counter
                };
            column_invocations.insert(column_name.clone(), invocation_counter);
        }

        Ok(Self {
            properties,
            aad_file_unique,
            file_aad,
            footer_invocations,
            column_invocations,
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
            || self.properties.column_key_entry(column_path).is_some()
    }

    pub(crate) fn get_footer_encryptor(&self) -> ParquetResult<Box<dyn BlockEncryptor>> {
        Ok(Box::new(AesGcmBlockEncryptor::new_with_counter(
            &self.properties.footer_key.key,
            Arc::clone(&self.footer_invocations),
        )?))
    }

    fn column_key_and_counter(
        &self,
        column_path: &str,
    ) -> ParquetResult<(&[u8], Arc<CipherInvocationCounter>)> {
        if self.properties.column_keys.is_empty() {
            return Ok((
                &self.properties.footer_key.key,
                Arc::clone(&self.footer_invocations),
            ));
        }

        let (configured_path, column_key) = self
            .properties
            .column_key_entry(column_path)
            .ok_or_else(|| {
                ParquetError::InvalidParameter(format!("column '{column_path}' is not encrypted"))
            })?;
        let invocation_counter = self
            .column_invocations
            .get(configured_path)
            .expect("counter created for every column key");
        Ok((&column_key.key, Arc::clone(invocation_counter)))
    }

    pub(crate) fn get_column_metadata_encryptor(
        &self,
        column_path: &str,
    ) -> ParquetResult<Box<dyn BlockEncryptor>> {
        let (key, invocation_counter) = self.column_key_and_counter(column_path)?;
        Ok(Box::new(AesGcmBlockEncryptor::new_with_counter(
            key,
            invocation_counter,
        )?))
    }

    pub(crate) fn get_column_data_encryptor(
        &self,
        column_path: &str,
    ) -> ParquetResult<Box<dyn BlockEncryptor>> {
        let (key, invocation_counter) = self.column_key_and_counter(column_path)?;
        match self.properties.algorithm() {
            ParquetEncryptionAlgorithm::AesGcmV1 => Ok(Box::new(
                AesGcmBlockEncryptor::new_with_counter(key, invocation_counter)?,
            )),
            ParquetEncryptionAlgorithm::AesGcmCtrV1 => Ok(Box::new(
                AesCtrBlockEncryptor::new_with_counter(key, invocation_counter)?,
            )),
        }
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
        match self.properties.algorithm() {
            ParquetEncryptionAlgorithm::AesGcmV1 => EncryptionAlgorithm::AESGCMV1(AesGcmV1 {
                aad_prefix,
                aad_file_unique: Some(self.aad_file_unique().clone()),
                supply_aad_prefix,
            }),
            ParquetEncryptionAlgorithm::AesGcmCtrV1 => {
                EncryptionAlgorithm::AESGCMCTRV1(AesGcmCtrV1 {
                    aad_prefix,
                    aad_file_unique: Some(self.aad_file_unique().clone()),
                    supply_aad_prefix,
                })
            },
        }
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
            .column_key_entry(column_path)
            .map(|(_, encryption_key)| encryption_key)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_column_key_matches_nested_leaf_and_exact_key_wins() {
        let properties = FileEncryptionProperties::builder(b"0123456789abcdef".to_vec())
            .with_column_key("nested", b"abcdef0123456789".to_vec())
            .with_column_key("nested.private", b"fedcba9876543210".to_vec())
            .build()
            .unwrap();

        assert_eq!(
            properties.column_key_entry("nested.public").unwrap().0,
            "nested"
        );
        assert_eq!(
            properties.column_key_entry("nested.private").unwrap().0,
            "nested.private"
        );
        assert!(properties.column_key_entry("nested_other").is_none());
    }

    #[test]
    fn identical_keys_share_invocation_counter() {
        let shared_key = b"abcdef0123456789".to_vec();
        let properties = FileEncryptionProperties::builder(b"0123456789abcdef".to_vec())
            .with_column_key("first", shared_key.clone())
            .with_column_key("second", shared_key)
            .build()
            .unwrap();
        let encryptor = FileEncryptor::new(properties).unwrap();

        assert!(Arc::ptr_eq(
            &encryptor.column_invocations["first"],
            &encryptor.column_invocations["second"]
        ));
    }

    #[test]
    fn cloned_properties_keep_identity_but_new_properties_do_not_compare_equal() {
        let properties = FileEncryptionProperties::builder(b"0123456789abcdef".to_vec())
            .build()
            .unwrap();
        let cloned = Arc::clone(&properties);
        let other = FileEncryptionProperties::builder(b"0123456789abcdef".to_vec())
            .build()
            .unwrap();

        assert_eq!(properties, cloned);
        assert_ne!(properties, other);
    }
}
