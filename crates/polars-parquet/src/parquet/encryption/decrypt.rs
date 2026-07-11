// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use std::borrow::Cow;
use std::fmt::Formatter;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polars_parquet_format::{
    AesGcmCtrV1, AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use zeroize::Zeroizing;

use super::ParquetEncryptionAlgorithm;
use super::ciphers::{
    AesCtrBlockDecryptor, AesGcmBlockDecryptor, BlockDecryptor, SIZE_LEN, TAG_LEN,
};
use super::modules::{ModuleType, create_footer_aad, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};

pub trait KeyRetriever: Send + Sync {
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>>;
}

static NEXT_DECRYPTION_PROPERTIES_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn read_and_decrypt<T: Read>(
    decryptor: &Arc<dyn BlockDecryptor>,
    input: &mut T,
    aad: &[u8],
    max_ciphertext_len: usize,
) -> ParquetResult<Vec<u8>> {
    let mut len_bytes = [0; SIZE_LEN];
    input.read_exact(&mut len_bytes)?;
    let ciphertext_len = u32::from_le_bytes(len_bytes) as usize;
    if ciphertext_len > max_ciphertext_len {
        return Err(ParquetError::WouldOverAllocate);
    }
    let mut ciphertext = vec![0; SIZE_LEN + ciphertext_len];
    ciphertext[..SIZE_LEN].copy_from_slice(&len_bytes);
    input.read_exact(&mut ciphertext[SIZE_LEN..])?;
    decryptor.decrypt(&ciphertext, aad)
}

#[derive(Debug, Clone)]
pub(crate) struct CryptoContext {
    pub(crate) row_group_index: usize,
    pub(crate) column_ordinal: usize,
    pub(crate) page_ordinal: Option<usize>,
    pub(crate) dictionary_page: bool,
    data_decryptor: Arc<dyn BlockDecryptor>,
    metadata_decryptor: Arc<dyn BlockDecryptor>,
    file_aad: Vec<u8>,
}

impl CryptoContext {
    pub(crate) fn for_column(
        file_decryptor: &FileDecryptor,
        column_crypto_metadata: &ColumnCryptoMetaData,
        row_group_index: usize,
        column_ordinal: usize,
    ) -> ParquetResult<Self> {
        let (data_decryptor, metadata_decryptor) = match column_crypto_metadata {
            ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(_) => {
                let data_decryptor = file_decryptor.get_footer_data_decryptor()?;
                let metadata_decryptor = file_decryptor.get_footer_decryptor()?;
                (data_decryptor, metadata_decryptor)
            },
            ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(column_key_encryption) => {
                let column_name = column_name_from_metadata(column_key_encryption);
                let (data_decryptor, metadata_decryptor) = file_decryptor.get_column_decryptors(
                    &column_name,
                    column_key_encryption.key_metadata.as_deref(),
                )?;
                (data_decryptor, metadata_decryptor)
            },
        };

        Ok(Self {
            row_group_index,
            column_ordinal,
            page_ordinal: None,
            dictionary_page: false,
            data_decryptor,
            metadata_decryptor,
            file_aad: file_decryptor.file_aad().clone(),
        })
    }

    pub(crate) fn with_page_ordinal(&self, page_ordinal: usize) -> Self {
        Self {
            row_group_index: self.row_group_index,
            column_ordinal: self.column_ordinal,
            page_ordinal: Some(page_ordinal),
            dictionary_page: false,
            data_decryptor: self.data_decryptor.clone(),
            metadata_decryptor: self.metadata_decryptor.clone(),
            file_aad: self.file_aad.clone(),
        }
    }

    pub(crate) fn for_dictionary_page(&self) -> Self {
        Self {
            row_group_index: self.row_group_index,
            column_ordinal: self.column_ordinal,
            page_ordinal: self.page_ordinal,
            dictionary_page: true,
            data_decryptor: self.data_decryptor.clone(),
            metadata_decryptor: self.metadata_decryptor.clone(),
            file_aad: self.file_aad.clone(),
        }
    }

    pub(crate) fn create_page_header_aad(&self) -> ParquetResult<Vec<u8>> {
        let module_type = if self.dictionary_page {
            ModuleType::DictionaryPageHeader
        } else {
            ModuleType::DataPageHeader
        };
        create_module_aad(
            &self.file_aad,
            module_type,
            self.row_group_index,
            self.column_ordinal,
            self.page_ordinal,
        )
    }

    pub(crate) fn create_page_aad(&self) -> ParquetResult<Vec<u8>> {
        let module_type = if self.dictionary_page {
            ModuleType::DictionaryPage
        } else {
            ModuleType::DataPage
        };
        create_module_aad(
            &self.file_aad,
            module_type,
            self.row_group_index,
            self.column_ordinal,
            self.page_ordinal,
        )
    }

    pub(crate) fn create_column_metadata_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            &self.file_aad,
            ModuleType::ColumnMetaData,
            self.row_group_index,
            self.column_ordinal,
            None,
        )
    }

    pub(crate) fn create_column_index_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            &self.file_aad,
            ModuleType::ColumnIndex,
            self.row_group_index,
            self.column_ordinal,
            None,
        )
    }

    pub(crate) fn create_offset_index_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            &self.file_aad,
            ModuleType::OffsetIndex,
            self.row_group_index,
            self.column_ordinal,
            None,
        )
    }

    #[cfg(feature = "bloom_filter")]
    pub(crate) fn create_bloom_filter_header_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            &self.file_aad,
            ModuleType::BloomFilterHeader,
            self.row_group_index,
            self.column_ordinal,
            None,
        )
    }

    #[cfg(feature = "bloom_filter")]
    pub(crate) fn create_bloom_filter_bitset_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            &self.file_aad,
            ModuleType::BloomFilterBitset,
            self.row_group_index,
            self.column_ordinal,
            None,
        )
    }

    pub(crate) fn data_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.data_decryptor
    }

    pub(crate) fn metadata_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.metadata_decryptor
    }
}

#[derive(Clone)]
struct ExplicitDecryptionKeys {
    footer_key: Zeroizing<Vec<u8>>,
    column_keys: PlHashMap<String, Zeroizing<Vec<u8>>>,
}

#[derive(Clone)]
enum DecryptionKeys {
    Explicit(ExplicitDecryptionKeys),
    ViaRetriever(Arc<dyn KeyRetriever>),
}

#[derive(Clone)]
pub struct FileDecryptionProperties {
    identity: u64,
    keys: DecryptionKeys,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl Eq for FileDecryptionProperties {}

impl PartialEq for FileDecryptionProperties {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Hash for FileDecryptionProperties {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

impl FileDecryptionProperties {
    pub fn builder(footer_key: Vec<u8>) -> DecryptionPropertiesBuilder {
        DecryptionPropertiesBuilder::new(footer_key)
    }

    pub fn with_key_retriever(
        key_retriever: Arc<dyn KeyRetriever>,
    ) -> DecryptionPropertiesBuilderWithRetriever {
        DecryptionPropertiesBuilderWithRetriever::new(key_retriever)
    }

    pub fn aad_prefix(&self) -> Option<&Vec<u8>> {
        self.aad_prefix.as_ref()
    }

    pub fn check_plaintext_footer_integrity(&self) -> bool {
        self.footer_signature_verification
    }

    pub fn footer_key(&self, key_metadata: Option<&[u8]>) -> ParquetResult<Cow<'_, [u8]>> {
        match &self.keys {
            DecryptionKeys::Explicit(keys) => Ok(Cow::Borrowed(&keys.footer_key)),
            DecryptionKeys::ViaRetriever(retriever) => Ok(Cow::Owned(
                retriever.retrieve_key(key_metadata.unwrap_or_default())?,
            )),
        }
    }

    pub fn column_key(
        &self,
        column_name: &str,
        key_metadata: Option<&[u8]>,
    ) -> ParquetResult<Cow<'_, [u8]>> {
        match &self.keys {
            DecryptionKeys::Explicit(keys) => keys
                .column_keys
                .iter()
                .filter(|(configured_path, _)| column_path_matches(configured_path, column_name))
                .max_by_key(|(configured_path, _)| configured_path.len())
                .map(|(_, key)| key)
                .map(|key| Cow::Borrowed(key.as_slice()))
                .ok_or_else(|| {
                    ParquetError::InvalidParameter(format!(
                        "no decryption key set for encrypted column '{column_name}'"
                    ))
                }),
            DecryptionKeys::ViaRetriever(retriever) => Ok(Cow::Owned(
                retriever.retrieve_key(key_metadata.unwrap_or_default())?,
            )),
        }
    }
}

impl std::fmt::Debug for FileDecryptionProperties {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("FileDecryptionProperties { .. }")
    }
}

pub struct DecryptionPropertiesBuilder {
    footer_key: Vec<u8>,
    column_keys: PlHashMap<String, Vec<u8>>,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl DecryptionPropertiesBuilder {
    pub fn new(footer_key: Vec<u8>) -> Self {
        Self {
            footer_key,
            column_keys: PlHashMap::new(),
            aad_prefix: None,
            footer_signature_verification: true,
        }
    }

    pub fn build(self) -> ParquetResult<Arc<FileDecryptionProperties>> {
        validate_key_length(&self.footer_key, "footer")?;
        for (column_name, key) in &self.column_keys {
            validate_key_length(key, &format!("column '{column_name}'"))?;
        }
        Ok(Arc::new(FileDecryptionProperties {
            identity: NEXT_DECRYPTION_PROPERTIES_ID.fetch_add(1, Ordering::Relaxed),
            keys: DecryptionKeys::Explicit(ExplicitDecryptionKeys {
                footer_key: Zeroizing::new(self.footer_key),
                column_keys: self
                    .column_keys
                    .into_iter()
                    .map(|(name, key)| (name, Zeroizing::new(key)))
                    .collect(),
            }),
            aad_prefix: self.aad_prefix,
            footer_signature_verification: self.footer_signature_verification,
        }))
    }

    pub fn with_aad_prefix(mut self, value: Vec<u8>) -> Self {
        self.aad_prefix = Some(value);
        self
    }

    pub fn with_column_key(mut self, column_name: &str, decryption_key: Vec<u8>) -> Self {
        self.column_keys
            .insert(column_name.to_string(), decryption_key);
        self
    }

    pub fn disable_footer_signature_verification(mut self) -> Self {
        self.footer_signature_verification = false;
        self
    }
}

fn validate_key_length(key: &[u8], key_name: &str) -> ParquetResult<()> {
    if matches!(key.len(), 16 | 24 | 32) {
        Ok(())
    } else {
        Err(ParquetError::InvalidParameter(format!(
            "{key_name} decryption key must contain 16, 24, or 32 bytes, got {}",
            key.len()
        )))
    }
}

fn column_path_matches(configured_path: &str, column_path: &str) -> bool {
    column_path == configured_path
        || column_path
            .strip_prefix(configured_path)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

pub struct DecryptionPropertiesBuilderWithRetriever {
    key_retriever: Arc<dyn KeyRetriever>,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl DecryptionPropertiesBuilderWithRetriever {
    pub fn new(key_retriever: Arc<dyn KeyRetriever>) -> Self {
        Self {
            key_retriever,
            aad_prefix: None,
            footer_signature_verification: true,
        }
    }

    pub fn build(self) -> ParquetResult<Arc<FileDecryptionProperties>> {
        Ok(Arc::new(FileDecryptionProperties {
            identity: NEXT_DECRYPTION_PROPERTIES_ID.fetch_add(1, Ordering::Relaxed),
            keys: DecryptionKeys::ViaRetriever(self.key_retriever),
            aad_prefix: self.aad_prefix,
            footer_signature_verification: self.footer_signature_verification,
        }))
    }

    pub fn with_aad_prefix(mut self, value: Vec<u8>) -> Self {
        self.aad_prefix = Some(value);
        self
    }

    pub fn disable_footer_signature_verification(mut self) -> Self {
        self.footer_signature_verification = false;
        self
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FileDecryptor {
    decryption_properties: Arc<FileDecryptionProperties>,
    footer_decryptor: Arc<dyn BlockDecryptor>,
    footer_data_decryptor: Arc<dyn BlockDecryptor>,
    file_aad: Vec<u8>,
    algorithm: ParquetEncryptionAlgorithm,
}

impl PartialEq for FileDecryptor {
    fn eq(&self, other: &Self) -> bool {
        self.decryption_properties == other.decryption_properties
            && self.file_aad == other.file_aad
            && self.algorithm == other.algorithm
    }
}

impl FileDecryptor {
    pub(crate) fn new(
        decryption_properties: &Arc<FileDecryptionProperties>,
        footer_key_metadata: Option<&[u8]>,
        aad_file_unique: Vec<u8>,
        aad_prefix: Vec<u8>,
        algorithm: ParquetEncryptionAlgorithm,
    ) -> ParquetResult<Self> {
        let file_aad = [aad_prefix.as_slice(), aad_file_unique.as_slice()].concat();
        let footer_key = decryption_properties.footer_key(footer_key_metadata)?;
        let footer_decryptor = AesGcmBlockDecryptor::new(&footer_key).map_err(|err| {
            ParquetError::InvalidParameter(format!("invalid footer decryption key: {err}"))
        })?;
        let footer_data_decryptor: Arc<dyn BlockDecryptor> = match algorithm {
            ParquetEncryptionAlgorithm::AesGcmV1 => {
                Arc::new(AesGcmBlockDecryptor::new(&footer_key)?)
            },
            ParquetEncryptionAlgorithm::AesGcmCtrV1 => {
                Arc::new(AesCtrBlockDecryptor::new(&footer_key)?)
            },
        };

        Ok(Self {
            footer_decryptor: Arc::new(footer_decryptor),
            footer_data_decryptor,
            decryption_properties: Arc::clone(decryption_properties),
            file_aad,
            algorithm,
        })
    }

    pub(crate) fn from_algorithm(
        encryption_algorithm: EncryptionAlgorithm,
        footer_key_metadata: Option<&[u8]>,
        decryption_properties: &Arc<FileDecryptionProperties>,
    ) -> ParquetResult<Self> {
        let (algorithm, aad_prefix, aad_file_unique, supply_aad_prefix) = match encryption_algorithm
        {
            EncryptionAlgorithm::AESGCMV1(AesGcmV1 {
                aad_prefix,
                aad_file_unique,
                supply_aad_prefix,
            }) => (
                ParquetEncryptionAlgorithm::AesGcmV1,
                aad_prefix,
                aad_file_unique,
                supply_aad_prefix,
            ),
            EncryptionAlgorithm::AESGCMCTRV1(AesGcmCtrV1 {
                aad_prefix,
                aad_file_unique,
                supply_aad_prefix,
            }) => (
                ParquetEncryptionAlgorithm::AesGcmCtrV1,
                aad_prefix,
                aad_file_unique,
                supply_aad_prefix,
            ),
        };
        if supply_aad_prefix.unwrap_or(false) && decryption_properties.aad_prefix().is_none() {
            return Err(ParquetError::InvalidParameter(
                "file requires an AAD prefix, but none was provided".to_string(),
            ));
        }
        let aad_file_unique = aad_file_unique.ok_or_else(|| {
            ParquetError::oos("encrypted parquet metadata does not contain aad_file_unique")
        })?;
        let aad_prefix = decryption_properties
            .aad_prefix()
            .cloned()
            .or(aad_prefix)
            .unwrap_or_default();

        Self::new(
            decryption_properties,
            footer_key_metadata,
            aad_file_unique,
            aad_prefix,
            algorithm,
        )
    }

    pub(crate) fn get_footer_decryptor(&self) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        Ok(self.footer_decryptor.clone())
    }

    pub(crate) fn get_footer_data_decryptor(&self) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        Ok(self.footer_data_decryptor.clone())
    }

    pub(crate) fn verify_plaintext_footer_signature(
        &self,
        plaintext_footer: &[u8],
    ) -> ParquetResult<()> {
        if plaintext_footer.len() < TAG_LEN {
            return Err(ParquetError::oos(
                "plaintext encrypted footer is missing signature bytes",
            ));
        }

        let expected_tag = &plaintext_footer[plaintext_footer.len() - TAG_LEN..];
        let aad = create_footer_aad(self.file_aad())?;
        let computed_tag = self
            .footer_decryptor
            .compute_plaintext_tag(&aad, plaintext_footer)?;

        if computed_tag != expected_tag {
            return Err(ParquetError::oos(
                "plaintext encrypted parquet footer signature verification failed",
            ));
        }
        Ok(())
    }

    fn create_data_decryptor(&self, key: &[u8]) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        match self.algorithm {
            ParquetEncryptionAlgorithm::AesGcmV1 => Ok(Arc::new(AesGcmBlockDecryptor::new(key)?)),
            ParquetEncryptionAlgorithm::AesGcmCtrV1 => {
                Ok(Arc::new(AesCtrBlockDecryptor::new(key)?))
            },
        }
    }

    pub(crate) fn get_column_decryptors(
        &self,
        column_name: &str,
        key_metadata: Option<&[u8]>,
    ) -> ParquetResult<(Arc<dyn BlockDecryptor>, Arc<dyn BlockDecryptor>)> {
        let column_key = self
            .decryption_properties
            .column_key(column_name, key_metadata)?;
        Ok((
            self.create_data_decryptor(&column_key)?,
            Arc::new(AesGcmBlockDecryptor::new(&column_key)?),
        ))
    }

    pub(crate) fn file_aad(&self) -> &Vec<u8> {
        &self.file_aad
    }
}

pub(crate) fn column_name_from_metadata(metadata: &EncryptionWithColumnKey) -> String {
    metadata.path_in_schema.join(".")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct NeverDecrypt;

    impl BlockDecryptor for NeverDecrypt {
        fn decrypt(&self, _: &[u8], _: &[u8]) -> ParquetResult<Vec<u8>> {
            panic!("length guard must run before decryption")
        }

        fn compute_plaintext_tag(&self, _: &[u8], _: &[u8]) -> ParquetResult<Vec<u8>> {
            unreachable!()
        }
    }

    #[test]
    fn encrypted_module_length_is_checked_before_allocation() {
        let decryptor: Arc<dyn BlockDecryptor> = Arc::new(NeverDecrypt);
        let mut input = std::io::Cursor::new(u32::MAX.to_le_bytes());
        assert!(matches!(
            read_and_decrypt(&decryptor, &mut input, b"aad", 1024),
            Err(ParquetError::WouldOverAllocate)
        ));
    }

    #[test]
    fn explicit_root_key_matches_nested_column() {
        let key = b"0123456789abcdef".to_vec();
        let properties = FileDecryptionProperties::builder(key.clone())
            .with_column_key("nested", key.clone())
            .build()
            .unwrap();

        assert_eq!(
            properties.column_key("nested.leaf", None).unwrap().as_ref(),
            key
        );
    }

    #[test]
    fn decryption_builder_rejects_invalid_key_lengths() {
        let error = FileDecryptionProperties::builder(vec![0; 15])
            .build()
            .unwrap_err();
        assert!(error.to_string().contains("16, 24, or 32"));
    }

    #[test]
    fn decryption_properties_compare_by_identity() {
        let properties = FileDecryptionProperties::builder(b"0123456789abcdef".to_vec())
            .build()
            .unwrap();
        let cloned = Arc::clone(&properties);
        let other = FileDecryptionProperties::builder(b"0123456789abcdef".to_vec())
            .build()
            .unwrap();

        assert_eq!(properties, cloned);
        assert_ne!(properties, other);
    }
}
