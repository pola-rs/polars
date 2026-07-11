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

use polars_parquet_format::{
    AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
};
use polars_utils::aliases::{InitHashMaps, PlHashMap};

use super::ciphers::{BlockDecryptor, RingGcmBlockDecryptor, SIZE_LEN, TAG_LEN};
use super::modules::{ModuleType, create_footer_aad, create_module_aad};
use crate::parquet::error::{ParquetError, ParquetResult};

pub trait KeyRetriever: Send + Sync {
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>>;
}

pub(crate) fn read_and_decrypt<T: Read>(
    decryptor: &Arc<dyn BlockDecryptor>,
    input: &mut T,
    aad: &[u8],
) -> ParquetResult<Vec<u8>> {
    let mut len_bytes = [0; SIZE_LEN];
    input.read_exact(&mut len_bytes)?;
    let ciphertext_len = u32::from_le_bytes(len_bytes) as usize;
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
                let data_decryptor = file_decryptor.get_footer_decryptor()?;
                let metadata_decryptor = file_decryptor.get_footer_decryptor()?;
                (data_decryptor, metadata_decryptor)
            },
            ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(column_key_encryption) => {
                let column_name = column_name_from_metadata(column_key_encryption);
                let data_decryptor = file_decryptor.get_column_data_decryptor(
                    &column_name,
                    column_key_encryption.key_metadata.as_deref(),
                )?;
                let metadata_decryptor = file_decryptor.get_column_metadata_decryptor(
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

    pub(crate) fn data_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.data_decryptor
    }

    pub(crate) fn metadata_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.metadata_decryptor
    }
}

#[derive(Clone, PartialEq, Eq)]
struct ExplicitDecryptionKeys {
    footer_key: Vec<u8>,
    column_keys: PlHashMap<String, Vec<u8>>,
}

impl Hash for ExplicitDecryptionKeys {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.footer_key.hash(state);

        let mut column_keys = self.column_keys.iter().collect::<Vec<_>>();
        column_keys.sort_unstable_by_key(|(column_name, _)| *column_name);
        for (column_name, key) in column_keys {
            column_name.hash(state);
            key.hash(state);
        }
    }
}

#[derive(Clone)]
enum DecryptionKeys {
    Explicit(ExplicitDecryptionKeys),
    ViaRetriever(Arc<dyn KeyRetriever>),
}

impl PartialEq for DecryptionKeys {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Explicit(keys), Self::Explicit(other_keys)) => keys == other_keys,
            (Self::ViaRetriever(_), Self::ViaRetriever(_)) => true,
            _ => false,
        }
    }
}

impl Eq for DecryptionKeys {}

impl Hash for DecryptionKeys {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Explicit(keys) => {
                0u8.hash(state);
                keys.hash(state);
            },
            Self::ViaRetriever(_) => {
                1u8.hash(state);
            },
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct FileDecryptionProperties {
    keys: DecryptionKeys,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl Eq for FileDecryptionProperties {}

impl Hash for FileDecryptionProperties {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.keys.hash(state);
        self.aad_prefix.hash(state);
        self.footer_signature_verification.hash(state);
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

    pub fn footer_key(&self, key_metadata: Option<&[u8]>) -> ParquetResult<Cow<'_, Vec<u8>>> {
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
    ) -> ParquetResult<Cow<'_, Vec<u8>>> {
        match &self.keys {
            DecryptionKeys::Explicit(keys) => keys
                .column_keys
                .get(column_name)
                .map(Cow::Borrowed)
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
        Ok(Arc::new(FileDecryptionProperties {
            keys: DecryptionKeys::Explicit(ExplicitDecryptionKeys {
                footer_key: self.footer_key,
                column_keys: self.column_keys,
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
    file_aad: Vec<u8>,
}

impl PartialEq for FileDecryptor {
    fn eq(&self, other: &Self) -> bool {
        self.decryption_properties == other.decryption_properties && self.file_aad == other.file_aad
    }
}

impl FileDecryptor {
    pub(crate) fn new(
        decryption_properties: &Arc<FileDecryptionProperties>,
        footer_key_metadata: Option<&[u8]>,
        aad_file_unique: Vec<u8>,
        aad_prefix: Vec<u8>,
    ) -> ParquetResult<Self> {
        let file_aad = [aad_prefix.as_slice(), aad_file_unique.as_slice()].concat();
        let footer_key = decryption_properties.footer_key(footer_key_metadata)?;
        let footer_decryptor = RingGcmBlockDecryptor::new(&footer_key).map_err(|err| {
            ParquetError::InvalidParameter(format!("invalid footer decryption key: {err}"))
        })?;

        Ok(Self {
            footer_decryptor: Arc::new(footer_decryptor),
            decryption_properties: Arc::clone(decryption_properties),
            file_aad,
        })
    }

    pub(crate) fn from_algorithm(
        encryption_algorithm: EncryptionAlgorithm,
        footer_key_metadata: Option<&[u8]>,
        decryption_properties: &Arc<FileDecryptionProperties>,
    ) -> ParquetResult<Self> {
        match encryption_algorithm {
            EncryptionAlgorithm::AESGCMV1(AesGcmV1 {
                aad_prefix,
                aad_file_unique,
                supply_aad_prefix,
            }) => {
                if supply_aad_prefix.unwrap_or(false)
                    && decryption_properties.aad_prefix().is_none()
                {
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
                )
            },
            EncryptionAlgorithm::AESGCMCTRV1(_) => Err(ParquetError::not_supported(
                "AES_GCM_CTR_V1 parquet encryption is not supported yet",
            )),
        }
    }

    pub(crate) fn get_footer_decryptor(&self) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        Ok(self.footer_decryptor.clone())
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

    pub(crate) fn get_column_data_decryptor(
        &self,
        column_name: &str,
        key_metadata: Option<&[u8]>,
    ) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        let column_key = self
            .decryption_properties
            .column_key(column_name, key_metadata)?;
        Ok(Arc::new(RingGcmBlockDecryptor::new(&column_key)?))
    }

    pub(crate) fn get_column_metadata_decryptor(
        &self,
        column_name: &str,
        key_metadata: Option<&[u8]>,
    ) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        self.get_column_data_decryptor(column_name, key_metadata)
    }

    pub(crate) fn file_aad(&self) -> &Vec<u8> {
        &self.file_aad
    }
}

pub(crate) fn column_name_from_metadata(metadata: &EncryptionWithColumnKey) -> String {
    metadata.path_in_schema.join(".")
}
