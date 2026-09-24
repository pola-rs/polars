use std::borrow::Cow;
use std::io::Read;
use std::sync::Arc;

use polars_parquet_format::ColumnCryptoMetaData;
use polars_utils::aliases::PlHashMap;

use super::ciphers::{BlockDecryptor, RingGcmBlockDecryptor, TAG_LEN};
use super::modules::{ModuleType, create_footer_aad, create_module_aad};
use crate::parquet::error::ParquetResult;

/// Trait for retrieving an encryption key using the key's metadata
///
/// # Example
///
/// This shows how you might use a `KeyRetriever` to decrypt a Parquet file
/// if you have a set of known encryption keys with identifiers, but at read time
/// you may not know which columns were encrypted and which keys were used.
///
/// In practice, the key metadata might instead store an encrypted key that must
/// be decrypted with a Key Management Server.
///
/// ```
/// # use std::collections::HashMap;
/// # use std::sync::{Arc, Mutex};
/// # use polars_parquet::parquet::encryption::decrypt::{FileDecryptionProperties, KeyRetriever};
/// # use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
/// # use polars_parquet::parquet::error::{ParquetError, ParquetResult};
/// // Define known encryption keys
/// let mut keys = HashMap::new();
/// keys.insert("kf".to_owned(), b"0123456789012345".to_vec());
/// keys.insert("kc1".to_owned(), b"1234567890123450".to_vec());
/// keys.insert("kc2".to_owned(), b"1234567890123451".to_vec());
///
/// // Create encryption properties for writing a file,
/// // and specify the key identifiers as the key metadata.
/// let encryption_properties = FileEncryptionProperties::builder(keys.get("kf").unwrap().clone())
///     .with_footer_key_metadata("kf".into())
///     .with_column_key_and_metadata("x", keys.get("kc1").unwrap().clone(), "kc1".as_bytes().into())
///     .with_column_key_and_metadata("y", keys.get("kc2").unwrap().clone(), "kc2".as_bytes().into())
///     .build()?;
///
/// // Write an encrypted file with the properties
/// // ...
///
/// // Define a KeyRetriever that can get encryption keys using their identifiers
/// struct CustomKeyRetriever {
///     keys: Mutex<HashMap<String, Vec<u8>>>,
/// }
///
/// impl KeyRetriever for CustomKeyRetriever {
///     fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>> {
///         // Metadata is bytes, so convert it to a string identifier
///         let key_metadata = std::str::from_utf8(key_metadata).map_err(|e| {
///             ParquetError::Encryption(format!("Could not convert key metadata to string: {e}"))
///         })?;
///         // Lookup the key
///         let keys = self.keys.lock().unwrap();
///         match keys.get(key_metadata) {
///             Some(key) => Ok(key.clone()),
///             None => Err(ParquetError::Encryption(format!(
///                 "Could not retrieve key for metadata {key_metadata:?}"
///             ))),
///         }
///     }
/// }
///
/// let key_retriever = Arc::new(CustomKeyRetriever {
///     keys: Mutex::new(keys),
/// });
///
/// // Create decryption properties for reading an encrypted file.
/// // Note that we don't need to specify which columns are encrypted,
/// // this is determined by the file metadata, and the required keys will be retrieved
/// // dynamically using our key retriever.
/// let decryption_properties = FileDecryptionProperties::with_key_retriever(key_retriever)
///     .build()?;
///
/// // Read an encrypted file with the decryption properties
/// // ...
///
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
pub trait KeyRetriever: Send + Sync {
    /// Retrieve a decryption key given the key metadata
    fn retrieve_key(&self, key_metadata: &[u8]) -> ParquetResult<Vec<u8>>;
}

pub(crate) fn read_and_decrypt<T: Read>(
    decryptor: &Arc<dyn BlockDecryptor>,
    input: &mut T,
    aad: &[u8],
) -> ParquetResult<Vec<u8>> {
    let mut len_bytes = [0; 4];
    input.read_exact(&mut len_bytes)?;
    let ciphertext_len = u32::from_le_bytes(len_bytes) as usize;
    let mut ciphertext = vec![0; 4 + ciphertext_len];
    input.read_exact(&mut ciphertext[4..])?;

    decryptor.decrypt(&ciphertext, aad.as_ref())
}

// CryptoContext is a data structure that holds the context required to
// decrypt parquet modules (data pages, dictionary pages, etc.).
#[derive(Debug, Clone)]
pub(crate) struct CryptoContext {
    pub(crate) row_group_idx: usize,
    pub(crate) column_ordinal: usize,
    pub(crate) page_ordinal: Option<usize>,
    pub(crate) dictionary_page: bool,
    // We have separate data and metadata decryptors because
    // in GCM CTR mode, the metadata and data pages use
    // different algorithms.
    data_decryptor: Arc<dyn BlockDecryptor>,
    metadata_decryptor: Arc<dyn BlockDecryptor>,
    file_aad: Vec<u8>,
}

impl CryptoContext {
    pub(crate) fn for_column(
        file_decryptor: &FileDecryptor,
        column_crypto_metadata: &ColumnCryptoMetaData,
        row_group_idx: usize,
        column_ordinal: usize,
    ) -> ParquetResult<Self> {
        let (data_decryptor, metadata_decryptor) = match column_crypto_metadata {
            ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(_) => {
                // TODO: In GCM-CTR mode will this need to be a non-GCM decryptor?
                let data_decryptor = file_decryptor.get_footer_decryptor()?;
                let metadata_decryptor = file_decryptor.get_footer_decryptor()?;
                (data_decryptor, metadata_decryptor)
            },
            ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(column_key_encryption) => {
                let key_metadata = &column_key_encryption.key_metadata;
                let full_column_name;
                let column_name = if column_key_encryption.path_in_schema.len() == 1 {
                    &column_key_encryption.path_in_schema[0]
                } else {
                    full_column_name = column_key_encryption.path_in_schema.join(".");
                    &full_column_name
                };
                let data_decryptor = file_decryptor
                    .get_column_data_decryptor(column_name, key_metadata.as_deref())?;
                let metadata_decryptor = file_decryptor
                    .get_column_metadata_decryptor(column_name, key_metadata.as_deref())?;
                (data_decryptor, metadata_decryptor)
            },
        };

        Ok(CryptoContext {
            row_group_idx,
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
            row_group_idx: self.row_group_idx,
            column_ordinal: self.column_ordinal,
            page_ordinal: Some(page_ordinal),
            dictionary_page: false,
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
            self.file_aad(),
            module_type,
            self.row_group_idx,
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
            self.file_aad(),
            module_type,
            self.row_group_idx,
            self.column_ordinal,
            self.page_ordinal,
        )
    }

    pub(crate) fn create_column_index_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            self.file_aad(),
            ModuleType::ColumnIndex,
            self.row_group_idx,
            self.column_ordinal,
            self.page_ordinal,
        )
    }

    pub(crate) fn create_offset_index_aad(&self) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            self.file_aad(),
            ModuleType::OffsetIndex,
            self.row_group_idx,
            self.column_ordinal,
            self.page_ordinal,
        )
    }

    pub(crate) fn for_dictionary_page(&self) -> Self {
        Self {
            row_group_idx: self.row_group_idx,
            column_ordinal: self.column_ordinal,
            page_ordinal: self.page_ordinal,
            dictionary_page: true,
            data_decryptor: self.data_decryptor.clone(),
            metadata_decryptor: self.metadata_decryptor.clone(),
            file_aad: self.file_aad.clone(),
        }
    }

    pub(crate) fn data_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.data_decryptor
    }

    pub(crate) fn metadata_decryptor(&self) -> &Arc<dyn BlockDecryptor> {
        &self.metadata_decryptor
    }

    pub(crate) fn file_aad(&self) -> &Vec<u8> {
        &self.file_aad
    }
}

#[derive(Clone, PartialEq)]
struct ExplicitDecryptionKeys {
    footer_key: Vec<u8>,
    column_keys: PlHashMap<String, Vec<u8>>,
}

#[derive(Clone)]
enum DecryptionKeys {
    Explicit(ExplicitDecryptionKeys),
    ViaRetriever(Arc<dyn KeyRetriever>),
}

impl PartialEq for DecryptionKeys {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DecryptionKeys::Explicit(keys), DecryptionKeys::Explicit(other_keys)) => {
                keys.footer_key == other_keys.footer_key
                    && keys.column_keys == other_keys.column_keys
            },
            (DecryptionKeys::ViaRetriever(_), DecryptionKeys::ViaRetriever(_)) => true,
            _ => false,
        }
    }
}

/// `FileDecryptionProperties` hold keys and AAD data required to decrypt a Parquet file.
///
/// # Examples
///
/// Create `FileDecryptionProperties` for a file encrypted with uniform encryption,
/// where all metadata and data are encrypted with the footer key:
/// ```
/// # use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
/// let file_decryption_properties = FileDecryptionProperties::builder(b"0123456789012345".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
///
/// Create properties for a file where columns are encrypted with different keys:
/// ```
/// # use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
/// let file_decryption_properties = FileDecryptionProperties::builder(b"0123456789012345".into())
///     .with_column_key("x", b"1234567890123450".into())
///     .with_column_key("y", b"1234567890123451".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
///
/// Specify additional authenticated data, used to protect against data replacement.
/// This must match the AAD prefix provided when the file was written, otherwise
/// data decryption will fail.
/// ```
/// # use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
/// let file_decryption_properties = FileDecryptionProperties::builder(b"0123456789012345".into())
///     .with_aad_prefix("example_file".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
#[derive(Clone, PartialEq)]
pub struct FileDecryptionProperties {
    keys: DecryptionKeys,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl FileDecryptionProperties {
    /// Returns a new [`FileDecryptionProperties`] builder that will use the provided key to
    /// decrypt footer metadata.
    pub fn builder(footer_key: Vec<u8>) -> DecryptionPropertiesBuilder {
        DecryptionPropertiesBuilder::new(footer_key)
    }

    /// Returns a new [`FileDecryptionProperties`] builder that uses a [`KeyRetriever`]
    /// to get decryption keys based on key metadata.
    pub fn with_key_retriever(
        key_retriever: Arc<dyn KeyRetriever>,
    ) -> DecryptionPropertiesBuilderWithRetriever {
        DecryptionPropertiesBuilderWithRetriever::new(key_retriever)
    }

    /// AAD prefix string uniquely identifies the file and prevents file swapping
    pub fn aad_prefix(&self) -> Option<&Vec<u8>> {
        self.aad_prefix.as_ref()
    }

    /// Returns true if footer signature verification is enabled for files with plaintext footers.
    pub fn check_plaintext_footer_integrity(&self) -> bool {
        self.footer_signature_verification
    }

    /// Get the encryption key for decrypting a file's footer,
    /// and also column data if uniform encryption is used.
    pub fn footer_key(&self, key_metadata: Option<&[u8]>) -> ParquetResult<Cow<'_, Vec<u8>>> {
        match &self.keys {
            DecryptionKeys::Explicit(keys) => Ok(Cow::Borrowed(&keys.footer_key)),
            DecryptionKeys::ViaRetriever(retriever) => {
                let key = retriever.retrieve_key(key_metadata.unwrap_or_default())?;
                Ok(Cow::Owned(key))
            },
        }
    }

    /// Get the column-specific encryption key for decrypting column data and metadata within a file
    pub fn column_key(
        &self,
        column_name: &str,
        key_metadata: Option<&[u8]>,
    ) -> ParquetResult<Cow<'_, Vec<u8>>> {
        match &self.keys {
            DecryptionKeys::Explicit(keys) => match keys.column_keys.get(column_name) {
                None => Err(encryption_err!(
                    "No column decryption key set for encrypted column '{}'",
                    column_name
                )),
                Some(key) => Ok(Cow::Borrowed(key)),
            },
            DecryptionKeys::ViaRetriever(retriever) => {
                let key = retriever.retrieve_key(key_metadata.unwrap_or_default())?;
                Ok(Cow::Owned(key))
            },
        }
    }

    /// Get the column names and associated decryption keys that have been configured.
    /// If a key retriever is used rather than explicit decryption keys, the result
    /// will be empty.
    /// Provided for testing consumer code.
    pub fn column_keys(&self) -> (Vec<String>, Vec<Vec<u8>>) {
        let mut column_names: Vec<String> = Vec::new();
        let mut column_keys: Vec<Vec<u8>> = Vec::new();
        if let DecryptionKeys::Explicit(keys) = &self.keys {
            for (key, value) in &keys.column_keys {
                column_names.push(key.clone());
                column_keys.push(value.clone());
            }
        }
        (column_names, column_keys)
    }

    /// Whether these decryption properties use a key retriever.
    /// When false, explicit keys were provided up front and can
    /// be retrieved without providing key metadata, rather than
    /// resolved on demand.
    pub fn uses_key_retriever(&self) -> bool {
        matches!(self.keys, DecryptionKeys::ViaRetriever(_))
    }
}

impl std::fmt::Debug for FileDecryptionProperties {
    // Don't output keys.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("FileDecryptionProperties");
        match &self.keys {
            DecryptionKeys::Explicit(keys) => {
                let mut column_names: Vec<&str> =
                    keys.column_keys.keys().map(String::as_str).collect();
                column_names.sort_unstable();
                s.field("column_key_names", &column_names)
            },
            DecryptionKeys::ViaRetriever(_) => s.field("key_retriever", &".."),
        };
        s.field("aad_prefix", &self.aad_prefix)
            .field(
                "footer_signature_verification",
                &self.footer_signature_verification,
            )
            .finish()
    }
}

/// Builder for [`FileDecryptionProperties`]
///
/// See [`FileDecryptionProperties`] for example usage.
pub struct DecryptionPropertiesBuilder {
    footer_key: Vec<u8>,
    column_keys: PlHashMap<String, Vec<u8>>,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl DecryptionPropertiesBuilder {
    /// Create a new [`DecryptionPropertiesBuilder`] builder that will use the provided key to
    /// decrypt footer metadata.
    pub fn new(footer_key: Vec<u8>) -> DecryptionPropertiesBuilder {
        Self {
            footer_key,
            column_keys: PlHashMap::default(),
            aad_prefix: None,
            footer_signature_verification: true,
        }
    }

    /// Finalize the builder and return created [`FileDecryptionProperties`]
    pub fn build(self) -> ParquetResult<Arc<FileDecryptionProperties>> {
        let keys = DecryptionKeys::Explicit(ExplicitDecryptionKeys {
            footer_key: self.footer_key,
            column_keys: self.column_keys,
        });
        Ok(Arc::new(FileDecryptionProperties {
            keys,
            aad_prefix: self.aad_prefix,
            footer_signature_verification: self.footer_signature_verification,
        }))
    }

    /// Specify the expected AAD prefix to be used for decryption.
    /// This must be set if the file was written with an AAD prefix and the
    /// prefix is not stored in the file metadata.
    pub fn with_aad_prefix(mut self, value: Vec<u8>) -> Self {
        self.aad_prefix = Some(value);
        self
    }

    /// Specify the decryption key to use for a column.
    /// The column name is the column's dot-separated path in the Parquet schema,
    /// e.g. `a.b.c` for a nested column.
    pub fn with_column_key(mut self, column_name: &str, decryption_key: Vec<u8>) -> Self {
        self.column_keys
            .insert(column_name.to_string(), decryption_key);
        self
    }

    /// Specify multiple column decryption keys.
    /// Column names are dot-separated paths in the Parquet schema,
    /// e.g. `a.b.c` for a nested column.
    pub fn with_column_keys(
        mut self,
        column_names: Vec<&str>,
        keys: Vec<Vec<u8>>,
    ) -> ParquetResult<Self> {
        if column_names.len() != keys.len() {
            return Err(encryption_err!(
                "The number of column names ({}) does not match the number of keys ({})",
                column_names.len(),
                keys.len()
            ));
        }
        for (column_name, key) in column_names.into_iter().zip(keys) {
            self.column_keys.insert(column_name.to_string(), key);
        }
        Ok(self)
    }

    /// Disable verification of footer tags for files that use plaintext footers.
    /// Signature verification is enabled by default.
    pub fn disable_footer_signature_verification(mut self) -> Self {
        self.footer_signature_verification = false;
        self
    }
}

/// Builder for [`FileDecryptionProperties`] that uses a [`KeyRetriever`]
///
/// See the [`KeyRetriever`] documentation for example usage.
pub struct DecryptionPropertiesBuilderWithRetriever {
    key_retriever: Arc<dyn KeyRetriever>,
    aad_prefix: Option<Vec<u8>>,
    footer_signature_verification: bool,
}

impl DecryptionPropertiesBuilderWithRetriever {
    /// Create a new [`DecryptionPropertiesBuilderWithRetriever`] by providing a [`KeyRetriever`] that
    /// can be used to get decryption keys based on key metadata.
    pub fn new(key_retriever: Arc<dyn KeyRetriever>) -> DecryptionPropertiesBuilderWithRetriever {
        Self {
            key_retriever,
            aad_prefix: None,
            footer_signature_verification: true,
        }
    }

    /// Finalize the builder and return created [`FileDecryptionProperties`]
    pub fn build(self) -> ParquetResult<Arc<FileDecryptionProperties>> {
        let keys = DecryptionKeys::ViaRetriever(self.key_retriever);
        Ok(Arc::new(FileDecryptionProperties {
            keys,
            aad_prefix: self.aad_prefix,
            footer_signature_verification: self.footer_signature_verification,
        }))
    }

    /// Specify the expected AAD prefix to be used for decryption.
    /// This must be set if the file was written with an AAD prefix and the
    /// prefix is not stored in the file metadata.
    pub fn with_aad_prefix(mut self, value: Vec<u8>) -> Self {
        self.aad_prefix = Some(value);
        self
    }

    /// Disable verification of footer tags for files that use plaintext footers.
    /// Signature verification is enabled by default.
    pub fn disable_footer_signature_verification(mut self) -> Self {
        self.footer_signature_verification = false;
        self
    }
}

#[derive(Clone, Debug)]
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
        let footer_decryptor = RingGcmBlockDecryptor::new(&footer_key).map_err(|e| {
            encryption_err!(
                "Invalid footer key. {}",
                e.to_string().replace("Parquet error: ", "")
            )
        })?;

        Ok(Self {
            footer_decryptor: Arc::new(footer_decryptor),
            decryption_properties: Arc::clone(decryption_properties),
            file_aad,
        })
    }

    pub(crate) fn get_footer_decryptor(&self) -> ParquetResult<Arc<dyn BlockDecryptor>> {
        Ok(self.footer_decryptor.clone())
    }

    /// Verify the signature of the footer
    pub(crate) fn verify_plaintext_footer_signature(
        &self,
        plaintext_footer: &[u8],
    ) -> ParquetResult<()> {
        // Plaintext footer format is: [plaintext metadata, nonce, authentication tag]
        let tag = &plaintext_footer[plaintext_footer.len() - TAG_LEN..];
        let aad = create_footer_aad(self.file_aad())?;
        let footer_decryptor = self.get_footer_decryptor()?;

        let computed_tag = footer_decryptor.compute_plaintext_tag(&aad, plaintext_footer)?;

        if computed_tag != tag {
            return Err(encryption_err!(
                "Footer signature verification failed. Computed: {:?}, Expected: {:?}",
                computed_tag,
                tag
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
        // Once GCM CTR mode is implemented, data and metadata decryptors may be different
        self.get_column_data_decryptor(column_name, key_metadata)
    }

    pub(crate) fn file_aad(&self) -> &Vec<u8> {
        &self.file_aad
    }
}
