use std::borrow::Cow;
use std::sync::Arc;

use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use polars_parquet_format::{
    AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
    EncryptionWithFooterKey, FileCryptoMetaData,
};
use polars_utils::aliases::PlHashMap;

use super::ciphers::{
    AesGcmBlockEncryptor, BlockEncryptor, NONCE_LEN, SIZE_LEN, TAG_LEN, aes_gcm_algorithm,
};
use super::modules::{ModuleType, create_footer_aad, create_module_aad};
use crate::parquet::error::ParquetResult;
use crate::parquet::metadata::{ColumnDescriptor, SchemaDescriptor};
use crate::parquet::page::ParquetPageHeader;
use crate::parquet::write::WriteThrift;

#[derive(Clone, PartialEq)]
struct EncryptionKey {
    key: Vec<u8>,
    key_metadata: Option<Vec<u8>>,
}

impl std::fmt::Debug for EncryptionKey {
    // Don't output the key itself.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionKey")
            .field("key_metadata", &self.key_metadata)
            .finish_non_exhaustive()
    }
}

impl EncryptionKey {
    fn new(key: Vec<u8>) -> EncryptionKey {
        Self {
            key,
            key_metadata: None,
        }
    }

    fn with_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(metadata);
        self
    }

    fn key(&self) -> &Vec<u8> {
        &self.key
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Defines how data in a Parquet file should be encrypted
///
/// # Examples
///
/// Create `FileEncryptionProperties` for a file encrypted with uniform encryption,
/// where all metadata and data are encrypted with the footer key:
/// ```
/// # use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
/// let file_encryption_properties = FileEncryptionProperties::builder(b"0123456789012345".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
///
/// Create properties for a file where columns are encrypted with different keys.
/// Any columns without a key specified will be unencrypted:
/// ```
/// # use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
/// let file_encryption_properties = FileEncryptionProperties::builder(b"0123456789012345".into())
///     .with_column_key("x", b"1234567890123450".into())
///     .with_column_key("y", b"1234567890123451".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
///
/// Specify additional authenticated data, used to protect against data replacement.
/// This should represent the file identity:
/// ```
/// # use polars_parquet::parquet::encryption::encrypt::FileEncryptionProperties;
/// let file_encryption_properties = FileEncryptionProperties::builder(b"0123456789012345".into())
///     .with_aad_prefix("example_file".into())
///     .build()?;
/// # Ok::<(), polars_parquet::parquet::error::ParquetError>(())
/// ```
pub struct FileEncryptionProperties {
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: PlHashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl FileEncryptionProperties {
    /// Create a new builder for encryption properties with the given footer encryption key
    pub fn builder(footer_key: Vec<u8>) -> EncryptionPropertiesBuilder {
        EncryptionPropertiesBuilder::new(footer_key)
    }

    /// Should the footer be encrypted
    pub fn encrypt_footer(&self) -> bool {
        self.encrypt_footer
    }

    /// Retrieval metadata of key used for encryption of footer and (possibly) columns
    pub fn footer_key_metadata(&self) -> Option<&Vec<u8>> {
        self.footer_key.key_metadata.as_ref()
    }

    /// AAD prefix used to protect against swapping files written with different prefixes
    pub fn aad_prefix(&self) -> Option<&Vec<u8>> {
        self.aad_prefix.as_ref()
    }

    /// Should the AAD prefix be stored in the file
    pub fn store_aad_prefix(&self) -> bool {
        self.store_aad_prefix && self.aad_prefix.is_some()
    }

    /// The names of the columns encrypted with column keys
    pub fn encrypted_column_names(&self) -> impl Iterator<Item = &str> {
        self.column_keys.keys().map(String::as_str)
    }

    /// Checks that each column that is to be encrypted is present in the schema exactly once
    pub fn validate_encrypted_column_names(&self, schema: &SchemaDescriptor) -> ParquetResult<()> {
        // Different columns can have the same dot-separated path, e.g. a top-level column
        // named `a.b` and field `b` of struct column `a`.
        let mut column_path_counts = PlHashMap::<Cow<'_, str>, usize>::default();
        for column in schema.columns() {
            *column_path_counts
                .entry(column_path_string(column))
                .or_default() += 1;
        }

        let mut columns_missing_in_schema = self
            .column_keys
            .keys()
            .filter(|column| !column_path_counts.contains_key(column.as_str()))
            .map(String::as_str)
            .collect::<Vec<_>>();
        if !columns_missing_in_schema.is_empty() {
            columns_missing_in_schema.sort();
            return Err(encryption_err!(
                "The following columns with encryption keys specified were not found in the schema: {}",
                columns_missing_in_schema.join(", ")
            ));
        }

        let mut ambiguous_columns = self
            .column_keys
            .keys()
            .filter(|column| column_path_counts[column.as_str()] > 1)
            .map(String::as_str)
            .collect::<Vec<_>>();
        if !ambiguous_columns.is_empty() {
            ambiguous_columns.sort();
            return Err(encryption_err!(
                "The following columns with encryption keys specified match multiple columns \
                with the same dot-separated path, so can't be encrypted separately: {}",
                ambiguous_columns.join(", ")
            ));
        }
        Ok(())
    }
}

/// Builder for [`FileEncryptionProperties`]
///
/// See [`FileEncryptionProperties`] for example usage.
pub struct EncryptionPropertiesBuilder {
    encrypt_footer: bool,
    footer_key: EncryptionKey,
    column_keys: PlHashMap<String, EncryptionKey>,
    aad_prefix: Option<Vec<u8>>,
    store_aad_prefix: bool,
}

impl EncryptionPropertiesBuilder {
    /// Create a new [`EncryptionPropertiesBuilder`] with the given footer encryption key
    pub fn new(footer_key: Vec<u8>) -> EncryptionPropertiesBuilder {
        Self {
            footer_key: EncryptionKey::new(footer_key),
            column_keys: PlHashMap::default(),
            aad_prefix: None,
            encrypt_footer: true,
            store_aad_prefix: false,
        }
    }

    /// Set if the footer should be stored in plaintext (not encrypted). Defaults to false.
    pub fn with_plaintext_footer(mut self, plaintext_footer: bool) -> Self {
        self.encrypt_footer = !plaintext_footer;
        self
    }

    /// Set retrieval metadata of key used for encryption of footer and (possibly) columns
    pub fn with_footer_key_metadata(mut self, metadata: Vec<u8>) -> Self {
        self.footer_key = self.footer_key.with_metadata(metadata);
        self
    }

    /// Set the key used for encryption of a column. Note that if no column keys are configured then
    /// all columns will be encrypted with the footer key.
    /// If any column keys are configured then only the columns with a key will be encrypted.
    /// The column name is the column's dot-separated path in the Parquet schema,
    /// e.g. `a.b.c` for a nested column. If the path matches more than one column, for example
    /// a top-level column named `a.b` and field `b` of struct column `a`, writing will fail.
    pub fn with_column_key(mut self, column_name: &str, key: Vec<u8>) -> Self {
        self.column_keys
            .insert(column_name.to_string(), EncryptionKey::new(key));
        self
    }

    /// Set the key used for encryption of a column and its metadata. The Key's metadata field is to
    /// enable file readers to recover the key. For example, the metadata can keep a serialized
    /// ID of a data key. Note that if no column keys are configured then all columns
    /// will be encrypted with the footer key. If any column keys are configured then only the
    /// columns with a key will be encrypted.
    /// The column name is the column's dot-separated path in the Parquet schema,
    /// e.g. `a.b.c` for a nested column.
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

    /// Set the AAD (additional authenticated data) prefix. Readers that provide an AAD prefix
    /// can only read files written with the same prefix, which protects against a file being
    /// swapped for one written with a different prefix, e.g. an older version of the file or
    /// a different partition file in the same data set (table). It does not protect against
    /// swapping files that were written with the same prefix. When not specified, no AAD
    /// prefix is used.
    pub fn with_aad_prefix(mut self, aad_prefix: Vec<u8>) -> Self {
        self.aad_prefix = Some(aad_prefix);
        self
    }

    /// Should the AAD prefix be stored in the file. If false, readers will need to provide the
    /// AAD prefix to be able to decrypt data. Defaults to false.
    pub fn with_aad_prefix_storage(mut self, store_aad_prefix: bool) -> Self {
        self.store_aad_prefix = store_aad_prefix;
        self
    }

    /// Build the encryption properties.
    ///
    /// Returns an error if any key has an unsupported length. Keys must be 16 or 32 bytes,
    /// for AES-128 and AES-256 respectively.
    pub fn build(self) -> ParquetResult<Arc<FileEncryptionProperties>> {
        aes_gcm_algorithm(self.footer_key.key(), "footer key")?;
        for (column_name, column_key) in &self.column_keys {
            aes_gcm_algorithm(column_key.key(), &format!("key for column '{column_name}'"))?;
        }
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
/// The encryption configuration for a single Parquet file
pub(crate) struct FileEncryptor {
    properties: Arc<FileEncryptionProperties>,
    aad_file_unique: Vec<u8>,
    file_aad: Vec<u8>,
    /// Encryptor for the footer, and for all columns under uniform encryption.
    footer_encryptor: AesGcmBlockEncryptor,
    /// Encryptors for columns with their own key, keyed by the column path.
    column_encryptors: PlHashMap<String, AesGcmBlockEncryptor>,
}

impl FileEncryptor {
    /// Create a [`FileEncryptor`] from encryption properties and verifies keys are valid.
    pub(crate) fn new(properties: Arc<FileEncryptionProperties>) -> ParquetResult<Self> {
        // Generate unique AAD for file
        let rng = SystemRandom::new();
        let mut aad_file_unique = vec![0u8; 8];
        rng.fill(&mut aad_file_unique)?;

        let file_aad = match properties.aad_prefix.as_ref() {
            None => aad_file_unique.clone(),
            Some(aad_prefix) => [aad_prefix.clone(), aad_file_unique.clone()].concat(),
        };

        let footer_encryptor = AesGcmBlockEncryptor::new(properties.footer_key.key())?;
        let column_encryptors = properties
            .column_keys
            .iter()
            .map(|(column_path, column_key)| {
                let encryptor = AesGcmBlockEncryptor::new(column_key.key())?;
                Ok((column_path.clone(), encryptor))
            })
            .collect::<ParquetResult<PlHashMap<_, _>>>()?;

        Ok(Self {
            properties,
            aad_file_unique,
            file_aad,
            footer_encryptor,
            column_encryptors,
        })
    }

    /// Get the encryptor's file encryption properties
    pub fn properties(&self) -> &Arc<FileEncryptionProperties> {
        &self.properties
    }

    /// Returns whether data for the specified column should be encrypted
    pub fn is_column_encrypted(&self, column_path: &str) -> bool {
        if self.properties.column_keys.is_empty() {
            // Uniform encryption
            true
        } else {
            self.properties.column_keys.contains_key(column_path)
        }
    }

    /// Get the block encryptor for a column.
    /// Will return an error if the column is not an encrypted column.
    ///
    /// This takes the fields it needs rather than `&mut self`, so that callers can borrow
    /// other fields at the same time.
    fn column_block_encryptor<'a>(
        properties: &FileEncryptionProperties,
        footer_encryptor: &'a mut AesGcmBlockEncryptor,
        column_encryptors: &'a mut PlHashMap<String, AesGcmBlockEncryptor>,
        column_path: &str,
    ) -> ParquetResult<&'a mut AesGcmBlockEncryptor> {
        if properties.column_keys.is_empty() {
            // Uniform encryption
            return Ok(footer_encryptor);
        }
        column_encryptors
            .get_mut(column_path)
            .ok_or_else(|| encryption_err!("Column '{}' is not encrypted", column_path))
    }

    /// Create a [`FileEncryptor`] for writing a file with the given schema.
    ///
    /// This checks that all columns with keys are in the schema, and that all keys are
    /// valid, so that errors are raised before any data is written.
    pub(crate) fn try_new_for_schema(
        properties: Arc<FileEncryptionProperties>,
        schema: &SchemaDescriptor,
    ) -> ParquetResult<Self> {
        properties.validate_encrypted_column_names(schema)?;
        Self::new(properties)
    }

    /// The encryption algorithm to store in the file.
    pub(crate) fn encryption_algorithm(&self) -> EncryptionAlgorithm {
        let properties = &self.properties;
        let supply_aad_prefix = properties
            .aad_prefix()
            .map(|_| !properties.store_aad_prefix());
        let aad_prefix = if properties.store_aad_prefix() {
            properties.aad_prefix().cloned()
        } else {
            None
        };
        EncryptionAlgorithm::AESGCMV1(AesGcmV1::new(
            aad_prefix,
            self.aad_file_unique.clone(),
            supply_aad_prefix,
        ))
    }

    /// The crypto metadata stored before an encrypted footer.
    pub(crate) fn file_crypto_metadata(&self) -> FileCryptoMetaData {
        FileCryptoMetaData::new(
            self.encryption_algorithm(),
            self.properties.footer_key_metadata().cloned(),
        )
    }

    /// Encrypt a serialized footer.
    pub(crate) fn encrypt_footer(&mut self, footer: &[u8]) -> ParquetResult<Vec<u8>> {
        let aad = create_footer_aad(&self.file_aad)?;
        self.footer_encryptor.encrypt(footer, &aad)
    }

    /// Compute the signature of a serialized plaintext footer, which is the nonce and
    /// authentication tag from encrypting the footer.
    pub(crate) fn sign_footer(&mut self, footer: &[u8]) -> ParquetResult<Vec<u8>> {
        let encrypted = self.encrypt_footer(footer)?;
        let nonce = &encrypted[SIZE_LEN..SIZE_LEN + NONCE_LEN];
        let tag = &encrypted[encrypted.len() - TAG_LEN..];
        Ok([nonce, tag].concat())
    }

    /// Encrypt a serialized module of an encrypted column, such as its column metadata,
    /// column index or offset index.
    pub(crate) fn encrypt_column_module(
        &mut self,
        column: &ColumnDescriptor,
        module_type: ModuleType,
        row_group_idx: usize,
        column_ordinal: usize,
        plaintext: &[u8],
    ) -> ParquetResult<Vec<u8>> {
        let aad = create_module_aad(
            &self.file_aad,
            module_type,
            row_group_idx,
            column_ordinal,
            None,
        )?;
        let encryptor = Self::column_block_encryptor(
            &self.properties,
            &mut self.footer_encryptor,
            &mut self.column_encryptors,
            &column_path_string(column),
        )?;
        encryptor.encrypt(plaintext, &aad)
    }

    /// Whether data for the column should be encrypted
    pub(crate) fn is_column_descriptor_encrypted(&self, column: &ColumnDescriptor) -> bool {
        self.is_column_encrypted(&column_path_string(column))
    }

    /// Get a [`PageEncryptor`] for writing the pages of a column chunk, or `None` if the
    /// column isn't encrypted.
    pub(crate) fn page_encryptor(
        &mut self,
        column: &ColumnDescriptor,
        row_group_idx: usize,
        column_ordinal: usize,
    ) -> ParquetResult<Option<PageEncryptor<'_>>> {
        let column_path = column_path_string(column);
        if !self.is_column_encrypted(&column_path) {
            return Ok(None);
        }
        Ok(Some(PageEncryptor {
            file_aad: &self.file_aad,
            block_encryptor: Self::column_block_encryptor(
                &self.properties,
                &mut self.footer_encryptor,
                &mut self.column_encryptors,
                &column_path,
            )?,
            row_group_idx,
            column_ordinal,
            page_ordinal: 0,
        }))
    }
}

/// Encrypts the pages and page headers of a single column chunk.
pub(crate) struct PageEncryptor<'a> {
    file_aad: &'a [u8],
    block_encryptor: &'a mut AesGcmBlockEncryptor,
    row_group_idx: usize,
    column_ordinal: usize,
    /// Ordinal of the current data page. Dictionary pages don't have an ordinal.
    page_ordinal: usize,
}

impl PageEncryptor<'_> {
    /// Encrypt the (possibly compressed) data of a page.
    pub(crate) fn encrypt_page(
        &mut self,
        page: &[u8],
        is_dictionary: bool,
    ) -> ParquetResult<Vec<u8>> {
        let module_type = if is_dictionary {
            ModuleType::DictionaryPage
        } else {
            ModuleType::DataPage
        };
        let aad = self.create_aad(module_type)?;
        self.block_encryptor.encrypt(page, &aad)
    }

    /// Serialize and encrypt a page header.
    pub(crate) fn encrypt_page_header(
        &mut self,
        header: &ParquetPageHeader,
        is_dictionary: bool,
    ) -> ParquetResult<Vec<u8>> {
        let module_type = if is_dictionary {
            ModuleType::DictionaryPageHeader
        } else {
            ModuleType::DataPageHeader
        };
        let aad = self.create_aad(module_type)?;
        self.block_encryptor
            .encrypt(&header.to_thrift_bytes()?, &aad)
    }

    /// Move to the next data page. Must be called after writing each data page.
    pub(crate) fn increment_page(&mut self) {
        self.page_ordinal += 1;
    }

    fn create_aad(&self, module_type: ModuleType) -> ParquetResult<Vec<u8>> {
        create_module_aad(
            self.file_aad,
            module_type,
            self.row_group_idx,
            self.column_ordinal,
            Some(self.page_ordinal),
        )
    }
}

/// Get the crypto metadata for a column from the file encryption properties
pub(crate) fn get_column_crypto_metadata(
    properties: &Arc<FileEncryptionProperties>,
    column: &ColumnDescriptor,
) -> Option<ColumnCryptoMetaData> {
    if properties.column_keys.is_empty() {
        // Uniform encryption
        Some(ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(
            EncryptionWithFooterKey {},
        ))
    } else {
        properties
            .column_keys
            .get(column_path_string(column).as_ref())
            .map(|encryption_key| {
                // Column is encrypted with a column specific key
                ColumnCryptoMetaData::ENCRYPTIONWITHCOLUMNKEY(EncryptionWithColumnKey {
                    path_in_schema: column
                        .path_in_schema
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    key_metadata: encryption_key.key_metadata.clone(),
                })
            })
    }
}

/// The dot-separated path of a column, as used to identify columns in encryption properties.
fn column_path_string(column: &ColumnDescriptor) -> Cow<'_, str> {
    match column.path_in_schema.as_slice() {
        [name] => Cow::Borrowed(name.as_str()),
        path => Cow::Owned(
            path.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("."),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parquet::encryption::ciphers::{AesGcmBlockDecryptor, BlockDecryptor};

    const FOOTER_KEY: &[u8] = b"0123456789012345";

    fn aes_gcm_v1(builder: EncryptionPropertiesBuilder) -> (AesGcmV1, Vec<u8>) {
        let encryptor = FileEncryptor::new(builder.build().unwrap()).unwrap();
        let EncryptionAlgorithm::AESGCMV1(algorithm) = encryptor.encryption_algorithm() else {
            panic!("expected AES_GCM_V1");
        };
        (algorithm, encryptor.aad_file_unique)
    }

    #[test]
    fn test_encryption_algorithm() {
        let builder = FileEncryptionProperties::builder(FOOTER_KEY.to_vec());
        let (algorithm, aad_file_unique) = aes_gcm_v1(builder);
        assert_eq!(algorithm.aad_file_unique, Some(aad_file_unique));
        assert_eq!(algorithm.aad_prefix, None);
        assert_eq!(algorithm.supply_aad_prefix, None);

        let builder = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_aad_prefix(b"prefix".to_vec());
        let (algorithm, _) = aes_gcm_v1(builder);
        assert_eq!(algorithm.aad_prefix, None);
        assert_eq!(algorithm.supply_aad_prefix, Some(true));

        let builder = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_aad_prefix(b"prefix".to_vec())
            .with_aad_prefix_storage(true);
        let (algorithm, _) = aes_gcm_v1(builder);
        assert_eq!(algorithm.aad_prefix, Some(b"prefix".to_vec()));
        assert_eq!(algorithm.supply_aad_prefix, Some(false));
    }

    #[test]
    fn test_build_validates_key_lengths() {
        let err = FileEncryptionProperties::builder(b"short_key".to_vec())
            .build()
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("footer key: unsupported key length: 9"),
            "{err}"
        );

        let err = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_column_key("x", b"short_key".to_vec())
            .build()
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("key for column 'x': unsupported key length: 9"),
            "{err}"
        );

        // AES-256 keys are also accepted
        FileEncryptionProperties::builder(vec![0u8; 32])
            .with_column_key("x", vec![1u8; 32])
            .build()
            .unwrap();
    }

    #[test]
    fn test_unique_aad_per_file() {
        let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .build()
            .unwrap();
        let a = FileEncryptor::new(properties.clone()).unwrap();
        let b = FileEncryptor::new(properties).unwrap();
        assert_ne!(a.aad_file_unique, b.aad_file_unique);
    }

    #[test]
    fn test_footer_signature() {
        let properties = FileEncryptionProperties::builder(FOOTER_KEY.to_vec())
            .with_plaintext_footer(true)
            .build()
            .unwrap();
        let mut encryptor = FileEncryptor::new(properties).unwrap();
        let footer = b"plaintext footer".to_vec();
        let signature = encryptor.sign_footer(&footer).unwrap();
        assert_eq!(signature.len(), NONCE_LEN + TAG_LEN);

        // The reader verifies the signature by recomputing the tag from the nonce
        let decryptor = AesGcmBlockDecryptor::new(FOOTER_KEY).unwrap();
        let aad = create_footer_aad(&encryptor.file_aad).unwrap();
        let signed_footer = [footer.as_slice(), &signature].concat();
        let tag = decryptor
            .compute_plaintext_tag(&aad, &signed_footer)
            .unwrap();
        assert_eq!(tag, signature[NONCE_LEN..]);
    }
}
