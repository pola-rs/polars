use std::borrow::Cow;
use std::sync::Arc;

use aws_lc_rs::rand::{SecureRandom, SystemRandom};
use polars_parquet_format::{
    AesGcmV1, ColumnCryptoMetaData, EncryptionAlgorithm, EncryptionWithColumnKey,
    EncryptionWithFooterKey, FileCryptoMetaData,
};
use polars_utils::aliases::{PlHashMap, PlHashSet};

use super::ciphers::{AesGcmBlockEncryptor, BlockEncryptor, NONCE_LEN, SIZE_LEN, TAG_LEN};
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

    /// Retrieval of key used for encryption of footer and (possibly) columns
    pub fn footer_key(&self) -> &Vec<u8> {
        &self.footer_key.key
    }

    /// Get the column names, keys, and metadata for columns to be encrypted
    pub fn column_keys(&self) -> (Vec<String>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let mut column_names: Vec<String> = Vec::with_capacity(self.column_keys.len());
        let mut keys: Vec<Vec<u8>> = Vec::with_capacity(self.column_keys.len());
        let mut meta: Vec<Vec<u8>> = Vec::with_capacity(self.column_keys.len());
        for (key, value) in &self.column_keys {
            column_names.push(key.clone());
            keys.push(value.key.clone());
            if let Some(metadata) = value.key_metadata.as_ref() {
                meta.push(metadata.clone());
            }
        }
        (column_names, keys, meta)
    }

    /// AAD prefix string uniquely identifies the file and prevents file swapping
    pub fn aad_prefix(&self) -> Option<&Vec<u8>> {
        self.aad_prefix.as_ref()
    }

    /// Should the AAD prefix be stored in the file
    pub fn store_aad_prefix(&self) -> bool {
        self.store_aad_prefix && self.aad_prefix.is_some()
    }

    /// Checks if columns that are to be encrypted are present in schema
    pub(crate) fn validate_encrypted_column_names(
        &self,
        schema: &SchemaDescriptor,
    ) -> ParquetResult<()> {
        let column_paths = schema
            .columns()
            .iter()
            .map(|column| column_path_string(column).into_owned())
            .collect::<PlHashSet<_>>();
        let encryption_columns = self
            .column_keys
            .keys()
            .cloned()
            .collect::<PlHashSet<String>>();
        if !encryption_columns.is_subset(&column_paths) {
            let mut columns_missing_in_schema = encryption_columns
                .difference(&column_paths)
                .cloned()
                .collect::<Vec<String>>();
            columns_missing_in_schema.sort();
            return Err(encryption_err!(
                "The following columns with encryption keys specified were not found in the schema: {}",
                columns_missing_in_schema.join(", ")
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
    /// e.g. `a.b.c` for a nested column.
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

    /// The AAD prefix uniquely identifies the file and allows to differentiate it e.g. from
    /// older versions of the file or from other partition files in the same data set (table).
    /// These bytes are optionally passed by a writer upon file creation. When not specified, no
    /// AAD prefix is used.
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

    /// Build the encryption properties
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
/// The encryption configuration for a single Parquet file
pub(crate) struct FileEncryptor {
    properties: Arc<FileEncryptionProperties>,
    aad_file_unique: Vec<u8>,
    file_aad: Vec<u8>,
}

impl FileEncryptor {
    pub(crate) fn new(properties: Arc<FileEncryptionProperties>) -> ParquetResult<Self> {
        // Generate unique AAD for file
        let rng = SystemRandom::new();
        let mut aad_file_unique = vec![0u8; 8];
        rng.fill(&mut aad_file_unique)?;

        let file_aad = match properties.aad_prefix.as_ref() {
            None => aad_file_unique.clone(),
            Some(aad_prefix) => [aad_prefix.clone(), aad_file_unique.clone()].concat(),
        };

        Ok(Self {
            properties,
            aad_file_unique,
            file_aad,
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

    /// Get the BlockEncryptor for the footer
    pub(crate) fn get_footer_encryptor(&self) -> ParquetResult<Box<dyn BlockEncryptor>> {
        Ok(Box::new(AesGcmBlockEncryptor::new(
            &self.properties.footer_key.key,
        )?))
    }

    /// Get the encryptor for a column.
    /// Will return an error if the column is not an encrypted column.
    pub(crate) fn get_column_encryptor(
        &self,
        column_path: &str,
    ) -> ParquetResult<Box<dyn BlockEncryptor>> {
        if self.properties.column_keys.is_empty() {
            return self.get_footer_encryptor();
        }
        match self.properties.column_keys.get(column_path) {
            None => Err(encryption_err!("Column '{}' is not encrypted", column_path)),
            Some(column_key) => Ok(Box::new(AesGcmBlockEncryptor::new(column_key.key())?)),
        }
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
        let encryptor = Self::new(properties)?;
        encryptor.get_footer_encryptor()?;
        for column_path in encryptor.properties.column_keys.keys() {
            encryptor.get_column_encryptor(column_path)?;
        }
        Ok(encryptor)
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
    pub(crate) fn encrypt_footer(&self, footer: &[u8]) -> ParquetResult<Vec<u8>> {
        let aad = create_footer_aad(&self.file_aad)?;
        self.get_footer_encryptor()?.encrypt(footer, &aad)
    }

    /// Compute the signature of a serialized plaintext footer, which is the nonce and
    /// authentication tag from encrypting the footer.
    pub(crate) fn sign_footer(&self, footer: &[u8]) -> ParquetResult<Vec<u8>> {
        let encrypted = self.encrypt_footer(footer)?;
        let nonce = &encrypted[SIZE_LEN..SIZE_LEN + NONCE_LEN];
        let tag = &encrypted[encrypted.len() - TAG_LEN..];
        Ok([nonce, tag].concat())
    }

    /// Encrypt the serialized metadata of an encrypted column.
    pub(crate) fn encrypt_column_metadata(
        &self,
        column: &ColumnDescriptor,
        row_group_idx: usize,
        column_ordinal: usize,
        metadata: &[u8],
    ) -> ParquetResult<Vec<u8>> {
        self.encrypt_column_module(
            column,
            ModuleType::ColumnMetaData,
            row_group_idx,
            column_ordinal,
            metadata,
        )
    }

    /// Encrypt the serialized column index of an encrypted column.
    pub(crate) fn encrypt_column_index(
        &self,
        column: &ColumnDescriptor,
        row_group_idx: usize,
        column_ordinal: usize,
        index: &[u8],
    ) -> ParquetResult<Vec<u8>> {
        self.encrypt_column_module(
            column,
            ModuleType::ColumnIndex,
            row_group_idx,
            column_ordinal,
            index,
        )
    }

    /// Encrypt the serialized offset index of an encrypted column.
    pub(crate) fn encrypt_offset_index(
        &self,
        column: &ColumnDescriptor,
        row_group_idx: usize,
        column_ordinal: usize,
        index: &[u8],
    ) -> ParquetResult<Vec<u8>> {
        self.encrypt_column_module(
            column,
            ModuleType::OffsetIndex,
            row_group_idx,
            column_ordinal,
            index,
        )
    }

    fn encrypt_column_module(
        &self,
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
        self.get_column_encryptor(&column_path_string(column))?
            .encrypt(plaintext, &aad)
    }

    /// Whether data for the column should be encrypted
    pub(crate) fn is_column_descriptor_encrypted(&self, column: &ColumnDescriptor) -> bool {
        self.is_column_encrypted(&column_path_string(column))
    }

    /// Get a [`PageEncryptor`] for writing the pages of a column chunk, or `None` if the
    /// column isn't encrypted.
    pub(crate) fn page_encryptor(
        &self,
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
            block_encryptor: self.get_column_encryptor(&column_path)?,
            row_group_idx,
            column_ordinal,
            page_ordinal: 0,
        }))
    }
}

/// Encrypts the pages and page headers of a single column chunk.
pub(crate) struct PageEncryptor<'a> {
    file_aad: &'a [u8],
    block_encryptor: Box<dyn BlockEncryptor>,
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
        let encryptor = FileEncryptor::new(properties).unwrap();
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
