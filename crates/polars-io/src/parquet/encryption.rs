//! Configuration for reading and writing files encrypted with Parquet modular encryption.

#[cfg(feature = "parquet_kms")]
pub mod kms;

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use polars_error::{PolarsResult, polars_bail, polars_err};
use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
use polars_parquet::parquet::encryption::encrypt::{
    FileEncryptionProperties, validate_encrypted_column_names,
};
use polars_parquet::parquet::metadata::SchemaDescriptor;
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_path::PlRefPath;
use polars_utils::unique_id::UniqueId;

/// Creates the properties used to encrypt each Parquet file written.
pub trait EncryptionPropertiesFactory: Send + Sync {
    /// Create encryption properties for a new file.
    fn file_encryption_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>>;

    /// The dot-separated paths of the columns that are encrypted with their own key,
    /// or none for uniform encryption.
    ///
    /// This is used to validate the columns before any file is written.
    fn encrypted_column_names(&self) -> Vec<String>;
}

/// Creates the properties used to decrypt each Parquet file read.
pub trait DecryptionPropertiesFactory: Send + Sync {
    /// Create decryption properties for a file.
    fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>>;

    /// Select properties by source path, returning none for a plaintext file.
    fn file_decryption_properties_for_path(
        &self,
        _path: Option<&PlRefPath>,
    ) -> PolarsResult<Option<Arc<FileDecryptionProperties>>> {
        self.file_decryption_properties().map(Some)
    }

    /// Whether this factory also selects keys for associated deletion files.
    fn applies_to_deletion_files(&self) -> bool {
        false
    }
}

struct PerFileDecryption(PlHashMap<PlRefPath, Arc<FileDecryptionProperties>>);

impl DecryptionPropertiesFactory for PerFileDecryption {
    fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        polars_bail!(InvalidOperation: "path-specific decryption requires a file path")
    }

    fn file_decryption_properties_for_path(
        &self,
        path: Option<&PlRefPath>,
    ) -> PolarsResult<Option<Arc<FileDecryptionProperties>>> {
        let path = path.ok_or_else(
            || polars_err!(InvalidOperation: "path-specific decryption requires a file path"),
        )?;
        Ok(self.0.get(path).cloned())
    }

    fn applies_to_deletion_files(&self) -> bool {
        true
    }
}

/// File encryption properties can be specified directly and used as a factory.
impl EncryptionPropertiesFactory for Arc<FileEncryptionProperties> {
    fn file_encryption_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        Ok(Arc::clone(self))
    }

    fn encrypted_column_names(&self) -> Vec<String> {
        FileEncryptionProperties::encrypted_column_names(self)
            .map(str::to_string)
            .collect()
    }
}

/// File decryption properties can be specified directly and used as a factory.
impl DecryptionPropertiesFactory for Arc<FileDecryptionProperties> {
    fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        Ok(Arc::clone(self))
    }
}

/// A source of encryption or decryption properties, with an ID that identifies it.
///
/// The ID is kept when serializing, so that a deserialized plan has the same equality
/// as the original, and [`EncryptionSharing`] can share its values again.
struct Identified<T> {
    id: UniqueId,
    source: T,
}

impl<T> Identified<T> {
    fn new(source: T) -> Arc<Self> {
        Arc::new(Self {
            id: UniqueId::new(),
            source,
        })
    }
}

/// How encryption properties are created.
enum EncryptionSource {
    /// The same properties, with explicit keys, are used for every file.
    Fixed(Arc<FileEncryptionProperties>),
    /// Keys are generated for each file, and wrapped using a KMS.
    #[cfg(feature = "parquet_kms")]
    Kms(Arc<kms::KmsEncryption>),
    /// Any other factory, which can't be serialized.
    Custom(Box<dyn EncryptionPropertiesFactory>),
}

impl EncryptionSource {
    fn factory(&self) -> &dyn EncryptionPropertiesFactory {
        match self {
            Self::Fixed(properties) => properties,
            #[cfg(feature = "parquet_kms")]
            Self::Kms(kms) => kms.as_ref(),
            Self::Custom(factory) => factory.as_ref(),
        }
    }
}

/// Determines how to encrypt Parquet files when writing.
///
/// This is compared and hashed by an ID assigned when it is created, which is kept
/// when serializing. It can be serialized when it uses fixed properties or a KMS, and
/// the serialized form includes any secret keys and tokens.
#[derive(Clone)]
pub struct ParquetEncryption(Arc<Identified<EncryptionSource>>);

impl ParquetEncryption {
    /// Create properties for each file using a factory.
    ///
    /// A query using a custom factory can't be serialized.
    pub fn new(factory: impl EncryptionPropertiesFactory + 'static) -> Self {
        Self(Identified::new(EncryptionSource::Custom(Box::new(factory))))
    }

    /// Get the properties to use for encrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        self.0.source.factory().file_encryption_properties()
    }

    /// The dot-separated paths of the columns that are encrypted with their own key,
    /// or none for uniform encryption.
    pub fn encrypted_column_names(&self) -> Vec<String> {
        self.0.source.factory().encrypted_column_names()
    }

    /// Checks that each column that is to be encrypted with its own key is present in
    /// the schema exactly once.
    pub fn validate_encrypted_column_names(&self, schema: &SchemaDescriptor) -> PolarsResult<()> {
        let encrypted_columns = self.encrypted_column_names();
        validate_encrypted_column_names(encrypted_columns.iter().map(String::as_str), schema)?;
        Ok(())
    }
}

impl From<Arc<FileEncryptionProperties>> for ParquetEncryption {
    fn from(properties: Arc<FileEncryptionProperties>) -> Self {
        Self(Identified::new(EncryptionSource::Fixed(properties)))
    }
}

#[cfg(feature = "parquet_kms")]
impl From<kms::KmsEncryption> for ParquetEncryption {
    fn from(kms: kms::KmsEncryption) -> Self {
        Self(Identified::new(EncryptionSource::Kms(Arc::new(kms))))
    }
}

/// How decryption properties are created.
enum DecryptionSource {
    /// The same properties, with explicit keys, are used for every file.
    Fixed(Arc<FileDecryptionProperties>),
    /// Keys are unwrapped using a KMS.
    #[cfg(feature = "parquet_kms")]
    Kms(Arc<kms::KmsDecryption>),
    /// Any other factory, which can't be serialized.
    Custom(Box<dyn DecryptionPropertiesFactory>),
}

impl DecryptionSource {
    fn factory(&self) -> &dyn DecryptionPropertiesFactory {
        match self {
            Self::Fixed(properties) => properties,
            #[cfg(feature = "parquet_kms")]
            Self::Kms(kms) => kms.as_ref(),
            Self::Custom(factory) => factory.as_ref(),
        }
    }
}

/// Determines how to decrypt Parquet files when reading.
///
/// This is compared and hashed by an ID assigned when it is created, which is kept
/// when serializing. It can be serialized when it uses fixed properties or a KMS, and
/// the serialized form includes any secret keys and tokens.
// This is a thin pointer, keeping `FileScanIR` small.
#[derive(Clone)]
pub struct ParquetDecryption(Arc<Identified<DecryptionSource>>);

impl ParquetDecryption {
    /// Create properties for each file using a factory.
    ///
    /// A query using a custom factory can't be serialized.
    pub fn new(factory: impl DecryptionPropertiesFactory + 'static) -> Self {
        Self(Identified::new(DecryptionSource::Custom(Box::new(factory))))
    }

    /// Select keys by path for data and deletion files. Unlisted paths are plaintext.
    pub fn from_file_properties(
        files: PlHashMap<PlRefPath, Arc<FileDecryptionProperties>>,
    ) -> Self {
        Self::new(PerFileDecryption(files))
    }

    /// Get the properties to use for decrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        self.0.source.factory().file_decryption_properties()
    }

    pub fn file_properties_for_path(
        &self,
        path: Option<&PlRefPath>,
    ) -> PolarsResult<Option<Arc<FileDecryptionProperties>>> {
        self.0
            .source
            .factory()
            .file_decryption_properties_for_path(path)
    }

    pub fn for_deletion_files(&self) -> Option<Self> {
        self.0
            .source
            .factory()
            .applies_to_deletion_files()
            .then(|| self.clone())
    }
}

impl From<Arc<FileDecryptionProperties>> for ParquetDecryption {
    fn from(properties: Arc<FileDecryptionProperties>) -> Self {
        Self(Identified::new(DecryptionSource::Fixed(properties)))
    }
}

#[cfg(feature = "parquet_kms")]
impl From<kms::KmsDecryption> for ParquetDecryption {
    fn from(kms: kms::KmsDecryption) -> Self {
        Self(Identified::new(DecryptionSource::Kms(Arc::new(kms))))
    }
}

/// Get the properties to use for decrypting a single file, if decryption is configured.
pub fn file_decryption_properties(
    decryption: Option<&ParquetDecryption>,
) -> PolarsResult<Option<Arc<FileDecryptionProperties>>> {
    decryption
        .map(ParquetDecryption::file_properties)
        .transpose()
}

pub fn file_decryption_properties_for_path(
    decryption: Option<&ParquetDecryption>,
    path: Option<&PlRefPath>,
) -> PolarsResult<Option<Arc<FileDecryptionProperties>>> {
    decryption
        .map(|decryption| decryption.file_properties_for_path(path))
        .transpose()
        .map(Option::flatten)
}

/// The serialized forms of [`ParquetEncryption`] and [`ParquetDecryption`].
#[cfg(any(feature = "serde", feature = "dsl-schema"))]
mod serialization {
    use super::*;

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetEncryption"))]
    pub(super) struct SerializableParquetEncryption {
        id: UniqueId,
        source: SerializableEncryptionSource,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetEncryptionSource"))]
    enum SerializableEncryptionSource {
        Fixed(Arc<FileEncryptionProperties>),
        #[cfg(feature = "parquet_kms")]
        Kms(Arc<kms::KmsEncryption>),
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetDecryption"))]
    pub(super) struct SerializableParquetDecryption {
        id: UniqueId,
        source: SerializableDecryptionSource,
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetDecryptionSource"))]
    enum SerializableDecryptionSource {
        Fixed(Arc<FileDecryptionProperties>),
        #[cfg(feature = "parquet_kms")]
        Kms(Arc<kms::KmsDecryption>),
    }

    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    impl ParquetEncryption {
        pub(super) fn to_serializable(&self) -> Option<SerializableParquetEncryption> {
            let source = match &self.0.source {
                EncryptionSource::Fixed(properties) => {
                    SerializableEncryptionSource::Fixed(Arc::clone(properties))
                },
                #[cfg(feature = "parquet_kms")]
                EncryptionSource::Kms(kms) => SerializableEncryptionSource::Kms(Arc::clone(kms)),
                EncryptionSource::Custom(_) => return None,
            };
            Some(SerializableParquetEncryption {
                id: self.0.id,
                source,
            })
        }

        #[cfg(feature = "serde")]
        pub(super) fn from_serializable(serializable: SerializableParquetEncryption) -> Self {
            let SerializableParquetEncryption { id, source } = serializable;
            let source = match source {
                SerializableEncryptionSource::Fixed(properties) => {
                    EncryptionSource::Fixed(properties)
                },
                #[cfg(feature = "parquet_kms")]
                SerializableEncryptionSource::Kms(kms) => EncryptionSource::Kms(kms),
            };
            Self(Arc::new(Identified { id, source }))
        }
    }

    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    impl ParquetDecryption {
        pub(super) fn to_serializable(&self) -> Option<SerializableParquetDecryption> {
            let source = match &self.0.source {
                DecryptionSource::Fixed(properties) => {
                    SerializableDecryptionSource::Fixed(Arc::clone(properties))
                },
                #[cfg(feature = "parquet_kms")]
                DecryptionSource::Kms(kms) => SerializableDecryptionSource::Kms(Arc::clone(kms)),
                DecryptionSource::Custom(_) => return None,
            };
            Some(SerializableParquetDecryption {
                id: self.0.id,
                source,
            })
        }

        #[cfg(feature = "serde")]
        pub(super) fn from_serializable(serializable: SerializableParquetDecryption) -> Self {
            let SerializableParquetDecryption { id, source } = serializable;
            let source = match source {
                SerializableDecryptionSource::Fixed(properties) => {
                    DecryptionSource::Fixed(properties)
                },
                #[cfg(feature = "parquet_kms")]
                SerializableDecryptionSource::Kms(kms) => DecryptionSource::Kms(kms),
            };
            Self(Arc::new(Identified { id, source }))
        }
    }
}

/// Implement traits that compare and hash by ID, and serialize through the
/// serializable form.
macro_rules! impl_opaque_traits {
    ($t:ident, $serializable:ident, $description:literal) => {
        impl Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(stringify!($t))
            }
        }

        impl Eq for $t {}

        impl PartialEq for $t {
            fn eq(&self, other: &Self) -> bool {
                self.0.id == other.0.id
            }
        }

        impl Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.id.hash(state)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $t {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                serialization::$serializable::deserialize(deserializer).map(Self::from_serializable)
            }
        }

        #[cfg(feature = "serde")]
        impl serde::Serialize for $t {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                use serde::ser::Error;
                match self.to_serializable() {
                    Some(serializable) => serializable.serialize(serializer),
                    None => Err(S::Error::custom(concat!(
                        "cannot serialize ",
                        $description,
                        " created by a custom factory"
                    ))),
                }
            }
        }

        #[cfg(feature = "dsl-schema")]
        impl schemars::JsonSchema for $t {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                serialization::$serializable::schema_name()
            }

            fn schema_id() -> std::borrow::Cow<'static, str> {
                serialization::$serializable::schema_id()
            }

            fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
                serialization::$serializable::json_schema(generator)
            }
        }
    };
}

impl_opaque_traits!(
    ParquetEncryption,
    SerializableParquetEncryption,
    "parquet encryption properties"
);
impl_opaque_traits!(
    ParquetDecryption,
    SerializableParquetDecryption,
    "parquet decryption properties"
);

/// Shares the encryption and decryption configurations in a plan that have the same
/// ID, along with the KMS crypto factories they use.
///
/// Values with the same ID are separate objects after a plan is deserialized. Sharing
/// them again means they share caches, such as the KMS clients and keys of a crypto
/// factory. The first value seen with each ID is used.
#[derive(Default)]
pub struct EncryptionSharing {
    encryption: PlHashMap<UniqueId, ParquetEncryption>,
    decryption: PlHashMap<UniqueId, ParquetDecryption>,
    #[cfg(feature = "parquet_kms")]
    crypto_factories: kms::SeenCryptoFactories,
}

impl EncryptionSharing {
    /// Returns the shared encryption with the same ID as `encryption`.
    pub fn share_encryption(&mut self, encryption: ParquetEncryption) -> ParquetEncryption {
        let id = encryption.0.id;
        if let Some(first) = self.encryption.get(&id) {
            return first.clone();
        }

        #[cfg(feature = "parquet_kms")]
        let encryption = match &encryption.0.source {
            EncryptionSource::Kms(kms) => {
                match kms.with_shared_crypto_factory(&mut self.crypto_factories) {
                    Some(kms) => ParquetEncryption(Arc::new(Identified {
                        id,
                        source: EncryptionSource::Kms(Arc::new(kms)),
                    })),
                    None => encryption,
                }
            },
            _ => encryption,
        };

        self.encryption.insert(id, encryption.clone());
        encryption
    }

    /// Returns the shared decryption with the same ID as `decryption`.
    pub fn share_decryption(&mut self, decryption: ParquetDecryption) -> ParquetDecryption {
        let id = decryption.0.id;
        if let Some(first) = self.decryption.get(&id) {
            return first.clone();
        }

        #[cfg(feature = "parquet_kms")]
        let decryption = match &decryption.0.source {
            DecryptionSource::Kms(kms) => {
                match kms.with_shared_crypto_factory(&mut self.crypto_factories) {
                    Some(kms) => ParquetDecryption(Arc::new(Identified {
                        id,
                        source: DecryptionSource::Kms(Arc::new(kms)),
                    })),
                    None => decryption,
                }
            },
            _ => decryption,
        };

        self.decryption.insert(id, decryption.clone());
        decryption
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Factory;

    impl DecryptionPropertiesFactory for Factory {
        fn file_decryption_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
            Ok(FileDecryptionProperties::builder(b"0123456789012345".to_vec()).build()?)
        }
    }

    #[test]
    fn test_equality() {
        let properties = FileDecryptionProperties::builder(b"0123456789012345".to_vec())
            .build()
            .unwrap();
        let a = ParquetDecryption::from(Arc::clone(&properties));
        assert_eq!(a, a.clone());
        assert_ne!(a, ParquetDecryption::from(properties));

        let b = ParquetDecryption::new(Factory);
        assert_eq!(b, b.clone());
        assert_ne!(b, ParquetDecryption::new(Factory));
    }

    #[test]
    fn test_file_properties() {
        let decryption = ParquetDecryption::new(Factory);
        let a = decryption.file_properties().unwrap();
        let b = decryption.file_properties().unwrap();
        assert!(!Arc::ptr_eq(&a, &b));

        let decryption = ParquetDecryption::from(Arc::clone(&a));
        assert!(Arc::ptr_eq(&decryption.file_properties().unwrap(), &a));
    }

    #[test]
    fn test_factory_with_source_path() {
        let path = PlRefPath::new("data.parquet");
        let decryption = ParquetDecryption::new(Factory);
        for path in [None, Some(&path)] {
            let properties = file_decryption_properties_for_path(Some(&decryption), path)
                .unwrap()
                .unwrap();
            assert_eq!(
                properties.footer_key(None).unwrap().as_slice(),
                b"0123456789012345"
            );
        }
        assert!(decryption.for_deletion_files().is_none());
        let uniform = ParquetDecryption::from(decryption.file_properties().unwrap());
        assert!(uniform.for_deletion_files().is_none());
        assert!(
            file_decryption_properties_for_path(None, Some(&path))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_path_specific_properties() {
        let data_path = PlRefPath::new("data.parquet");
        let delete_path = PlRefPath::new("delete.parquet");
        let plaintext_path = PlRefPath::new("plaintext.parquet");
        let data = FileDecryptionProperties::builder(vec![1; 16])
            .with_aad_prefix(b"data-aad".to_vec())
            .build()
            .unwrap();
        let deletes = FileDecryptionProperties::builder(vec![2; 16])
            .with_aad_prefix(b"delete-aad".to_vec())
            .build()
            .unwrap();
        let decryption = ParquetDecryption::from_file_properties(
            [(data_path.clone(), data), (delete_path.clone(), deletes)]
                .into_iter()
                .collect(),
        );
        let deletion_decryption = decryption.for_deletion_files().unwrap();

        std::thread::scope(|scope| {
            for factory in [&decryption, &deletion_decryption] {
                for (path, key, aad) in [
                    (&delete_path, 2, b"delete-aad".as_slice()),
                    (&data_path, 1, b"data-aad".as_slice()),
                    (&delete_path, 2, b"delete-aad".as_slice()),
                ] {
                    scope.spawn(move || {
                        let properties =
                            file_decryption_properties_for_path(Some(factory), Some(path))
                                .unwrap()
                                .unwrap();
                        assert_eq!(properties.footer_key(None).unwrap().as_slice(), &[key; 16]);
                        assert_eq!(properties.aad_prefix().unwrap().as_slice(), aad);
                    });
                }
            }
        });
        assert!(
            deletion_decryption
                .file_properties_for_path(Some(&plaintext_path))
                .unwrap()
                .is_none()
        );
        assert!(decryption.file_properties_for_path(None).is_err());
        assert!(decryption.file_properties().is_err());
    }

    #[test]
    fn test_thin_pointers() {
        assert_eq!(
            std::mem::size_of::<Option<ParquetDecryption>>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<Option<ParquetEncryption>>(),
            std::mem::size_of::<usize>()
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_fixed() {
        let properties = FileDecryptionProperties::builder(b"0123456789012345".to_vec())
            .with_aad_prefix(b"prefix".to_vec())
            .build()
            .unwrap();
        let decryption = ParquetDecryption::from(Arc::clone(&properties));
        let json = serde_json::to_string(&decryption).unwrap();
        let deserialized: ParquetDecryption = serde_json::from_str(&json).unwrap();
        assert!(*deserialized.file_properties().unwrap() == *properties);

        let properties = FileEncryptionProperties::builder(b"0123456789012345".to_vec())
            .with_column_key("a", b"1234567890123450".to_vec())
            .build()
            .unwrap();
        let encryption = ParquetEncryption::from(Arc::clone(&properties));
        let json = serde_json::to_string(&encryption).unwrap();
        let deserialized: ParquetEncryption = serde_json::from_str(&json).unwrap();
        assert_eq!(*deserialized.file_properties().unwrap(), *properties);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_keeps_identity() {
        use polars_utils::pl_serialize::{deserialize_dsl, serialize_dsl};

        let properties = FileDecryptionProperties::builder(b"0123456789012345".to_vec())
            .build()
            .unwrap();
        let a = ParquetDecryption::from(Arc::clone(&properties));
        let b = ParquetDecryption::from(properties);
        let values = vec![a.clone(), b.clone(), a.clone()];

        let mut bytes = Vec::new();
        serialize_dsl(&mut bytes, &values).unwrap();
        let deserialized: Vec<ParquetDecryption> = deserialize_dsl(&mut bytes.as_slice()).unwrap();

        // Equality is kept, but each occurrence is a separate object.
        assert_eq!(deserialized, values);
        assert_ne!(deserialized[0], deserialized[1]);
        assert!(!Arc::ptr_eq(&deserialized[0].0, &deserialized[2].0));

        // Sharing uses the first object seen with each ID.
        let mut sharing = EncryptionSharing::default();
        let shared: Vec<ParquetDecryption> = deserialized
            .iter()
            .map(|d| sharing.share_decryption(d.clone()))
            .collect();
        assert_eq!(shared, values);
        assert!(Arc::ptr_eq(&shared[0].0, &deserialized[0].0));
        assert!(Arc::ptr_eq(&shared[1].0, &deserialized[1].0));
        assert!(Arc::ptr_eq(&shared[2].0, &deserialized[0].0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_custom_factory_fails() {
        let err = serde_json::to_string(&ParquetDecryption::new(Factory)).unwrap_err();
        assert!(
            err.to_string().contains(
                "cannot serialize parquet decryption properties created by a custom factory"
            ),
            "{err}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serialize_path_specific_keys_fails() {
        let properties = FileDecryptionProperties::builder(b"0123456789012345".to_vec())
            .build()
            .unwrap();
        let decryption = ParquetDecryption::from_file_properties(
            [(PlRefPath::new("data.parquet"), properties)]
                .into_iter()
                .collect(),
        );
        let mut sharing = EncryptionSharing::default();
        let shared = sharing.share_decryption(decryption.clone());
        assert_eq!(shared, decryption);
        let err = serde_json::to_string(&shared).unwrap_err();
        assert!(err.to_string().contains("created by a custom factory"));
        assert!(!err.to_string().contains("0123456789012345"));
    }
}
