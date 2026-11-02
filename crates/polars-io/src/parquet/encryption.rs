//! Configuration for reading and writing files encrypted with Parquet modular encryption.

#[cfg(feature = "parquet_kms")]
pub mod kms;

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use polars_error::PolarsResult;
use polars_parquet::parquet::encryption::decrypt::FileDecryptionProperties;
use polars_parquet::parquet::encryption::encrypt::{
    FileEncryptionProperties, validate_encrypted_column_names,
};
use polars_parquet::parquet::metadata::SchemaDescriptor;

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
/// This is compared and hashed by pointer. It can be serialized when it uses fixed
/// properties or a KMS, and the serialized form includes any secret keys and tokens.
#[derive(Clone)]
pub struct ParquetEncryption(Arc<EncryptionSource>);

impl ParquetEncryption {
    /// Create properties for each file using a factory.
    ///
    /// A query using a custom factory can't be serialized.
    pub fn new(factory: impl EncryptionPropertiesFactory + 'static) -> Self {
        Self(Arc::new(EncryptionSource::Custom(Box::new(factory))))
    }

    /// Get the properties to use for encrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        self.0.factory().file_encryption_properties()
    }

    /// The dot-separated paths of the columns that are encrypted with their own key,
    /// or none for uniform encryption.
    pub fn encrypted_column_names(&self) -> Vec<String> {
        self.0.factory().encrypted_column_names()
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
        Self(Arc::new(EncryptionSource::Fixed(properties)))
    }
}

#[cfg(feature = "parquet_kms")]
impl From<kms::KmsEncryption> for ParquetEncryption {
    fn from(kms: kms::KmsEncryption) -> Self {
        Self(Arc::new(EncryptionSource::Kms(Arc::new(kms))))
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
/// This is compared and hashed by pointer. It can be serialized when it uses fixed
/// properties or a KMS, and the serialized form includes any secret keys and tokens.
// This is a thin pointer, keeping `FileScanIR` small.
#[derive(Clone)]
pub struct ParquetDecryption(Arc<DecryptionSource>);

impl ParquetDecryption {
    /// Create properties for each file using a factory.
    ///
    /// A query using a custom factory can't be serialized.
    pub fn new(factory: impl DecryptionPropertiesFactory + 'static) -> Self {
        Self(Arc::new(DecryptionSource::Custom(Box::new(factory))))
    }

    /// Get the properties to use for decrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        self.0.factory().file_decryption_properties()
    }
}

impl From<Arc<FileDecryptionProperties>> for ParquetDecryption {
    fn from(properties: Arc<FileDecryptionProperties>) -> Self {
        Self(Arc::new(DecryptionSource::Fixed(properties)))
    }
}

#[cfg(feature = "parquet_kms")]
impl From<kms::KmsDecryption> for ParquetDecryption {
    fn from(kms: kms::KmsDecryption) -> Self {
        Self(Arc::new(DecryptionSource::Kms(Arc::new(kms))))
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

/// The serialized forms of [`ParquetEncryption`] and [`ParquetDecryption`].
#[cfg(any(feature = "serde", feature = "dsl-schema"))]
mod serialization {
    use super::*;

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetEncryption"))]
    pub(super) enum SerializableParquetEncryption {
        Fixed(Arc<FileEncryptionProperties>),
        #[cfg(feature = "parquet_kms")]
        Kms(Arc<kms::KmsEncryption>),
    }

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
    #[cfg_attr(feature = "dsl-schema", schemars(rename = "ParquetDecryption"))]
    pub(super) enum SerializableParquetDecryption {
        Fixed(Arc<FileDecryptionProperties>),
        #[cfg(feature = "parquet_kms")]
        Kms(Arc<kms::KmsDecryption>),
    }

    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    impl ParquetEncryption {
        pub(super) fn to_serializable(&self) -> Option<SerializableParquetEncryption> {
            match &*self.0 {
                EncryptionSource::Fixed(properties) => {
                    Some(SerializableParquetEncryption::Fixed(Arc::clone(properties)))
                },
                #[cfg(feature = "parquet_kms")]
                EncryptionSource::Kms(kms) => {
                    Some(SerializableParquetEncryption::Kms(Arc::clone(kms)))
                },
                EncryptionSource::Custom(_) => None,
            }
        }
    }

    impl From<SerializableParquetEncryption> for ParquetEncryption {
        fn from(serializable: SerializableParquetEncryption) -> Self {
            Self(Arc::new(match serializable {
                SerializableParquetEncryption::Fixed(properties) => {
                    EncryptionSource::Fixed(properties)
                },
                #[cfg(feature = "parquet_kms")]
                SerializableParquetEncryption::Kms(kms) => EncryptionSource::Kms(kms),
            }))
        }
    }

    #[cfg_attr(not(feature = "serde"), allow(dead_code))]
    impl ParquetDecryption {
        pub(super) fn to_serializable(&self) -> Option<SerializableParquetDecryption> {
            match &*self.0 {
                DecryptionSource::Fixed(properties) => {
                    Some(SerializableParquetDecryption::Fixed(Arc::clone(properties)))
                },
                #[cfg(feature = "parquet_kms")]
                DecryptionSource::Kms(kms) => {
                    Some(SerializableParquetDecryption::Kms(Arc::clone(kms)))
                },
                DecryptionSource::Custom(_) => None,
            }
        }
    }

    impl From<SerializableParquetDecryption> for ParquetDecryption {
        fn from(serializable: SerializableParquetDecryption) -> Self {
            Self(Arc::new(match serializable {
                SerializableParquetDecryption::Fixed(properties) => {
                    DecryptionSource::Fixed(properties)
                },
                #[cfg(feature = "parquet_kms")]
                SerializableParquetDecryption::Kms(kms) => DecryptionSource::Kms(kms),
            }))
        }
    }
}

/// Implement traits that compare and hash by pointer, and serialize through the
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
                Arc::ptr_eq(&self.0, &other.0)
            }
        }

        impl Hash for $t {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                state.write_usize(Arc::as_ptr(&self.0).cast::<()>() as usize)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> serde::Deserialize<'de> for $t {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                serialization::$serializable::deserialize(deserializer).map(Self::from)
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
    fn test_pointer_equality() {
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
    fn test_serialize_custom_factory_fails() {
        let err = serde_json::to_string(&ParquetDecryption::new(Factory)).unwrap_err();
        assert!(
            err.to_string().contains(
                "cannot serialize parquet decryption properties created by a custom factory"
            ),
            "{err}"
        );
    }
}
