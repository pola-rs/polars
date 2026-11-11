//! Configuration for reading and writing files encrypted with Parquet modular encryption.

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

/// Determines how to encrypt Parquet files when writing.
///
/// This may hold secret keys, so can't be serialized, and is compared and hashed
/// by pointer.
#[derive(Clone)]
pub struct ParquetEncryption(Arc<dyn EncryptionPropertiesFactory>);

impl ParquetEncryption {
    /// Create properties for each file using a factory.
    pub fn new(factory: impl EncryptionPropertiesFactory + 'static) -> Self {
        Self(Arc::new(factory))
    }

    /// Get the properties to use for encrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileEncryptionProperties>> {
        self.0.file_encryption_properties()
    }

    /// The dot-separated paths of the columns that are encrypted with their own key,
    /// or none for uniform encryption.
    pub fn encrypted_column_names(&self) -> Vec<String> {
        self.0.encrypted_column_names()
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
        Self::new(properties)
    }
}

/// Determines how to decrypt Parquet files when reading.
///
/// This may hold secret keys, so can't be serialized, and is compared and hashed
/// by pointer.
// The factory is double-boxed to make this a thin pointer, keeping `FileScanIR` small.
#[derive(Clone)]
pub struct ParquetDecryption(Arc<Box<dyn DecryptionPropertiesFactory>>);

impl ParquetDecryption {
    /// Create properties for each file using a factory.
    pub fn new(factory: impl DecryptionPropertiesFactory + 'static) -> Self {
        Self(Arc::new(Box::new(factory)))
    }

    /// Get the properties to use for decrypting a single file.
    pub fn file_properties(&self) -> PolarsResult<Arc<FileDecryptionProperties>> {
        self.0.file_decryption_properties()
    }
}

impl From<Arc<FileDecryptionProperties>> for ParquetDecryption {
    fn from(properties: Arc<FileDecryptionProperties>) -> Self {
        Self::new(properties)
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

/// Implement traits that compare and hash by pointer, and refuse to serialize.
macro_rules! impl_opaque_traits {
    ($t:ident, $description:literal) => {
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
            fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                use serde::de::Error;
                Err(D::Error::custom(concat!(
                    "cannot deserialize ",
                    $description
                )))
            }
        }

        #[cfg(feature = "serde")]
        impl serde::Serialize for $t {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                use serde::ser::Error;
                Err(S::Error::custom(concat!("cannot serialize ", $description)))
            }
        }

        #[cfg(feature = "dsl-schema")]
        impl schemars::JsonSchema for $t {
            fn schema_name() -> std::borrow::Cow<'static, str> {
                stringify!($t).into()
            }

            fn schema_id() -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed(concat!(module_path!(), "::", stringify!($t)))
            }

            fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
                Vec::<u8>::json_schema(generator)
            }
        }
    };
}

impl_opaque_traits!(ParquetEncryption, "parquet encryption properties");
impl_opaque_traits!(ParquetDecryption, "parquet decryption properties");

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
}
