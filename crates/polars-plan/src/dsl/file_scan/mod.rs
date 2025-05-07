use std::hash::Hash;
use std::sync::Mutex;

use polars_io::cloud::CloudOptions;
#[cfg(feature = "csv")]
use polars_io::csv::read::CsvReadOptions;
#[cfg(feature = "ipc")]
use polars_io::ipc::IpcScanOptions;
#[cfg(feature = "parquet")]
use polars_io::parquet::metadata::FileMetadataRef;
#[cfg(feature = "parquet")]
use polars_io::parquet::read::ParquetOptions;
use polars_io::{HiveOptions, RowIndex};
use polars_utils::slice_enum::Slice;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use super::*;

#[cfg(feature = "python")]
pub mod python_dataset;
#[cfg(feature = "python")]
pub use python_dataset::{DATASET_PROVIDER_VTABLE, PythonDatasetProviderVTable};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ScanFlags : u32 {
        const SPECIALIZED_PREDICATE_FILTER = 0x01;
    }
}

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
// TODO: Arc<> some of the options and the cloud options.
pub enum FileScan {
    #[cfg(feature = "csv")]
    Csv { options: CsvReadOptions },

    #[cfg(feature = "json")]
    NDJson { options: NDJsonReadOptions },

    #[cfg(feature = "parquet")]
    Parquet {
        options: ParquetOptions,
        #[cfg_attr(feature = "serde", serde(skip))]
        metadata: Option<FileMetadataRef>,
    },

    #[cfg(feature = "ipc")]
    Ipc {
        options: IpcScanOptions,
        #[cfg_attr(feature = "serde", serde(skip))]
        metadata: Option<Arc<arrow::io::ipc::read::FileMetadata>>,
    },

    #[cfg(feature = "python")]
    PythonDataset {
        dataset_object: Arc<python_dataset::PythonDatasetProvider>,

        #[cfg_attr(feature = "serde", serde(skip, default))]
        cached_ir: Arc<Mutex<Option<ExpandedDataset>>>,
    },

    #[cfg_attr(feature = "serde", serde(skip))]
    Anonymous {
        options: Arc<AnonymousScanOptions>,
        function: Arc<dyn AnonymousScan>,
    },
}

impl FileScan {
    pub fn flags(&self) -> ScanFlags {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => ScanFlags::empty(),
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => ScanFlags::empty(),
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => ScanFlags::SPECIALIZED_PREDICATE_FILTER,
            #[cfg(feature = "json")]
            Self::NDJson { .. } => ScanFlags::empty(),
            #[allow(unreachable_patterns)]
            _ => ScanFlags::empty(),
        }
    }

    pub(crate) fn sort_projection(&self, _has_row_index: bool) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => _has_row_index,
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => false,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn streamable(&self) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => false,
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => true,
            #[cfg(feature = "json")]
            Self::NDJson { .. } => false,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MissingColumnsPolicy {
    #[default]
    Raise,
    /// Inserts full-NULL columns for the missing ones.
    Insert,
}

/// Used by scans.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CastColumnsPolicy {
    /// Allow casting when target dtype is lossless supertype
    pub integer_upcast: bool,

    /// Allow Float32 -> Float64
    pub float_upcast: bool,
    /// Allow Float64 -> Float32
    pub float_downcast: bool,

    /// Allow datetime[ns] to be casted to any lower precision. Important for
    /// being able to read datasets written by spark.
    pub datetime_nanoseconds_downcast: bool,

    /// Allow casting to change time units.
    pub datetime_convert_timezone: bool,

    pub missing_struct_fields: MissingColumnsPolicy,
    pub extra_struct_fields: ExtraColumnsPolicy,
}

impl CastColumnsPolicy {
    /// Configuration variant that defaults to raising on mismatch.
    pub const ERROR_ON_MISMATCH: Self = Self {
        integer_upcast: false,
        float_upcast: false,
        float_downcast: false,
        datetime_nanoseconds_downcast: false,
        datetime_convert_timezone: false,
        missing_struct_fields: MissingColumnsPolicy::Raise,
        extra_struct_fields: ExtraColumnsPolicy::Raise,
    };
}

impl Default for CastColumnsPolicy {
    fn default() -> Self {
        Self::ERROR_ON_MISMATCH
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtraColumnsPolicy {
    /// Error if there are extra columns outside the target schema.
    #[default]
    Raise,
    Ignore,
}

/// Scan arguments shared across different scan types.
#[derive(Debug, Clone, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UnifiedScanArgs {
    /// User-provided schema of the file. Will be inferred during IR conversion
    /// if None.
    pub schema: Option<SchemaRef>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,

    pub rechunk: bool,
    pub cache: bool,
    pub glob: bool,

    pub projection: Option<Arc<[PlSmallStr]>>,
    pub row_index: Option<RowIndex>,
    /// Slice applied before predicates
    pub pre_slice: Option<Slice>,

    pub cast_columns_policy: CastColumnsPolicy,
    pub missing_columns_policy: MissingColumnsPolicy,
    pub include_file_paths: Option<PlSmallStr>,
}

/// Manual impls of Eq/Hash, as some fields are `Arc<T>` where T does not have Eq/Hash. For these
/// fields we compare the pointer addresses instead.
mod _file_scan_eq_hash {
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    use super::FileScan;

    impl PartialEq for FileScan {
        fn eq(&self, other: &Self) -> bool {
            FileScanEqHashWrap::from(self) == FileScanEqHashWrap::from(other)
        }
    }

    impl Eq for FileScan {}

    impl Hash for FileScan {
        fn hash<H: Hasher>(&self, state: &mut H) {
            FileScanEqHashWrap::from(self).hash(state)
        }
    }

    /// # Hash / Eq safety
    /// * All usizes originate from `Arc<>`s, and the lifetime of this enum is bound to that of the
    ///   input ref.
    #[derive(PartialEq, Hash)]
    pub enum FileScanEqHashWrap<'a> {
        #[cfg(feature = "csv")]
        Csv {
            options: &'a polars_io::csv::read::CsvReadOptions,
        },

        #[cfg(feature = "json")]
        NDJson {
            options: &'a crate::prelude::NDJsonReadOptions,
        },

        #[cfg(feature = "parquet")]
        Parquet {
            options: &'a polars_io::prelude::ParquetOptions,
            metadata: Option<usize>,
        },

        #[cfg(feature = "ipc")]
        Ipc {
            options: &'a polars_io::prelude::IpcScanOptions,
            metadata: Option<usize>,
        },

        #[cfg(feature = "python")]
        PythonDataset {
            dataset_object: usize,
            cached_ir: usize,
        },

        Anonymous {
            options: &'a crate::dsl::AnonymousScanOptions,
            function: usize,
        },

        /// Variant to ensure the lifetime is used regardless of feature gate combination.
        #[expect(unused)]
        Phantom(&'a ()),
    }

    impl<'a> From<&'a FileScan> for FileScanEqHashWrap<'a> {
        fn from(value: &'a FileScan) -> Self {
            match value {
                #[cfg(feature = "csv")]
                FileScan::Csv { options } => FileScanEqHashWrap::Csv { options },

                #[cfg(feature = "json")]
                FileScan::NDJson { options } => FileScanEqHashWrap::NDJson { options },

                #[cfg(feature = "parquet")]
                FileScan::Parquet { options, metadata } => FileScanEqHashWrap::Parquet {
                    options,
                    metadata: metadata.as_ref().map(arc_as_ptr),
                },

                #[cfg(feature = "ipc")]
                FileScan::Ipc { options, metadata } => FileScanEqHashWrap::Ipc {
                    options,
                    metadata: metadata.as_ref().map(arc_as_ptr),
                },

                #[cfg(feature = "python")]
                FileScan::PythonDataset {
                    dataset_object,
                    cached_ir,
                } => FileScanEqHashWrap::PythonDataset {
                    dataset_object: arc_as_ptr(dataset_object),
                    cached_ir: arc_as_ptr(cached_ir),
                },

                FileScan::Anonymous { options, function } => FileScanEqHashWrap::Anonymous {
                    options,
                    function: arc_as_ptr(function),
                },
            }
        }
    }

    fn arc_as_ptr<T: ?Sized>(arc: &Arc<T>) -> usize {
        Arc::as_ptr(arc) as *const () as usize
    }
}
