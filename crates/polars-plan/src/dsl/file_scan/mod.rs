use std::hash::Hash;
use std::sync::Mutex;

use deletion::DeletionFilesList;
use polars_core::schema::iceberg::IcebergSchemaRef;
use polars_core::utils::get_numeric_upcast_supertype_lossless;
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
use crate::dsl::default_values::DefaultFieldValues;
pub mod default_values;
pub mod deletion;

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

const _: () = {
    assert!(std::mem::size_of::<FileScanDsl>() <= 100);
};

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
/// Note: This is cheaply cloneable.
pub enum FileScanDsl {
    #[cfg(feature = "csv")]
    Csv { options: Arc<CsvReadOptions> },

    #[cfg(feature = "json")]
    NDJson { options: NDJsonReadOptions },

    #[cfg(feature = "parquet")]
    Parquet { options: ParquetOptions },

    #[cfg(feature = "ipc")]
    Ipc { options: IpcScanOptions },

    #[cfg(feature = "python")]
    PythonDataset {
        dataset_object: Arc<python_dataset::PythonDatasetProvider>,
    },

    #[cfg(feature = "scan_lines")]
    Lines { name: PlSmallStr },

    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Anonymous {
        options: Arc<AnonymousScanOptions>,
        function: Arc<dyn AnonymousScan>,
        file_info: FileInfo,
    },
}

const _: () = {
    assert!(std::mem::size_of::<FileScanIR>() <= 80);
};

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
/// Note: This is cheaply cloneable.
pub enum FileScanIR {
    #[cfg(feature = "csv")]
    Csv { options: Arc<CsvReadOptions> },

    #[cfg(feature = "json")]
    NDJson { options: NDJsonReadOptions },

    #[cfg(feature = "parquet")]
    Parquet {
        options: ParquetOptions,
        #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
        metadata: Option<FileMetadataRef>,
    },

    #[cfg(feature = "ipc")]
    Ipc {
        options: IpcScanOptions,
        #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
        metadata: Option<Arc<arrow::io::ipc::read::FileMetadata>>,
    },

    #[cfg(feature = "python")]
    PythonDataset {
        dataset_object: Arc<python_dataset::PythonDatasetProvider>,
        cached_ir: Arc<Mutex<Option<ExpandedDataset>>>,
    },

    #[cfg(feature = "scan_lines")]
    Lines { name: PlSmallStr },

    #[cfg_attr(any(feature = "serde", feature = "dsl-schema"), serde(skip))]
    Anonymous {
        options: Arc<AnonymousScanOptions>,
        function: Arc<dyn AnonymousScan>,
    },
}

impl FileScanIR {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum MissingColumnsPolicy {
    #[default]
    Raise,
    /// Inserts full-NULL columns for the missing ones.
    Insert,
}

/// Used by scans.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct CastColumnsPolicy {
    /// Allow casting when target dtype is lossless supertype
    pub integer_upcast: bool,

    /// Allow casting integers to floats.
    #[cfg_attr(feature = "serde", serde(default))]
    pub integer_to_float_cast: bool,

    /// Allow upcasting from small floats to bigger floats
    pub float_upcast: bool,
    /// Allow downcasting from big floats to smaller floats
    pub float_downcast: bool,

    /// Allow datetime[ns] to be casted to any lower precision. Important for
    /// being able to read datasets written by spark.
    pub datetime_nanoseconds_downcast: bool,
    /// Allow datetime[us] to datetime[ms]
    pub datetime_microseconds_downcast: bool,

    /// Allow casting to change time units.
    pub datetime_convert_timezone: bool,

    /// DataType::Null to any
    pub null_upcast: bool,

    /// DataType::Categorical to string
    pub categorical_to_string: bool,

    pub missing_struct_fields: MissingColumnsPolicy,
    pub extra_struct_fields: ExtraColumnsPolicy,
}

impl CastColumnsPolicy {
    /// Configuration variant that defaults to raising on mismatch.
    pub const ERROR_ON_MISMATCH: Self = Self {
        integer_upcast: false,
        integer_to_float_cast: false,
        float_upcast: false,
        float_downcast: false,
        datetime_nanoseconds_downcast: false,
        datetime_microseconds_downcast: false,
        datetime_convert_timezone: false,
        null_upcast: true,
        categorical_to_string: false,
        missing_struct_fields: MissingColumnsPolicy::Raise,
        extra_struct_fields: ExtraColumnsPolicy::Raise,
    };
}

impl Default for CastColumnsPolicy {
    fn default() -> Self {
        Self::ERROR_ON_MISMATCH
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ExtraColumnsPolicy {
    /// Error if there are extra columns outside the target schema.
    #[default]
    Raise,
    Ignore,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq, strum_macros::IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ColumnMapping {
    Iceberg(IcebergSchemaRef),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct TableStatistics(pub Arc<DataFrame>);

impl PartialEq for TableStatistics {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for TableStatistics {}

impl Hash for TableStatistics {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(Arc::as_ptr(&self.0) as *const () as usize);
    }
}

impl std::ops::Deref for TableStatistics {
    type Target = Arc<DataFrame>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Scan arguments shared across different scan types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct UnifiedScanArgs {
    /// User-provided schema of the file. Will be inferred during IR conversion
    /// if None.
    pub schema: Option<SchemaRef>,
    pub cloud_options: Option<CloudOptions>,
    pub hive_options: HiveOptions,

    pub rechunk: bool,
    pub cache: bool,
    pub glob: bool,
    /// Files with these prefixes will not be read.
    pub hidden_file_prefix: Option<Arc<[PlSmallStr]>>,

    pub projection: Option<Arc<[PlSmallStr]>>,
    pub column_mapping: Option<ColumnMapping>,
    /// Default values for missing columns.
    pub default_values: Option<DefaultFieldValues>,
    pub row_index: Option<RowIndex>,
    /// Slice applied before predicates
    pub pre_slice: Option<Slice>,

    pub cast_columns_policy: CastColumnsPolicy,
    pub missing_columns_policy: MissingColumnsPolicy,
    pub extra_columns_policy: ExtraColumnsPolicy,
    pub include_file_paths: Option<PlSmallStr>,

    pub deletion_files: Option<DeletionFilesList>,
    pub table_statistics: Option<TableStatistics>,
    /// Stores (physical, deleted) row counts of the table if known upfront (e.g. for Iceberg).
    /// This allows for row-count queries to succeed without scanning all files.
    ///
    /// Note, intentionally store u64 instead of IdxSize to avoid erroring if it's unused.
    pub row_count: Option<(u64, u64)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct PredicateFileSkip {
    /// If `true` the predicate can be skipped at runtime.
    pub no_residual_predicate: bool,
    /// Number of files before skipping
    pub original_len: usize,
}

impl UnifiedScanArgs {
    pub fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}

// Manual default, we have `glob: true` by default.
impl Default for UnifiedScanArgs {
    fn default() -> Self {
        Self {
            schema: None,
            cloud_options: None,
            hive_options: HiveOptions::new_enabled(),
            rechunk: false,
            cache: false,
            glob: true,
            hidden_file_prefix: None,
            projection: None,
            column_mapping: None,
            default_values: None,
            row_index: None,
            pre_slice: None,
            cast_columns_policy: CastColumnsPolicy::default(),
            missing_columns_policy: MissingColumnsPolicy::default(),
            extra_columns_policy: ExtraColumnsPolicy::default(),
            include_file_paths: None,
            deletion_files: None,
            table_statistics: None,
            row_count: None,
        }
    }
}

/// Manual impls of Eq/Hash, as some fields are `Arc<T>` where T does not have Eq/Hash. For these
/// fields we compare the pointer addresses instead.
mod _file_scan_eq_hash {
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    #[cfg(feature = "scan_lines")]
    use polars_utils::pl_str::PlSmallStr;

    use super::FileScanIR;

    impl Eq for FileScanIR {}

    impl Hash for FileScanIR {
        fn hash<H: Hasher>(&self, state: &mut H) {
            FileScanEqHashWrap::from(self).hash(state)
        }
    }

    impl PartialEq for FileScanIR {
        fn eq(&self, other: &Self) -> bool {
            FileScanEqHashWrap::from(self) == FileScanEqHashWrap::from(other)
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

        #[cfg(feature = "scan_lines")]
        Lines { name: &'a PlSmallStr },

        Anonymous {
            options: &'a crate::dsl::AnonymousScanOptions,
            function: usize,
        },

        /// Variant to ensure the lifetime is used regardless of feature gate combination.
        #[expect(unused)]
        Phantom(&'a ()),
    }

    impl<'a> From<&'a FileScanIR> for FileScanEqHashWrap<'a> {
        fn from(value: &'a FileScanIR) -> Self {
            match value {
                #[cfg(feature = "csv")]
                FileScanIR::Csv { options } => FileScanEqHashWrap::Csv { options },

                #[cfg(feature = "json")]
                FileScanIR::NDJson { options } => FileScanEqHashWrap::NDJson { options },

                #[cfg(feature = "parquet")]
                FileScanIR::Parquet { options, metadata } => FileScanEqHashWrap::Parquet {
                    options,
                    metadata: metadata.as_ref().map(arc_as_ptr),
                },

                #[cfg(feature = "ipc")]
                FileScanIR::Ipc { options, metadata } => FileScanEqHashWrap::Ipc {
                    options,
                    metadata: metadata.as_ref().map(arc_as_ptr),
                },

                #[cfg(feature = "python")]
                FileScanIR::PythonDataset {
                    dataset_object,
                    cached_ir,
                } => FileScanEqHashWrap::PythonDataset {
                    dataset_object: arc_as_ptr(dataset_object),
                    cached_ir: arc_as_ptr(cached_ir),
                },

                #[cfg(feature = "scan_lines")]
                FileScanIR::Lines { name } => FileScanEqHashWrap::Lines { name },

                FileScanIR::Anonymous { options, function } => FileScanEqHashWrap::Anonymous {
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

impl CastColumnsPolicy {
    /// Checks if casting can be done to a dtype with a configured policy.
    ///
    /// # Returns
    /// * Ok(true): Cast needed to target dtype
    /// * Ok(false): No casting needed
    /// * Err(_): Forbidden by configuration, or incompatible types.
    pub fn should_cast_column(
        &self,
        column_name: &str,
        target_dtype: &DataType,
        incoming_dtype: &DataType,
    ) -> PolarsResult<bool> {
        let mismatch_err = |hint: &str| {
            let hint_spacing = if hint.is_empty() { "" } else { ", " };

            polars_bail!(
                SchemaMismatch:
                "data type mismatch for column {}: incoming: {:?} != target: {:?}{}{}",
                column_name,
                incoming_dtype,
                target_dtype,
                hint_spacing,
                hint,
            )
        };

        if incoming_dtype.is_null() && !target_dtype.is_null() {
            return if self.null_upcast {
                Ok(true)
            } else {
                mismatch_err("unimplemented: 'null-upcast' in scan cast options")
            };
        }

        // We intercept the nested types first to prevent an expensive recursive eq - recursion
        // is instead done manually through this function.

        #[cfg(feature = "dtype-struct")]
        if let DataType::Struct(target_fields) = target_dtype {
            let DataType::Struct(incoming_fields) = incoming_dtype else {
                return mismatch_err("");
            };

            let incoming_fields_schema = PlHashMap::from_iter(
                incoming_fields
                    .iter()
                    .enumerate()
                    .map(|(i, fld)| (fld.name.as_str(), (i, &fld.dtype))),
            );

            let mut should_cast = incoming_fields.len() != target_fields.len();

            for (target_idx, target_field) in target_fields.iter().enumerate() {
                let Some((incoming_idx, incoming_field_dtype)) =
                    incoming_fields_schema.get(target_field.name().as_str())
                else {
                    match self.missing_struct_fields {
                        MissingColumnsPolicy::Raise => {
                            return mismatch_err(&format!(
                                "encountered missing struct field: {}, \
                                hint: pass cast_options=pl.ScanCastOptions(missing_struct_fields='insert')",
                                target_field.name(),
                            ));
                        },
                        MissingColumnsPolicy::Insert => {
                            should_cast = true;
                            // Must keep checking the rest of the fields.
                            continue;
                        },
                    };
                };

                // # Note
                // We also need to cast if the struct fields are out of order. Currently there is
                // no API parameter to control this - we always do this by default.
                should_cast |= *incoming_idx != target_idx;

                should_cast |= self.should_cast_column(
                    column_name,
                    &target_field.dtype,
                    incoming_field_dtype,
                )?;
            }

            // Casting is also needed if there are extra fields, check them here.

            // Take and re-use hashmap
            let mut target_fields_schema = incoming_fields_schema;
            target_fields_schema.clear();

            target_fields_schema.extend(
                target_fields
                    .iter()
                    .enumerate()
                    .map(|(i, fld)| (fld.name.as_str(), (i, &fld.dtype))),
            );

            for fld in incoming_fields {
                if !target_fields_schema.contains_key(fld.name.as_str()) {
                    match self.extra_struct_fields {
                        ExtraColumnsPolicy::Ignore => {
                            should_cast = true;
                            break;
                        },
                        ExtraColumnsPolicy::Raise => {
                            return mismatch_err(&format!(
                                "encountered extra struct field: {}, \
                                hint: specify this field in the schema, or pass \
                                cast_options=pl.ScanCastOptions(extra_struct_fields='ignore')",
                                &fld.name,
                            ));
                        },
                    }
                }
            }

            return Ok(should_cast);
        }

        if let DataType::List(target_inner) = target_dtype {
            let DataType::List(incoming_inner) = incoming_dtype else {
                return mismatch_err("");
            };

            return self.should_cast_column(column_name, target_inner, incoming_inner);
        }

        #[cfg(feature = "dtype-array")]
        if let DataType::Array(target_inner, target_width) = target_dtype {
            let DataType::Array(incoming_inner, incoming_width) = incoming_dtype else {
                return mismatch_err("");
            };

            if incoming_width != target_width {
                return mismatch_err("");
            }

            return self.should_cast_column(column_name, target_inner, incoming_inner);
        }

        // Eq here should be cheap as we have intercepted all nested types above.

        debug_assert!(!target_dtype.is_nested());

        // If we were to drop cast on an `Unknown` incoming_dtype, it could eventually
        // lead to dtype errors. The reason is that the logic used by type coercion differs
        // from the casting logic used by `materialize_unknown`.
        if incoming_dtype.contains_unknown() {
            return Ok(true);
        }

        // Note: Only call this with non-nested types for performance
        let materialize_unknown = |dtype: &DataType| -> std::borrow::Cow<DataType> {
            dtype
                .clone()
                .materialize_unknown(true)
                .map(std::borrow::Cow::Owned)
                .unwrap_or(std::borrow::Cow::Borrowed(incoming_dtype))
        };

        let incoming_dtype = std::borrow::Cow::Borrowed(incoming_dtype);
        let target_dtype = materialize_unknown(target_dtype);

        if target_dtype == incoming_dtype {
            return Ok(false);
        }

        let incoming_dtype = incoming_dtype.as_ref();
        let target_dtype = target_dtype.as_ref();

        //
        // After this point the dtypes are mismatching.
        //

        if target_dtype.is_integer() && incoming_dtype.is_integer() {
            if !self.integer_upcast {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(integer_cast='upcast')",
                );
            }

            return match get_numeric_upcast_supertype_lossless(incoming_dtype, target_dtype) {
                Some(ref v) if v == target_dtype => Ok(true),
                _ => mismatch_err("incoming dtype cannot safely cast to target dtype"),
            };
        }

        if target_dtype.is_float() && incoming_dtype.is_float() {
            return match (target_dtype, incoming_dtype) {
                (DataType::Float64, DataType::Float32)
                | (DataType::Float64, DataType::Float16)
                | (DataType::Float32, DataType::Float16) => {
                    if self.float_upcast {
                        Ok(true)
                    } else {
                        mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='upcast')",
                        )
                    }
                },

                (DataType::Float16, DataType::Float32)
                | (DataType::Float16, DataType::Float64)
                | (DataType::Float32, DataType::Float64) => {
                    if self.float_downcast {
                        Ok(true)
                    } else {
                        mismatch_err(
                            "hint: pass cast_options=pl.ScanCastOptions(float_cast='downcast')",
                        )
                    }
                },

                _ => unreachable!(),
            };
        }

        if target_dtype.is_float() && incoming_dtype.is_integer() {
            return if !self.integer_to_float_cast {
                mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(integer_cast='allow-float')",
                )
            } else {
                Ok(true)
            };
        }

        if let (
            DataType::Datetime(target_unit, target_zone),
            DataType::Datetime(incoming_unit, incoming_zone),
        ) = (target_dtype, incoming_dtype)
        {
            // Check timezone
            if !self.datetime_convert_timezone
                && !TimeZone::eq_none_as_utc(incoming_zone.as_ref(), target_zone.as_ref())
            {
                return mismatch_err(
                    "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='convert-timezone')",
                );
            }

            // Check unit
            if target_unit != incoming_unit {
                return match (incoming_unit, target_unit) {
                    (TimeUnit::Nanoseconds, _) => {
                        if self.datetime_nanoseconds_downcast {
                            Ok(true)
                        } else {
                            mismatch_err(
                                "hint: pass cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast')",
                            )
                        }
                    },

                    (TimeUnit::Microseconds, TimeUnit::Milliseconds) => {
                        if self.datetime_microseconds_downcast {
                            Ok(true)
                        } else {
                            // TODO
                            mismatch_err(
                                "unimplemented: 'microsecond-downcast' in scan cast options",
                            )
                        }
                    },

                    _ => mismatch_err(""),
                };
            }

            // Dtype differs and we are allowed to coerce
            return Ok(true);
        }

        mismatch_err("")
    }
}
