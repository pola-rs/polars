use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::PlHashSet;
use polars_core::schema::Schema;
use polars_io::cloud::CloudOptions;
use polars_io::metrics::IOMetrics;
use polars_io::utils::file::Writeable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;

use super::FileWriteFormat;
use crate::dsl::file_provider::FileProviderType;
use crate::dsl::{AExpr, Expr, SpecialEq};
use crate::plans::{ExprIR, ToFieldContext};
use crate::prelude::PlanCallback;

type DynSinkTarget = SpecialEq<Arc<std::sync::Mutex<Option<Writeable>>>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct UnifiedSinkArgs {
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub cloud_options: Option<Arc<CloudOptions>>,
}

impl Default for UnifiedSinkArgs {
    fn default() -> Self {
        Self {
            mkdir: false,
            maintain_order: true,
            sync_on_close: SyncOnCloseType::None,
            cloud_options: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SinkDestination {
    File {
        target: SinkTarget,
    },
    Partitioned {
        base_path: PlRefPath,
        file_path_provider: Option<FileProviderType>,
        partition_strategy: PartitionStrategy,
        max_rows_per_file: IdxSize,
        approximate_bytes_per_file: u64,
    },
}

impl SinkDestination {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::File { target } => target.cloud_scheme(),
            Self::Partitioned { base_path, .. } => base_path.scheme(),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum SinkTarget {
    Path(PlRefPath),
    Dyn(DynSinkTarget),
}

impl SinkTarget {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            SinkTarget::Path(p) => CloudScheme::from_path(p.as_str()),
            SinkTarget::Dyn(_) => None,
        }
    }

    pub fn open_into_writeable(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
        cloud_upload_chunk_size: usize,
        cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Writeable> {
        match self {
            SinkTarget::Path(path) => {
                if mkdir {
                    polars_io::utils::mkdir::mkdir_recursive(path)?;
                }

                polars_io::utils::file::Writeable::try_new(
                    path.clone(),
                    cloud_options,
                    cloud_upload_chunk_size,
                    cloud_upload_concurrency,
                    io_metrics,
                )
            },
            SinkTarget::Dyn(memory_writer) => Ok(memory_writer.lock().unwrap().take().unwrap()),
        }
    }

    pub async fn open_into_writeable_async(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
        cloud_upload_chunk_size: usize,
        cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Writeable> {
        #[cfg(feature = "cloud")]
        {
            match self {
                SinkTarget::Path(path) => {
                    if mkdir {
                        polars_io::utils::mkdir::tokio_mkdir_recursive(path).await?;
                    }

                    polars_io::utils::file::Writeable::try_new(
                        path.clone(),
                        cloud_options,
                        cloud_upload_chunk_size,
                        cloud_upload_concurrency,
                        io_metrics,
                    )
                },
                SinkTarget::Dyn(memory_writer) => Ok(memory_writer.lock().unwrap().take().unwrap()),
            }
        }

        #[cfg(not(feature = "cloud"))]
        {
            self.open_into_writeable(
                cloud_options,
                mkdir,
                cloud_upload_chunk_size,
                cloud_upload_concurrency,
                io_metrics,
            )
        }
    }

    pub fn to_display_string(&self) -> String {
        match self {
            Self::Path(p) => p.to_string(),
            Self::Dyn(_) => "dynamic-target".to_string(),
        }
    }
}

impl fmt::Debug for SinkTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SinkTarget::")?;
        match self {
            Self::Path(p) => write!(f, "Path({p:?})"),
            Self::Dyn(_) => f.write_str("Dyn"),
        }
    }
}

impl std::hash::Hash for SinkTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Path(p) => p.hash(state),
            Self::Dyn(p) => Arc::as_ptr(p).hash(state),
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SinkTarget {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Path(p) => p.serialize(serializer),
            Self::Dyn(_) => Err(serde::ser::Error::custom(
                "cannot serialize in-memory sink target",
            )),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SinkTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self::Path(PlRefPath::deserialize(deserializer)?))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for SinkTarget {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SinkTarget".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "SinkTarget"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        PlRefPath::json_schema(generator)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkType {
    Memory,
    Callback(CallbackSinkType),
    File(FileSinkOptions),
    Partitioned(PartitionedSinkOptions),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct CallbackSinkType {
    pub function: PlanCallback<DataFrame, bool>,
    pub maintain_order: bool,
    pub chunk_size: Option<NonZeroUsize>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkTypeIR {
    /// In-memory DataFrame
    Memory,
    /// Callback function (e.g. Python `collect_batches()`).
    Callback(CallbackSinkType),
    /// Single file
    File(FileSinkOptions),
    /// Multiple files
    #[cfg_attr(all(feature = "serde", not(feature = "ir_serde")), serde(skip))]
    Partitioned(PartitionedSinkOptionsIR),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptions {
    pub base_path: PlRefPath,
    pub file_path_provider: Option<FileProviderType>,
    pub partition_strategy: PartitionStrategy,
    pub file_format: FileWriteFormat,
    pub unified_sink_args: UnifiedSinkArgs,
    pub max_rows_per_file: IdxSize,
    pub approximate_bytes_per_file: u64,
}

impl PartitionedSinkOptions {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_path(self.base_path.as_str())
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategy {
    Keyed {
        keys: Vec<Expr>,
        include_keys: bool,
        keys_pre_grouped: bool,
    },
    /// Split the size of the input stream into chunks.
    ///
    /// Semantically equivalent to a 0-key partition by.
    FileSize,
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, strum_macros::IntoStaticStr)]
pub enum PartitionStrategyIR {
    Keyed {
        keys: Vec<ExprIR>,
        include_keys: bool,
        keys_pre_grouped: bool,
    },
    /// Split the size of the input stream into chunks.
    ///
    /// Semantically equivalent to a 0-key partition by.
    FileSize,
}

#[cfg(feature = "cse")]
impl PartitionStrategyIR {
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Keyed {
                keys,
                include_keys,
                keys_pre_grouped,
            } => {
                for k in keys {
                    k.traverse_and_hash(expr_arena, state);
                }

                include_keys.hash(state);
                keys_pre_grouped.hash(state);
            },
            Self::FileSize => {},
        }
    }
}

impl SinkTypeIR {
    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Memory => {},
            Self::Callback(f) => f.hash(state),
            Self::File(options) => options.hash(state),
            Self::Partitioned(options) => options.traverse_and_hash(expr_arena, state),
        }
    }
}

impl SinkTypeIR {
    pub fn maintain_order(&self) -> bool {
        match self {
            SinkTypeIR::Memory => true,
            SinkTypeIR::Callback(s) => s.maintain_order,
            SinkTypeIR::File(FileSinkOptions {
                unified_sink_args, ..
            })
            | SinkTypeIR::Partitioned(PartitionedSinkOptionsIR {
                unified_sink_args, ..
            }) => unified_sink_args.maintain_order,
        }
    }
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptionsIR {
    pub base_path: PlRefPath,
    pub file_path_provider: FileProviderType,
    pub partition_strategy: PartitionStrategyIR,
    pub file_format: FileWriteFormat,
    pub unified_sink_args: UnifiedSinkArgs,
    pub max_rows_per_file: IdxSize,
    pub approximate_bytes_per_file: u64,
}

impl PartitionedSinkOptionsIR {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_path(self.base_path.as_str())
    }

    pub fn expr_irs_iter(&self) -> impl ExactSizeIterator<Item = &ExprIR> {
        match &self.partition_strategy {
            PartitionStrategyIR::Keyed {
                keys,
                include_keys: _,
                keys_pre_grouped: _,
            } => keys.iter(),
            PartitionStrategyIR::FileSize => [][..].iter(),
        }
    }

    pub fn file_output_schema<'a>(
        &self,
        input_schema: &'a Schema,
        expr_arena: &Arena<AExpr>,
    ) -> PolarsResult<Cow<'a, Schema>> {
        Ok(match &self.partition_strategy {
            PartitionStrategyIR::Keyed {
                keys,
                include_keys,
                keys_pre_grouped: _,
            } => {
                if keys.is_empty() {
                    Cow::Borrowed(input_schema)
                } else if !include_keys {
                    let key_output_names: PlHashSet<&PlSmallStr> =
                        keys.iter().map(|e| e.output_name()).collect();

                    Cow::Owned(
                        input_schema
                            .iter()
                            .filter(|(name, _)| !key_output_names.contains(*name))
                            .map(|(name, dtype)| (name.clone(), dtype.clone()))
                            .collect(),
                    )
                } else {
                    let mut out = input_schema.clone();

                    for e in keys {
                        out.with_column(
                            e.output_name().clone(),
                            expr_arena
                                .get(e.node())
                                .to_dtype(&ToFieldContext::new(expr_arena, input_schema))?,
                        );
                    }

                    Cow::Owned(out)
                }
            },
            PartitionStrategyIR::FileSize => Cow::Borrowed(input_schema),
        })
    }

    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        let PartitionedSinkOptionsIR {
            base_path,
            file_path_provider,
            partition_strategy,
            file_format,
            unified_sink_args,
            max_rows_per_file,
            approximate_bytes_per_file,
        } = self;

        base_path.hash(state);
        file_path_provider.hash(state);
        partition_strategy.traverse_and_hash(expr_arena, state);
        file_format.hash(state);
        unified_sink_args.hash(state);
        max_rows_per_file.hash(state);
        approximate_bytes_per_file.hash(state);
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct FileSinkOptions {
    pub target: SinkTarget,
    pub file_format: FileWriteFormat,
    pub unified_sink_args: UnifiedSinkArgs,
}
