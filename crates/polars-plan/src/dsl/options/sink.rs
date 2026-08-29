use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::PlIndexSet;
use polars_core::schema::Schema;
use polars_error::feature_gated;
use polars_io::cloud::CloudOptions;
use polars_io::metrics::IOMetrics;
use polars_io::utils::file::Writable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::itertools::Itertools;
use polars_utils::pl_path::{CloudScheme, PlRefPath};
use polars_utils::pl_str::PlSmallStr;

use super::FileWriteFormat;
use crate::dsl::file_provider::FileProviderType;
use crate::dsl::iceberg_sink_state::IcebergSinkState;
use crate::dsl::{AExpr, Expr, SpecialEq};
#[cfg(feature = "cse")]
use crate::plans::ExpressionHasher;
use crate::plans::{ExprIR, ExpressionComparator, ToFieldContext};
use crate::prelude::PlanCallback;

type DynSinkTarget = SpecialEq<Arc<std::sync::Mutex<Option<Writable>>>>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct UnifiedSinkArgs {
    pub mkdir: bool,
    pub maintain_order: bool,
    pub sync_on_close: SyncOnCloseType,
    pub cloud_options: Option<Arc<CloudOptions>>,
    pub sinked_paths_callback: Option<SinkedPathsCallback>,
}

impl Default for UnifiedSinkArgs {
    fn default() -> Self {
        Self {
            mkdir: false,
            maintain_order: true,
            sync_on_close: SyncOnCloseType::None,
            cloud_options: None,
            sinked_paths_callback: None,
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

    pub fn open_into_writable(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
        cloud_upload_chunk_size: Option<NonZeroUsize>,
        cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Writable> {
        match self {
            SinkTarget::Path(path) => {
                if mkdir {
                    polars_io::utils::mkdir::mkdir_recursive(path)?;
                }

                polars_io::utils::file::Writable::try_new(
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

    pub async fn open_into_writable_async(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
        cloud_upload_chunk_size: Option<NonZeroUsize>,
        cloud_upload_concurrency: usize,
        io_metrics: Option<Arc<IOMetrics>>,
    ) -> PolarsResult<Writable> {
        #[cfg(feature = "cloud")]
        {
            match self {
                SinkTarget::Path(path) => {
                    if mkdir {
                        polars_io::utils::mkdir::tokio_mkdir_recursive(path).await?;
                    }

                    polars_io::utils::file::Writable::try_new(
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
            self.open_into_writable(
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
    Iceberg(IcebergSinkState),
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl PartitionStrategyIR {
    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        match self {
            Self::Keyed {
                keys: l_keys,
                include_keys: l_include_keys,
                keys_pre_grouped: l_keys_pre_grouped,
            } => {
                let Self::Keyed {
                    keys: r_keys,
                    include_keys: r_include_keys,
                    keys_pre_grouped: r_keys_pre_grouped,
                } = other
                else {
                    return false;
                };

                (l_keys
                    .iter()
                    .eq_by_(r_keys.iter(), |lhs, rhs| expr_cmp.equals(lhs, rhs)))
                    && l_include_keys == r_include_keys
                    && l_keys_pre_grouped == r_keys_pre_grouped
            },
            Self::FileSize => matches!(other, Self::FileSize),
        }
    }
}

#[cfg(feature = "cse")]
impl PartitionStrategyIR {
    pub(crate) fn shallow_hash<H: Hasher>(&self, state: &mut H, expr_hash: &impl ExpressionHasher) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Keyed {
                keys,
                include_keys,
                keys_pre_grouped,
            } => {
                for k in keys {
                    expr_hash.hash_expr(k, state);
                }

                include_keys.hash(state);
                keys_pre_grouped.hash(state);
            },
            Self::FileSize => {},
        }
    }
}

impl SinkTypeIR {
    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        match self {
            Self::Memory => matches!(other, Self::Memory),
            Self::Callback(lhs) => matches!(other, Self::Callback(rhs)
                if lhs == rhs),
            Self::File(lhs) => matches!(other, Self::File(rhs)
                if lhs == rhs),
            Self::Partitioned(lhs) => matches!(other, Self::Partitioned(rhs)
                if lhs.shallow_eq(rhs, expr_cmp)),
        }
    }

    #[cfg(feature = "cse")]
    pub(crate) fn shallow_hash<H: Hasher>(&self, state: &mut H, expr_hash: &impl ExpressionHasher) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Memory => {},
            Self::Callback(f) => f.hash(state),
            Self::File(options) => options.hash(state),
            Self::Partitioned(options) => options.shallow_hash(state, expr_hash),
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

    pub fn set_maintain_order(&mut self, maintain_order: bool) {
        match self {
            SinkTypeIR::Memory => {},
            SinkTypeIR::Callback(s) => s.maintain_order = maintain_order,
            SinkTypeIR::File(FileSinkOptions {
                unified_sink_args, ..
            })
            | SinkTypeIR::Partitioned(PartitionedSinkOptionsIR {
                unified_sink_args, ..
            }) => unified_sink_args.maintain_order = maintain_order,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub(crate) fn shallow_eq(&self, other: &Self, expr_cmp: &impl ExpressionComparator) -> bool {
        let Self {
            base_path,
            file_path_provider,
            partition_strategy,
            file_format,
            unified_sink_args,
            max_rows_per_file,
            approximate_bytes_per_file,
        } = self;

        *base_path == other.base_path
            && *file_path_provider == other.file_path_provider
            && partition_strategy.shallow_eq(&other.partition_strategy, expr_cmp)
            && *file_format == other.file_format
            && *unified_sink_args == other.unified_sink_args
            && *max_rows_per_file == other.max_rows_per_file
            && *approximate_bytes_per_file == other.approximate_bytes_per_file
    }

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
                    let key_output_names: PlIndexSet<&PlSmallStr> =
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
    pub(crate) fn shallow_hash<H: Hasher>(&self, state: &mut H, expr_hash: &impl ExpressionHasher) {
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
        partition_strategy.shallow_hash(state, expr_hash);
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum SinkedPathsCallback {
    IcebergCommit(Box<IcebergSinkState>),
    Callback(PlanCallback<SinkedPathsCallbackArgs, ()>),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct SinkedPathsCallbackArgs {
    pub path_info_list: Vec<SinkedPathInfo>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Hash, PartialEq)]
pub struct SinkedPathInfo {
    pub path: PlRefPath,
    pub num_rows: u64,
    pub num_bytes: u64,
}

impl SinkedPathsCallback {
    pub fn call(&self, args: SinkedPathsCallbackArgs) -> PolarsResult<()> {
        use PlanCallback as CB;

        match self {
            Self::IcebergCommit(sink_state) => {
                feature_gated!("python", {
                    use pyo3::Python;

                    Python::attach(|py| {
                        use pyo3::intern;
                        use pyo3::types::PyList;

                        let py_paths = PyList::empty(py);

                        let SinkedPathsCallbackArgs { path_info_list } = args;

                        for SinkedPathInfo {
                            path,
                            num_rows: _,
                            num_bytes: _,
                        } in path_info_list
                        {
                            use pyo3::types::PyListMethods;

                            let path: &str = path.as_str();

                            py_paths.append(path)?;
                        }

                        sink_state
                            .as_ref()
                            .clone()
                            .into_sink_state_obj()?
                            .call_method1(py, intern!(py, "commit"), (py_paths,))?;

                        PolarsResult::Ok(())
                    })
                })
            },
            Self::Callback(CB::Rust(func)) => (func)(args),
            #[cfg(feature = "python")]
            Self::Callback(CB::Python(object)) => pyo3::Python::attach(|py| {
                use pyo3::intern;
                use pyo3::types::{PyAnyMethods, PyDict, PyList};

                let SinkedPathsCallbackArgs { path_info_list } = args;

                let py_sinked_paths_list = PyList::empty(py);

                let sinked_path_dataclass_cls =
                    polars_utils::python_convert_registry::get_python_convert_registry()
                        .py_sinked_path_dataclass(py);

                for SinkedPathInfo {
                    path,
                    num_rows,
                    num_bytes,
                } in path_info_list
                {
                    use pyo3::types::PyListMethods;

                    let path: &str = path.as_str();

                    let kwargs = PyDict::new(py);
                    kwargs.set_item(intern!(py, "path"), path)?;
                    kwargs.set_item(intern!(py, "num_bytes"), num_bytes)?;
                    kwargs.set_item(intern!(py, "num_rows"), num_rows)?;
                    py_sinked_paths_list.append(sinked_path_dataclass_cls.call(
                        py,
                        (),
                        Some(&kwargs),
                    )?)?;
                }

                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "paths"), py_sinked_paths_list)?;

                let args_dataclass =
                    polars_utils::python_convert_registry::get_python_convert_registry()
                        .py_sinked_paths_callback_args_dataclass(py)
                        .call(py, (), Some(&kwargs))?;

                object.call1(py, (args_dataclass,))?;

                Ok(())
            }),
        }
    }
}
