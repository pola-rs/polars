use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::scalar::Scalar;
use polars_io::cloud::CloudOptions;
use polars_io::utils::file::Writeable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::{CloudScheme, PlPath};

use super::{ExprIR, FileType};
use crate::dsl::{AExpr, Expr, PartitionStrategy, PartitionStrategyIR, SpecialEq, UnifiedSinkArgs};
use crate::prelude::PlanCallback;

/// Options that apply to all sinks.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct SinkOptions {
    /// Call sync when closing the file.
    pub sync_on_close: SyncOnCloseType,

    /// The output file needs to maintain order of the data that comes in.
    pub maintain_order: bool,

    /// Recursively create all the directories in the path.
    pub mkdir: bool,
}

impl Default for SinkOptions {
    fn default() -> Self {
        Self {
            sync_on_close: Default::default(),
            maintain_order: true,
            mkdir: false,
        }
    }
}

type DynSinkTarget = SpecialEq<Arc<std::sync::Mutex<Option<Writeable>>>>;

#[derive(Clone, PartialEq, Eq)]
pub enum SinkTarget {
    Path(PlPath),
    Dyn(DynSinkTarget),
}

impl SinkTarget {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            SinkTarget::Path(p) => CloudScheme::from_uri(p.to_str()),
            SinkTarget::Dyn(_) => None,
        }
    }

    pub fn open_into_writeable(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
    ) -> PolarsResult<Writeable> {
        match self {
            SinkTarget::Path(path) => {
                if mkdir {
                    polars_io::utils::mkdir::mkdir_recursive(path.as_ref())?;
                }

                polars_io::utils::file::Writeable::try_new(path.as_ref(), cloud_options)
            },
            SinkTarget::Dyn(memory_writer) => Ok(memory_writer.lock().unwrap().take().unwrap()),
        }
    }

    #[cfg(feature = "cloud")]
    pub async fn open_into_writeable_async(
        &self,
        cloud_options: Option<&CloudOptions>,
        mkdir: bool,
    ) -> PolarsResult<Writeable> {
        #[cfg(feature = "cloud")]
        {
            match self {
                SinkTarget::Path(path) => {
                    if mkdir {
                        polars_io::utils::mkdir::tokio_mkdir_recursive(path.as_ref()).await?;
                    }

                    polars_io::utils::file::Writeable::try_new(path.as_ref(), cloud_options)
                },
                SinkTarget::Dyn(memory_writer) => Ok(memory_writer.lock().unwrap().take().unwrap()),
            }
        }

        #[cfg(not(feature = "cloud"))]
        {
            self.open_into_writeable(cloud_options, mkdir)
        }
    }

    pub fn to_display_string(&self) -> String {
        match self {
            Self::Path(p) => p.display().to_string(),
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
        Ok(Self::Path(PlPath::deserialize(deserializer)?))
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
        PathBuf::json_schema(generator)
    }
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

#[cfg_attr(feature = "python", pyo3::pyclass)]
#[derive(Clone)]
pub struct PartitionTargetContextKey {
    pub name: PlSmallStr,
    pub raw_value: Scalar,
}

#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct PartitionTargetContext {
    pub file_idx: usize,
    pub part_idx: usize,
    pub in_part_idx: usize,
    pub keys: Vec<PartitionTargetContextKey>,
    pub file_path: String,
    pub full_path: PlPath,
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PartitionTargetContext {
    #[getter]
    pub fn file_idx(&self) -> usize {
        self.file_idx
    }
    #[getter]
    pub fn part_idx(&self) -> usize {
        self.part_idx
    }
    #[getter]
    pub fn in_part_idx(&self) -> usize {
        self.in_part_idx
    }
    #[getter]
    pub fn keys(&self) -> Vec<PartitionTargetContextKey> {
        self.keys.clone()
    }
    #[getter]
    pub fn file_path(&self) -> &str {
        self.file_path.as_str()
    }
    #[getter]
    pub fn full_path(&self) -> &str {
        self.full_path.to_str()
    }
}
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl PartitionTargetContextKey {
    #[getter]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
    #[getter]
    pub fn str_value(&self) -> pyo3::PyResult<String> {
        let value = self
            .raw_value
            .clone()
            .into_series(PlSmallStr::EMPTY)
            .strict_cast(&DataType::String)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
        let value = value.str().unwrap();
        let value = value.get(0).unwrap_or("null").as_bytes();
        let value = percent_encoding::percent_encode(value, polars_io::utils::URL_ENCODE_CHAR_SET);
        Ok(value.to_string())
    }
    #[getter]
    pub fn raw_value(&self) -> pyo3::Py<pyo3::PyAny> {
        let converter = polars_core::chunked_array::object::registry::get_pyobject_converter();
        *(converter.as_ref())(self.raw_value.as_any_value())
            .downcast::<pyo3::Py<pyo3::PyAny>>()
            .unwrap()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PartitionTargetCallback {
    Rust(
        SpecialEq<
            Arc<
                dyn Fn(PartitionTargetContext) -> PolarsResult<PartitionTargetCallbackResult>
                    + Send
                    + Sync,
            >,
        >,
    ),
    #[cfg(feature = "python")]
    Python(polars_utils::python_function::PythonFunction),
}

impl Hash for PartitionTargetCallback {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Rust(v) => v.hash(state),
            #[cfg(feature = "python")]
            Self::Python(_) => {},
        }
    }
}

#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct SinkWritten {
    pub file_idx: usize,
    pub part_idx: usize,
    pub in_part_idx: usize,
    pub keys: Vec<PartitionTargetContextKey>,
    pub file_path: PathBuf,
    pub full_path: PathBuf,
    pub num_rows: usize,
    pub file_size: usize,
    pub gathered: Option<DataFrame>,
}

#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct SinkFinishContext {
    pub written: Vec<SinkWritten>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SinkFinishCallback {
    Rust(SpecialEq<Arc<dyn Fn(DataFrame) -> PolarsResult<()> + Send + Sync>>),
    #[cfg(feature = "python")]
    Python(polars_utils::python_function::PythonFunction),
}

impl Hash for SinkFinishCallback {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Rust(v) => v.hash(state),
            #[cfg(feature = "python")]
            Self::Python(_) => {},
        }
    }
}

impl SinkFinishCallback {
    pub fn call(&self, df: DataFrame) -> PolarsResult<()> {
        match self {
            Self::Rust(f) => f(df),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::attach(|py| {
                let converter =
                    polars_utils::python_convert_registry::get_python_convert_registry();
                let df = (converter.to_py.df)(Box::new(df) as Box<dyn std::any::Any>)?;
                f.call1(py, (df,))?;
                PolarsResult::Ok(())
            }),
        }
    }

    pub fn display_str(&self) -> PlSmallStr {
        match self {
            Self::Rust(_) => PlSmallStr::from_static("Rust(<dyn Fn>)"),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::attach(|py| {
                use polars_utils::format_pl_smallstr;
                use pyo3::intern;
                use pyo3::pybacked::PyBackedStr;

                let class_name: PyBackedStr = f
                    .getattr(py, intern!(py, "__class__"))
                    .unwrap()
                    .extract(py)
                    .unwrap();

                format_pl_smallstr!("Python({class_name})")
            }),
        }
    }
}

#[derive(Clone)]
pub enum PartitionTargetCallbackResult {
    Str(String),
    Dyn(DynSinkTarget),
}

impl PartitionTargetCallback {
    pub fn call(&self, ctx: PartitionTargetContext) -> PolarsResult<PartitionTargetCallbackResult> {
        match self {
            Self::Rust(f) => f(ctx),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::attach(|py| {
                let partition_target = f.call1(py, (ctx,))?;
                let converter =
                    polars_utils::python_convert_registry::get_python_convert_registry();
                let partition_target =
                    (converter.from_py.partition_target_cb_result)(partition_target)?;
                let partition_target = partition_target
                    .downcast_ref::<PartitionTargetCallbackResult>()
                    .unwrap()
                    .clone();
                PolarsResult::Ok(partition_target)
            }),
        }
    }

    pub fn display_str(&self) -> PlSmallStr {
        match self {
            Self::Rust(_) => PlSmallStr::from_static("Rust(<dyn Fn>)"),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::attach(|py| {
                use polars_utils::format_pl_smallstr;
                use pyo3::intern;
                use pyo3::pybacked::PyBackedStr;

                let class_name: PyBackedStr = f
                    .getattr(py, intern!(py, "__class__"))
                    .unwrap()
                    .extract(py)
                    .unwrap();

                format_pl_smallstr!("Python({class_name})")
            }),
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for SinkFinishCallback {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;

        #[cfg(feature = "python")]
        if let Self::Python(v) = self {
            return v.serialize(_serializer);
        }

        Err(S::Error::custom(format!("cannot serialize {self:?}")))
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SinkFinishCallback {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[cfg(feature = "python")]
        {
            Ok(Self::Python(
                polars_utils::python_function::PythonFunction::deserialize(_deserializer)?,
            ))
        }
        #[cfg(not(feature = "python"))]
        {
            use serde::de::Error;
            Err(D::Error::custom(
                "cannot deserialize PartitionOutputCallback",
            ))
        }
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for SinkFinishCallback {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "PartitionTargetCallback".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "SinkFinishCallback"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PartitionTargetCallback {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[cfg(feature = "python")]
        {
            Ok(Self::Python(
                polars_utils::python_function::PythonFunction::deserialize(_deserializer)?,
            ))
        }
        #[cfg(not(feature = "python"))]
        {
            use serde::de::Error;
            Err(D::Error::custom(
                "cannot deserialize PartitionOutputCallback",
            ))
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for PartitionTargetCallback {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;

        #[cfg(feature = "python")]
        if let Self::Python(v) = self {
            return v.serialize(_serializer);
        }

        Err(S::Error::custom(format!("cannot serialize {self:?}")))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for PartitionTargetCallback {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "PartitionTargetCallback".into()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PartitionTargetCallback"))
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct SortColumn {
    pub expr: Expr,
    pub descending: bool,
    pub nulls_last: bool,
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct SortColumnIR {
    pub expr: ExprIR,
    pub descending: bool,
    pub nulls_last: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedSinkOptions {
    pub base_path: PlPath,
    pub file_path_provider: Option<PartitionTargetCallback>,
    pub partition_strategy: PartitionStrategy,
    /// TODO: Move this to UnifiedSinkArgs
    pub finish_callback: Option<SinkFinishCallback>,
    pub file_format: Box<FileType>,
    pub unified_sink_args: UnifiedSinkArgs,
}

impl PartitionedSinkOptions {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_uri(self.base_path.to_str())
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PartitionVariant {
    MaxSize(IdxSize),
    Parted {
        key_exprs: Vec<Expr>,
        include_key: bool,
    },
    ByKey {
        key_exprs: Vec<Expr>,
        include_key: bool,
    },
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, strum_macros::IntoStaticStr)]
pub enum PartitionVariantIR {
    MaxSize(IdxSize),
    Parted {
        key_exprs: Vec<ExprIR>,
        include_key: bool,
    },
    ByKey {
        key_exprs: Vec<ExprIR>,
        include_key: bool,
    },
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
    pub base_path: PlPath,
    pub file_path_provider: Option<PartitionTargetCallback>,
    pub partition_strategy: PartitionStrategyIR,
    /// TODO: Move this to UnifiedSinkArgs
    pub finish_callback: Option<SinkFinishCallback>,
    pub file_format: Box<FileType>,
    pub unified_sink_args: UnifiedSinkArgs,
}

impl PartitionedSinkOptionsIR {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        CloudScheme::from_uri(self.base_path.to_str())
    }

    pub fn expr_irs_iter(&self) -> impl ExactSizeIterator<Item = &ExprIR> {
        let mut partition_key_exprs: &[ExprIR] = &[];
        let sort_exprs: &[SortColumnIR];

        match &self.partition_strategy {
            PartitionStrategyIR::Keyed {
                keys,
                include_keys: _,
                keys_pre_grouped: _,
                per_partition_sort_by,
            } => {
                partition_key_exprs = keys.as_slice();
                sort_exprs = per_partition_sort_by.as_slice();
            },
            PartitionStrategyIR::MaxRowsPerFile {
                max_rows_per_file: _,
                per_file_sort_by,
            } => {
                sort_exprs = per_file_sort_by.as_slice();
            },
        };

        (0..partition_key_exprs.len() + sort_exprs.len()).map(|i| {
            if i < partition_key_exprs.len() {
                &partition_key_exprs[i]
            } else {
                &sort_exprs[i - partition_key_exprs.len()].expr
            }
        })
    }

    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        let PartitionedSinkOptionsIR {
            base_path,
            file_path_provider,
            partition_strategy,
            finish_callback,
            file_format,
            unified_sink_args,
        } = self;

        base_path.hash(state);
        file_path_provider.hash(state);
        partition_strategy.traverse_and_hash(expr_arena, state);
        finish_callback.hash(state);
        file_format.hash(state);
        unified_sink_args.hash(state);
    }
}

#[cfg(feature = "cse")]
impl SortColumnIR {
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        self.expr.traverse_and_hash(expr_arena, state);
        self.descending.hash(state);
        self.nulls_last.hash(state);
    }
}

impl PartitionVariantIR {
    #[cfg(feature = "cse")]
    #[allow(unused)] // TODO: Remove
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::MaxSize(size) => size.hash(state),
            Self::Parted {
                key_exprs,
                include_key,
            }
            | Self::ByKey {
                key_exprs,
                include_key,
            } => {
                include_key.hash(state);
                for key_expr in key_exprs.as_slice() {
                    key_expr.traverse_and_hash(expr_arena, state);
                }
            },
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct FileSinkOptions {
    pub target: SinkTarget,
    pub file_format: Box<FileType>,
    pub unified_sink_args: UnifiedSinkArgs,
}
