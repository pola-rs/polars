use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::frame::DataFrame;
use polars_core::prelude::DataType;
use polars_core::scalar::Scalar;
use polars_io::cloud::CloudOptions;
use polars_io::utils::file::{DynWriteable, Writeable};
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::PlPath;

use super::{ExprIR, FileType};
use crate::dsl::{AExpr, Expr, SpecialEq};

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

type DynSinkTarget = SpecialEq<Arc<std::sync::Mutex<Option<Box<dyn DynWriteable>>>>>;

#[derive(Clone, PartialEq, Eq)]
pub enum SinkTarget {
    Path(PlPath),
    Dyn(DynSinkTarget),
}

impl SinkTarget {
    pub fn open_into_writeable(
        &self,
        sink_options: &SinkOptions,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Writeable> {
        match self {
            SinkTarget::Path(addr) => {
                if sink_options.mkdir {
                    polars_io::utils::mkdir::mkdir_recursive(addr.as_ref())?;
                }

                polars_io::utils::file::Writeable::try_new(addr.as_ref(), cloud_options)
            },
            SinkTarget::Dyn(memory_writer) => Ok(Writeable::Dyn(
                memory_writer.lock().unwrap().take().unwrap(),
            )),
        }
    }

    #[cfg(not(feature = "cloud"))]
    pub async fn open_into_writeable_async(
        &self,
        sink_options: &SinkOptions,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Writeable> {
        self.open_into_writeable(sink_options, cloud_options)
    }

    #[cfg(feature = "cloud")]
    pub async fn open_into_writeable_async(
        &self,
        sink_options: &SinkOptions,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Writeable> {
        match self {
            SinkTarget::Path(addr) => {
                if sink_options.mkdir {
                    polars_io::utils::mkdir::tokio_mkdir_recursive(addr.as_ref()).await?;
                }

                polars_io::utils::file::Writeable::try_new(addr.as_ref(), cloud_options)
            },
            SinkTarget::Dyn(memory_writer) => Ok(Writeable::Dyn(
                memory_writer.lock().unwrap().take().unwrap(),
            )),
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
    fn schema_name() -> String {
        "SinkTarget".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "SinkTarget"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        PathBuf::json_schema(generator)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileSinkType {
    pub target: SinkTarget,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkTypeIR {
    Memory,
    File(FileSinkType),
    #[cfg_attr(all(feature = "serde", not(feature = "ir_serde")), serde(skip))]
    Partition(PartitionSinkTypeIR),
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
    pub fn raw_value(&self) -> pyo3::PyObject {
        let converter = polars_core::chunked_array::object::registry::get_pyobject_converter();
        *(converter.as_ref())(self.raw_value.as_any_value())
            .downcast::<pyo3::PyObject>()
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

impl SinkFinishCallback {
    pub fn call(&self, df: DataFrame) -> PolarsResult<()> {
        match self {
            Self::Rust(f) => f(df),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::with_gil(|py| {
                let converter =
                    polars_utils::python_convert_registry::get_python_convert_registry();
                let df = (converter.to_py.df)(Box::new(df) as Box<dyn std::any::Any>)?;
                f.call1(py, (df,))?;
                PolarsResult::Ok(())
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
            Self::Python(f) => pyo3::Python::with_gil(|py| {
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
    fn schema_name() -> String {
        "PartitionTargetCallback".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "SinkFinishCallback"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
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
    fn schema_name() -> String {
        "PartitionTargetCallback".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "PartitionTargetCallback"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
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
pub struct PartitionSinkType {
    pub base_path: Arc<PlPath>,
    pub file_path_cb: Option<PartitionTargetCallback>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariant,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
    pub per_partition_sort_by: Option<Vec<SortColumn>>,
    pub finish_callback: Option<SinkFinishCallback>,
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionSinkTypeIR {
    pub base_path: Arc<PlPath>,
    pub file_path_cb: Option<PartitionTargetCallback>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariantIR,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
    pub per_partition_sort_by: Option<Vec<SortColumnIR>>,
    pub finish_callback: Option<SinkFinishCallback>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkType {
    Memory,
    File(FileSinkType),
    Partition(PartitionSinkType),
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
#[derive(Clone, Debug, PartialEq, Eq)]
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

#[cfg(feature = "cse")]
impl SinkTypeIR {
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Memory => {},
            Self::File(f) => f.hash(state),
            Self::Partition(f) => f.traverse_and_hash(expr_arena, state),
        }
    }
}

impl SinkTypeIR {
    pub fn maintain_order(&self) -> bool {
        match self {
            SinkTypeIR::Memory => true,
            SinkTypeIR::File(s) => s.sink_options.maintain_order,
            SinkTypeIR::Partition(s) => s.sink_options.maintain_order,
        }
    }
}

#[cfg(feature = "cse")]
impl PartitionSinkTypeIR {
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        self.file_type.hash(state);
        self.sink_options.hash(state);
        self.variant.traverse_and_hash(expr_arena, state);
        self.cloud_options.hash(state);
        std::mem::discriminant(&self.per_partition_sort_by).hash(state);
        if let Some(v) = &self.per_partition_sort_by {
            v.len().hash(state);
            for v in v {
                v.traverse_and_hash(expr_arena, state);
            }
        }
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
#[derive(Clone, Debug)]
pub struct FileSinkOptions {
    pub path: Arc<PathBuf>,
    pub file_type: FileType,
}
