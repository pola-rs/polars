use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::DataType;
use polars_core::scalar::Scalar;
use polars_io::cloud::CloudOptions;
use polars_io::utils::file::{DynWriteable, Writeable};
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use super::{ExprIR, FileType};
use crate::dsl::{AExpr, Expr, SpecialEq};

/// Options that apply to all sinks.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    Path(Arc<PathBuf>),
    Dyn(DynSinkTarget),
}

impl SinkTarget {
    pub fn open_into_writeable(
        &self,
        sink_options: &SinkOptions,
        cloud_options: Option<&CloudOptions>,
    ) -> PolarsResult<Writeable> {
        match self {
            SinkTarget::Path(path) => {
                if sink_options.mkdir {
                    polars_io::utils::mkdir::mkdir_recursive(path.as_path())?;
                }

                let path = path.as_ref().display().to_string();
                polars_io::utils::file::Writeable::try_new(&path, cloud_options)
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
            SinkTarget::Path(path) => {
                if sink_options.mkdir {
                    polars_io::utils::mkdir::tokio_mkdir_recursive(path.as_path()).await?;
                }

                let path = path.as_ref().display().to_string();
                polars_io::utils::file::Writeable::try_new(&path, cloud_options)
            },
            SinkTarget::Dyn(memory_writer) => Ok(Writeable::Dyn(
                memory_writer.lock().unwrap().take().unwrap(),
            )),
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
        Ok(Self::Path(Arc::new(PathBuf::deserialize(deserializer)?)))
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub file_path: PathBuf,
    pub full_path: PathBuf,
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
    pub fn file_path(&self) -> &std::path::Path {
        self.file_path.as_path()
    }
    #[getter]
    pub fn full_path(&self) -> &std::path::Path {
        self.full_path.as_path()
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
    Rust(SpecialEq<Arc<dyn Fn(PartitionTargetContext) -> PolarsResult<SinkTarget> + Send + Sync>>),
    #[cfg(feature = "python")]
    Python(polars_utils::python_function::PythonFunction),
}

impl PartitionTargetCallback {
    pub fn call(&self, ctx: PartitionTargetContext) -> PolarsResult<SinkTarget> {
        match self {
            Self::Rust(f) => f(ctx),
            #[cfg(feature = "python")]
            Self::Python(f) => pyo3::Python::with_gil(|py| {
                let sink_target = f.call1(py, (ctx,))?;
                let converter =
                    polars_utils::python_convert_registry::get_python_convert_registry();
                let sink_target = (converter.from_py.sink_target)(sink_target)?;
                let sink_target = sink_target.downcast_ref::<SinkTarget>().unwrap().clone();
                PolarsResult::Ok(sink_target)
            }),
        }
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

        Err(S::Error::custom(format!("cannot serialize {:?}", self)))
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionSinkType {
    pub base_path: Arc<PathBuf>,
    pub file_path_cb: Option<PartitionTargetCallback>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariant,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionSinkTypeIR {
    pub base_path: Arc<PathBuf>,
    pub file_path_cb: Option<PartitionTargetCallback>,
    pub file_type: FileType,
    pub sink_options: SinkOptions,
    pub variant: PartitionVariantIR,
    pub cloud_options: Option<polars_io::cloud::CloudOptions>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum SinkType {
    Memory,
    File(FileSinkType),
    Partition(PartitionSinkType),
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl SinkTypeIR {
    #[cfg(feature = "cse")]
    pub(crate) fn traverse_and_hash<H: Hasher>(&self, expr_arena: &Arena<AExpr>, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Memory => {},
            Self::File(f) => f.hash(state),
            Self::Partition(f) => {
                f.file_type.hash(state);
                f.sink_options.hash(state);
                f.variant.traverse_and_hash(expr_arena, state);
                f.cloud_options.hash(state);
            },
        }
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
