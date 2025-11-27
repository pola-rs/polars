use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_error::PolarsResult;
use polars_io::cloud::CloudOptions;
use polars_io::utils::file::Writeable;
use polars_io::utils::sync_on_close::SyncOnCloseType;
use polars_utils::IdxSize;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::{CloudScheme, PlPath};

use super::Expr;
use super::sink::*;
use crate::plans::{AExpr, ExprIR};
use crate::prelude::PlanCallback;

#[derive(Clone, Debug, PartialEq)]
pub enum SinkDestination {
    File {
        target: SinkTarget,
    },
    Partitioned {
        base_path: PlPath,
        file_path_provider: Option<FileProviderType>,
        partition_strategy: PartitionStrategy,
        /// TODO: Remove
        finish_callback: Option<SinkFinishCallback>,
        max_rows_per_file: Option<IdxSize>,
        approximate_bytes_per_file: Option<u64>,
    },
}

impl SinkDestination {
    pub fn cloud_scheme(&self) -> Option<CloudScheme> {
        match self {
            Self::File { target } => target.cloud_scheme(),
            Self::Partitioned { base_path, .. } => base_path.cloud_scheme(),
        }
    }
}

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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategy {
    Keyed {
        keys: Vec<Expr>,
        include_keys: bool,
        keys_pre_grouped: bool,
        per_partition_sort_by: Vec<SortColumn>,
    },
    /// Split the size of the input stream into chunks.
    ///
    /// Semantically equivalent to a 0-key partition by.
    FileSize,
}

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionStrategyIR {
    Keyed {
        keys: Vec<ExprIR>,
        include_keys: bool,
        keys_pre_grouped: bool,
        per_partition_sort_by: Vec<SortColumnIR>,
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
                per_partition_sort_by,
            } => {
                for k in keys {
                    k.traverse_and_hash(expr_arena, state);
                }

                include_keys.hash(state);
                keys_pre_grouped.hash(state);

                for x in per_partition_sort_by {
                    x.traverse_and_hash(expr_arena, state);
                }
            },
            Self::FileSize => {},
        }
    }
}

#[derive(Debug)]
pub struct FileProviderArgs {
    pub index_in_partition: usize,
    pub partition_keys: Arc<DataFrame>,
}

pub enum FileProviderReturn {
    Path(String),
    Writeable(Writeable),
}

pub type FileProviderFunction = PlanCallback<FileProviderArgs, FileProviderReturn>;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum FileProviderType {
    Hive { extension: PlSmallStr },
    Function(FileProviderFunction),
    Legacy(PartitionTargetCallback),
}

impl FileProviderType {
    pub fn verbose_print_display(&self) -> impl std::fmt::Display + '_ {
        struct VerbosePrintDisplay<'a>(&'a FileProviderType);

        impl std::fmt::Display for VerbosePrintDisplay<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use FileProviderType as P;
                match &self.0 {
                    P::Hive { extension } => write!(f, "Hive {{ extension: {} }}", extension),
                    P::Function(FileProviderFunction::Python(_)) => write!(f, "PythonFileProvider"),
                    P::Function(FileProviderFunction::Rust(_)) => write!(f, "RustFileProvider"),
                    P::Legacy(_) => write!(f, "Legacy"),
                }
            }
        }

        VerbosePrintDisplay(self)
    }
}

impl FileProviderFunction {
    pub fn call(&self, args: FileProviderArgs) -> PolarsResult<FileProviderReturn> {
        match self {
            Self::Rust(func) => (func)(args),
            #[cfg(feature = "python")]
            Self::Python(object) => pyo3::Python::attach(|py| {
                use polars_error::PolarsError;
                use pyo3::intern;
                use pyo3::types::{PyAnyMethods, PyDict};

                let FileProviderArgs {
                    index_in_partition,
                    partition_keys,
                } = args;

                let convert_registry =
                    polars_utils::python_convert_registry::get_python_convert_registry();

                let partition_keys = convert_registry
                    .to_py
                    .df_to_wrapped_pydf(partition_keys.as_ref())
                    .map_err(PolarsError::from)?;

                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "index_in_partition"), index_in_partition)?;
                kwargs.set_item(intern!(py, "partition_keys"), partition_keys)?;

                let args_dataclass = convert_registry.py_file_provider_args_dataclass().call(
                    py,
                    (),
                    Some(&kwargs),
                )?;

                let out = object.call1(py, (args_dataclass,))?;
                let out = (convert_registry.from_py.file_provider_result)(out)?;
                let out: FileProviderReturn = *out.downcast().unwrap();

                PolarsResult::Ok(out)
            }),
        }
    }
}
