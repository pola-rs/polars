mod count;
mod dsl;
#[cfg(feature = "merge_sorted")]
mod merge_sorted;
#[cfg(feature = "python")]
mod python_udf;
mod rename;
mod schema;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub use dsl::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;
use strum_macros::IntoStaticStr;

#[cfg(feature = "python")]
use crate::dsl::python_udf::PythonFunction;
#[cfg(feature = "merge_sorted")]
use crate::plans::functions::merge_sorted::merge_sorted;
use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[derive(Clone, IntoStaticStr)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionIR {
    #[cfg(feature = "python")]
    OpaquePython(OpaquePythonUdf),
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    Opaque {
        function: Arc<dyn DataFrameUdf>,
        schema: Option<Arc<dyn UdfSchema>>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        streamable: bool,
        // used for formatting
        fmt_str: String,
    },
    FastCount {
        paths: Arc<Vec<PathBuf>>,
        scan_type: FileScan,
        alias: Option<Arc<str>>,
    },
    /// Streaming engine pipeline
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    Pipeline {
        function: Arc<Mutex<dyn DataFrameUdfMut>>,
        schema: SchemaRef,
        original: Option<Arc<IRPlan>>,
    },
    Unnest {
        columns: Arc<[ColumnName]>,
    },
    Rechunk,
    // The two DataFrames are temporary concatenated
    // this indicates until which chunk the data is from the left df
    // this trick allows us to reuse the `Union` architecture to get map over
    // two DataFrames
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        // sorted column that serves as the key
        column: Arc<str>,
    },
    Rename {
        existing: Arc<[SmartString]>,
        new: Arc<[SmartString]>,
        // A column name gets swapped with an existing column
        swapping: bool,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    Explode {
        columns: Arc<[ColumnName]>,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    #[cfg(feature = "pivot")]
    Unpivot {
        args: Arc<UnpivotArgsIR>,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    RowIndex {
        name: Arc<str>,
        // Might be cached.
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
        offset: Option<IdxSize>,
    },
}

impl Eq for FunctionIR {}

impl PartialEq for FunctionIR {
    fn eq(&self, other: &Self) -> bool {
        use FunctionIR::*;
        match (self, other) {
            (Rechunk, Rechunk) => true,
            (FastCount { paths: paths_l, .. }, FastCount { paths: paths_r, .. }) => {
                paths_l == paths_r
            },
            (
                Rename {
                    existing: existing_l,
                    new: new_l,
                    ..
                },
                Rename {
                    existing: existing_r,
                    new: new_r,
                    ..
                },
            ) => existing_l == existing_r && new_l == new_r,
            (Explode { columns: l, .. }, Explode { columns: r, .. }) => l == r,
            #[cfg(feature = "pivot")]
            (Unpivot { args: l, .. }, Unpivot { args: r, .. }) => l == r,
            (RowIndex { name: l, .. }, RowIndex { name: r, .. }) => l == r,
            #[cfg(feature = "merge_sorted")]
            (MergeSorted { column: l }, MergeSorted { column: r }) => l == r,
            _ => false,
        }
    }
}

impl Hash for FunctionIR {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "python")]
            FunctionIR::OpaquePython { .. } => {},
            FunctionIR::Opaque { fmt_str, .. } => fmt_str.hash(state),
            FunctionIR::FastCount {
                paths,
                scan_type,
                alias,
            } => {
                paths.hash(state);
                scan_type.hash(state);
                alias.hash(state);
            },
            FunctionIR::Pipeline { .. } => {},
            FunctionIR::Unnest { columns } => columns.hash(state),
            FunctionIR::Rechunk => {},
            #[cfg(feature = "merge_sorted")]
            FunctionIR::MergeSorted { column } => column.hash(state),
            FunctionIR::Rename {
                existing,
                new,
                swapping: _,
                ..
            } => {
                existing.hash(state);
                new.hash(state);
            },
            FunctionIR::Explode { columns, schema: _ } => columns.hash(state),
            #[cfg(feature = "pivot")]
            FunctionIR::Unpivot { args, schema: _ } => args.hash(state),
            FunctionIR::RowIndex {
                name,
                schema: _,
                offset,
            } => {
                name.hash(state);
                offset.hash(state);
            },
        }
    }
}

impl FunctionIR {
    /// Whether this function can run on batches of data at a time.
    pub fn is_streamable(&self) -> bool {
        use FunctionIR::*;
        match self {
            Rechunk | Pipeline { .. } => false,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => false,
            FastCount { .. } | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Opaque { streamable, .. } => *streamable,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { streamable, .. }) => *streamable,
            RowIndex { .. } => false,
        }
    }

    /// Whether this function will increase the number of rows
    pub fn expands_rows(&self) -> bool {
        use FunctionIR::*;
        match self {
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Explode { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionIR::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { predicate_pd, .. }) => *predicate_pd,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Rechunk | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } | FastCount { .. } => false,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionIR::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { projection_pd, .. }) => *projection_pd,
            Rechunk | FastCount { .. } | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } => true,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn additional_projection_pd_columns(&self) -> Cow<[Arc<str>]> {
        use FunctionIR::*;
        match self {
            Unnest { columns } => Cow::Borrowed(columns.as_ref()),
            Explode { columns, .. } => Cow::Borrowed(columns.as_ref()),
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column, .. } => Cow::Owned(vec![column.clone()]),
            _ => Cow::Borrowed(&[]),
        }
    }

    pub fn evaluate(&self, mut df: DataFrame) -> PolarsResult<DataFrame> {
        use FunctionIR::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf {
                function,
                validate_output,
                schema,
                ..
            }) => python_udf::call_python_udf(function, df, *validate_output, schema.as_deref()),
            FastCount {
                paths, scan_type, ..
            } => count::count_rows(paths, scan_type),
            Rechunk => {
                df.as_single_chunk_par();
                Ok(df)
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column } => merge_sorted(&df, column.as_ref()),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    df.unnest(_columns.as_ref())
                }
                #[cfg(not(feature = "dtype-struct"))]
                {
                    panic!("activate feature 'dtype-struct'")
                }
            },
            Pipeline { function, .. } => {
                // we use a global string cache here as streaming chunks all have different rev maps
                #[cfg(feature = "dtype-categorical")]
                {
                    let _sc = StringCacheHolder::hold();
                    function.lock().unwrap().call_udf(df)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    function.lock().unwrap().call_udf(df)
                }
            },
            Rename { existing, new, .. } => rename::rename_impl(df, existing, new),
            Explode { columns, .. } => df.explode(columns.as_ref()),
            #[cfg(feature = "pivot")]
            Unpivot { args, .. } => {
                use polars_ops::pivot::UnpivotDF;
                let args = (**args).clone();
                df.unpivot2(args)
            },
            RowIndex { name, offset, .. } => df.with_row_index(name.as_ref(), *offset),
        }
    }

    pub fn to_streaming_lp(&self) -> Option<IRPlanRef> {
        let Self::Pipeline {
            function: _,
            schema: _,
            original,
        } = self
        else {
            return None;
        };

        Some(original.as_ref()?.as_ref().as_ref())
    }
}

impl Debug for FunctionIR {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for FunctionIR {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionIR::*;
        match self {
            Opaque { fmt_str, .. } => write!(f, "{fmt_str}"),
            Unnest { columns } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")
            },
            Pipeline { original, .. } => {
                if let Some(original) = original {
                    let ir_display = original.as_ref().display();

                    writeln!(f, "--- STREAMING")?;
                    write!(f, "{ir_display}")?;
                    let indent = 2;
                    write!(f, "{:indent$}--- END STREAMING", "")
                } else {
                    write!(f, "STREAMING")
                }
            },
            v => {
                let s: &str = v.into();
                write!(f, "{s}")
            },
        }
    }
}
