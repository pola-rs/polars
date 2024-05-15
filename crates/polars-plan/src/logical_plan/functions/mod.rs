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
use std::sync::Arc;

pub use dsl::*;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

#[cfg(feature = "python")]
use crate::dsl::python_udf::PythonFunction;
#[cfg(feature = "merge_sorted")]
use crate::logical_plan::functions::merge_sorted::merge_sorted;
use crate::prelude::*;

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FunctionNode {
    #[cfg(feature = "python")]
    OpaquePython {
        function: PythonFunction,
        schema: Option<SchemaRef>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        streamable: bool,
        validate_output: bool,
    },
    #[cfg_attr(feature = "serde", serde(skip))]
    Opaque {
        function: Arc<dyn DataFrameUdf>,
        schema: Option<Arc<dyn UdfSchema>>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        streamable: bool,
        // used for formatting
        #[cfg_attr(feature = "serde", serde(skip))]
        fmt_str: &'static str,
    },
    Count {
        paths: Arc<[PathBuf]>,
        scan_type: FileScan,
        alias: Option<Arc<str>>,
    },
    #[cfg_attr(feature = "serde", serde(skip))]
    Pipeline {
        function: Arc<dyn DataFrameUdfMut>,
        schema: SchemaRef,
        original: Option<Arc<DslPlan>>,
    },
    Unnest {
        columns: Arc<[Arc<str>]>,
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
        #[cfg_attr(feature = "serde", serde(skip))]
        schema: CachedSchema,
    },
    Explode {
        columns: Arc<[Arc<str>]>,
        #[cfg_attr(feature = "serde", serde(skip))]
        schema: CachedSchema,
    },
    Melt {
        args: Arc<MeltArgs>,
        #[cfg_attr(feature = "serde", serde(skip))]
        schema: CachedSchema,
    },
    RowIndex {
        name: Arc<str>,
        // Might be cached.
        #[cfg_attr(feature = "serde", serde(skip))]
        schema: CachedSchema,
        offset: Option<IdxSize>,
    },
}

impl Eq for FunctionNode {}

impl PartialEq for FunctionNode {
    fn eq(&self, other: &Self) -> bool {
        use FunctionNode::*;
        match (self, other) {
            (Rechunk, Rechunk) => true,
            (Count { paths: paths_l, .. }, Count { paths: paths_r, .. }) => paths_l == paths_r,
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
            (Melt { args: l, .. }, Melt { args: r, .. }) => l == r,
            (RowIndex { name: l, .. }, RowIndex { name: r, .. }) => l == r,
            #[cfg(feature = "merge_sorted")]
            (MergeSorted { column: l }, MergeSorted { column: r }) => l == r,
            _ => false,
        }
    }
}

impl Hash for FunctionNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "python")]
            FunctionNode::OpaquePython { .. } => {},
            FunctionNode::Opaque { fmt_str, .. } => fmt_str.hash(state),
            FunctionNode::Count {
                paths,
                scan_type,
                alias,
            } => {
                paths.hash(state);
                scan_type.hash(state);
                alias.hash(state);
            },
            FunctionNode::Pipeline { .. } => {},
            FunctionNode::Unnest { columns } => columns.hash(state),
            FunctionNode::Rechunk => {},
            #[cfg(feature = "merge_sorted")]
            FunctionNode::MergeSorted { column } => column.hash(state),
            FunctionNode::Rename {
                existing,
                new,
                swapping: _,
                ..
            } => {
                existing.hash(state);
                new.hash(state);
            },
            FunctionNode::Explode { columns, schema: _ } => columns.hash(state),
            FunctionNode::Melt { args, schema: _ } => args.hash(state),
            FunctionNode::RowIndex {
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

impl FunctionNode {
    /// Whether this function can run on batches of data at a time.
    pub fn is_streamable(&self) -> bool {
        use FunctionNode::*;
        match self {
            Rechunk | Pipeline { .. } => false,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => false,
            Count { .. } | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            Melt { args, .. } => args.streamable,
            Opaque { streamable, .. } => *streamable,
            #[cfg(feature = "python")]
            OpaquePython { streamable, .. } => *streamable,
            RowIndex { .. } => false,
        }
    }

    /// Whether this function will increase the number of rows
    pub fn expands_rows(&self) -> bool {
        use FunctionNode::*;
        match self {
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            Explode { .. } | Melt { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            #[cfg(feature = "python")]
            OpaquePython { predicate_pd, .. } => *predicate_pd,
            Rechunk | Unnest { .. } | Rename { .. } | Explode { .. } | Melt { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } | Count { .. } => false,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            #[cfg(feature = "python")]
            OpaquePython { projection_pd, .. } => *projection_pd,
            Rechunk
            | Count { .. }
            | Unnest { .. }
            | Rename { .. }
            | Explode { .. }
            | Melt { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } => true,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn additional_projection_pd_columns(&self) -> Cow<[Arc<str>]> {
        use FunctionNode::*;
        match self {
            Unnest { columns } => Cow::Borrowed(columns.as_ref()),
            Explode { columns, .. } => Cow::Borrowed(columns.as_ref()),
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column, .. } => Cow::Owned(vec![column.clone()]),
            _ => Cow::Borrowed(&[]),
        }
    }

    pub fn evaluate(&mut self, mut df: DataFrame) -> PolarsResult<DataFrame> {
        use FunctionNode::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            #[cfg(feature = "python")]
            OpaquePython {
                function,
                validate_output,
                schema,
                ..
            } => python_udf::call_python_udf(function, df, *validate_output, schema.as_deref()),
            Count {
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
                    Arc::get_mut(function).unwrap().call_udf(df)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    Arc::get_mut(function).unwrap().call_udf(df)
                }
            },
            Rename { existing, new, .. } => rename::rename_impl(df, existing, new),
            Explode { columns, .. } => df.explode(columns.as_ref()),
            Melt { args, .. } => {
                let args = (**args).clone();
                df.melt2(args)
            },
            RowIndex { name, offset, .. } => df.with_row_index(name.as_ref(), *offset),
        }
    }
}

impl Debug for FunctionNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for FunctionNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionNode::*;
        match self {
            Opaque { fmt_str, .. } => write!(f, "{fmt_str}"),
            #[cfg(feature = "python")]
            OpaquePython { .. } => write!(f, "python dataframe udf"),
            Rechunk => write!(f, "RECHUNK"),
            Count { .. } => write!(f, "FAST COUNT(*)"),
            Unnest { columns } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => write!(f, "MERGE SORTED"),
            Pipeline { original, .. } => {
                if let Some(original) = original {
                    let ir_plan = original.as_ref().clone().to_alp().unwrap();
                    let ir_display = ir_plan.display();

                    writeln!(f, "--- STREAMING")?;
                    write!(f, "{ir_display}")?;
                    let indent = 2;
                    write!(f, "{:indent$}--- END STREAMING", "")
                } else {
                    write!(f, "STREAMING")
                }
            },
            Rename { .. } => write!(f, "RENAME"),
            Explode { .. } => write!(f, "EXPLODE"),
            Melt { .. } => write!(f, "MELT"),
            RowIndex { .. } => write!(f, "WITH ROW INDEX"),
        }
    }
}
