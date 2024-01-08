#[cfg(feature = "merge_sorted")]
mod merge_sorted;
#[cfg(feature = "python")]
mod python_udf;
mod rename;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use polars_core::prelude::*;
#[cfg(feature = "dtype-categorical")]
use polars_core::StringCacheHolder;
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
    #[cfg_attr(feature = "serde", serde(skip))]
    Pipeline {
        function: Arc<dyn DataFrameUdfMut>,
        schema: SchemaRef,
        original: Option<Arc<LogicalPlan>>,
    },
    Unnest {
        columns: Arc<[Arc<str>]>,
    },
    FastProjection {
        columns: Arc<[SmartString]>,
        duplicate_check: bool,
    },
    DropNulls {
        subset: Arc<[Arc<str>]>,
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
    },
    Explode {
        columns: Arc<[Arc<str>]>,
        schema: SchemaRef,
    },
    Melt {
        args: Arc<MeltArgs>,
        schema: SchemaRef,
    },
    RowCount {
        name: Arc<str>,
        schema: SchemaRef,
        offset: Option<IdxSize>,
    },
}

impl PartialEq for FunctionNode {
    fn eq(&self, other: &Self) -> bool {
        use FunctionNode::*;
        match (self, other) {
            (
                FastProjection {
                    columns: l,
                    duplicate_check: dl,
                },
                FastProjection {
                    columns: r,
                    duplicate_check: dr,
                },
            ) => l == r && dl == dr,
            (DropNulls { subset: l }, DropNulls { subset: r }) => l == r,
            (Rechunk, Rechunk) => true,
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
            (RowCount { name: l, .. }, RowCount { name: r, .. }) => l == r,
            _ => false,
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
            DropNulls { .. }
            | FastProjection { .. }
            | Unnest { .. }
            | Rename { .. }
            | Explode { .. } => true,
            Melt { args, .. } => args.streamable,
            Opaque { streamable, .. } => *streamable,
            #[cfg(feature = "python")]
            OpaquePython { streamable, .. } => *streamable,
            RowCount { .. } => false,
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

    pub(crate) fn schema<'a>(
        &self,
        input_schema: &'a SchemaRef,
    ) -> PolarsResult<Cow<'a, SchemaRef>> {
        use FunctionNode::*;
        match self {
            Opaque { schema, .. } => match schema {
                None => Ok(Cow::Borrowed(input_schema)),
                Some(schema_fn) => {
                    let output_schema = schema_fn.get_schema(input_schema)?;
                    Ok(Cow::Owned(output_schema))
                },
            },
            #[cfg(feature = "python")]
            OpaquePython { schema, .. } => Ok(schema
                .as_ref()
                .map(|schema| Cow::Owned(schema.clone()))
                .unwrap_or_else(|| Cow::Borrowed(input_schema))),
            Pipeline { schema, .. } => Ok(Cow::Owned(schema.clone())),
            FastProjection { columns, .. } => {
                let schema = columns
                    .iter()
                    .map(|name| {
                        let name = name.as_ref();
                        input_schema.try_get_field(name)
                    })
                    .collect::<PolarsResult<Schema>>()?;
                Ok(Cow::Owned(Arc::new(schema)))
            },
            DropNulls { .. } => Ok(Cow::Borrowed(input_schema)),
            Rechunk => Ok(Cow::Borrowed(input_schema)),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    let mut new_schema = Schema::with_capacity(input_schema.len() * 2);
                    for (name, dtype) in input_schema.iter() {
                        if _columns.iter().any(|item| item.as_ref() == name.as_str()) {
                            match dtype {
                                DataType::Struct(flds) => {
                                    for fld in flds {
                                        new_schema.with_column(
                                            fld.name().clone(),
                                            fld.data_type().clone(),
                                        );
                                    }
                                },
                                DataType::Unknown => {
                                    // pass through unknown
                                },
                                _ => {
                                    polars_bail!(
                                        SchemaMismatch: "expected struct dtype, got: `{}`", dtype
                                    );
                                },
                            }
                        } else {
                            new_schema.with_column(name.clone(), dtype.clone());
                        }
                    }

                    Ok(Cow::Owned(Arc::new(new_schema)))
                }
                #[cfg(not(feature = "dtype-struct"))]
                {
                    panic!("activate feature 'dtype-struct'")
                }
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => Ok(Cow::Borrowed(input_schema)),
            Rename { existing, new, .. } => rename::rename_schema(input_schema, existing, new),
            Explode { schema, .. } | RowCount { schema, .. } | Melt { schema, .. } => {
                Ok(Cow::Owned(schema.clone()))
            },
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            #[cfg(feature = "python")]
            OpaquePython { predicate_pd, .. } => *predicate_pd,
            FastProjection { .. }
            | DropNulls { .. }
            | Rechunk
            | Unnest { .. }
            | Rename { .. }
            | Explode { .. }
            | Melt { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowCount { .. } => false,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            #[cfg(feature = "python")]
            OpaquePython { projection_pd, .. } => *projection_pd,
            FastProjection { .. }
            | DropNulls { .. }
            | Rechunk
            | Unnest { .. }
            | Rename { .. }
            | Explode { .. }
            | Melt { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowCount { .. } => true,
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
            FastProjection {
                columns,
                duplicate_check,
            } => {
                if *duplicate_check {
                    df._select_impl(columns.as_ref())
                } else {
                    df._select_impl_unchecked(columns.as_ref())
                }
            },
            DropNulls { subset } => df.drop_nulls(Some(subset.as_ref())),
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
            RowCount { name, offset, .. } => df.with_row_index(name.as_ref(), *offset),
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
            FastProjection { columns, .. } => {
                write!(f, "FAST_PROJECT: ")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")
            },
            DropNulls { subset } => {
                write!(f, "DROP_NULLS by: ")?;
                let subset = subset.as_ref();
                fmt_column_delimited(f, subset, "[", "]")
            },
            Rechunk => write!(f, "RECHUNK"),
            Unnest { columns } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => write!(f, "MERGE SORTED"),
            Pipeline { original, .. } => {
                if let Some(original) = original {
                    writeln!(f, "--- STREAMING")?;
                    write!(f, "{:?}", original.as_ref())?;
                    let indent = 2;
                    writeln!(f, "{:indent$}--- END STREAMING", "")
                } else {
                    writeln!(f, "STREAMING")
                }
            },
            Rename { .. } => write!(f, "RENAME"),
            Explode { .. } => write!(f, "EXPLODE"),
            Melt { .. } => write!(f, "MELT"),
            RowCount { .. } => write!(f, "WITH ROW COUNT"),
        }
    }
}
