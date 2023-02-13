mod drop;
#[cfg(feature = "merge_sorted")]
mod merge_sorted;
mod rename;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use polars_core::prelude::*;
#[cfg(feature = "dtype-categorical")]
use polars_core::IUseStringCache;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "merge_sorted")]
use crate::logical_plan::functions::merge_sorted::merge_sorted;
use crate::prelude::*;

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FunctionNode {
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
        columns: Arc<Vec<Arc<str>>>,
    },
    FastProjection {
        columns: Arc<Vec<Arc<str>>>,
    },
    DropNulls {
        subset: Arc<Vec<String>>,
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
        existing: Arc<Vec<String>>,
        new: Arc<Vec<String>>,
        // A column name gets swapped with an existing column
        swapping: bool,
    },
    Drop {
        names: Arc<Vec<String>>,
    },
}

impl PartialEq for FunctionNode {
    fn eq(&self, other: &Self) -> bool {
        use FunctionNode::*;
        match (self, other) {
            (FastProjection { columns: l }, FastProjection { columns: r }) => l == r,
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
            (Drop { names: l }, Drop { names: r }) => l == r,
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
            | Drop { .. } => true,
            Opaque { streamable, .. } => *streamable,
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
                }
            },
            Pipeline { schema, .. } => Ok(Cow::Owned(schema.clone())),
            FastProjection { columns } => {
                let schema = columns
                    .iter()
                    .map(|name| {
                        let name = name.as_ref();
                        input_schema.get_field(name).ok_or_else(|| {
                            PolarsError::SchemaFieldNotFound(name.to_string().into())
                        })
                    })
                    .collect::<PolarsResult<Schema>>()?;
                Ok(Cow::Owned(Arc::new(schema)))
            }
            DropNulls { .. } => Ok(Cow::Borrowed(input_schema)),
            Rechunk => Ok(Cow::Borrowed(input_schema)),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    let mut new_schema = Schema::with_capacity(input_schema.len() * 2);
                    for (name, dtype) in input_schema.iter() {
                        if _columns.iter().any(|item| item.as_ref() == name.as_str()) {
                            if let DataType::Struct(flds) = dtype {
                                for fld in flds {
                                    new_schema
                                        .with_column(fld.name().clone(), fld.data_type().clone());
                                }
                            } else {
                                return Err(PolarsError::ComputeError(
                                    format!("expected struct dtype, got: '{dtype:?}'").into(),
                                ));
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
            }
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => Ok(Cow::Borrowed(input_schema)),
            Rename { existing, new, .. } => rename::rename_schema(input_schema, existing, new),
            Drop { names } => drop::drop_schema(input_schema, names),
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            FastProjection { .. }
            | DropNulls { .. }
            | Rechunk
            | Unnest { .. }
            | Rename { .. }
            | Drop { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            FastProjection { .. }
            | DropNulls { .. }
            | Rechunk
            | Unnest { .. }
            | Rename { .. }
            | Drop { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn additional_projection_pd_columns(&self) -> &[Arc<str>] {
        use FunctionNode::*;
        match self {
            Unnest { columns } => columns.as_slice(),
            _ => &[],
        }
    }

    pub fn evaluate(&mut self, mut df: DataFrame) -> PolarsResult<DataFrame> {
        use FunctionNode::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            FastProjection { columns } => df.select(columns.as_slice()),
            DropNulls { subset } => df.drop_nulls(Some(subset.as_slice())),
            Rechunk => {
                df.as_single_chunk_par();
                Ok(df)
            }
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column } => merge_sorted(&df, column.as_ref()),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    df.unnest(_columns.as_slice())
                }
                #[cfg(not(feature = "dtype-struct"))]
                {
                    panic!("activate feature 'dtype-struct'")
                }
            }
            Pipeline { function, .. } => {
                // we use a global string cache here as streaming chunks all have different rev maps
                #[cfg(feature = "dtype-categorical")]
                {
                    let _hold = IUseStringCache::new();
                    Arc::get_mut(function).unwrap().call_udf(df)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    Arc::get_mut(function).unwrap().call_udf(df)
                }
            }
            Rename { existing, new, .. } => rename::rename_impl(df, existing, new),
            Drop { names } => drop::drop_impl(df, names),
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
            FastProjection { columns } => {
                write!(f, "FAST_PROJECT: ")?;
                let columns = columns.as_slice();
                fmt_column_delimited(f, columns, "[", "]")
            }
            DropNulls { subset } => {
                write!(f, "DROP_NULLS by: ")?;
                let subset = subset.as_slice();
                fmt_column_delimited(f, subset, "[", "]")
            }
            Rechunk => write!(f, "RECHUNK"),
            Unnest { columns } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_slice();
                fmt_column_delimited(f, columns, "[", "]")
            }
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => write!(f, "MERGE SORTED"),
            Pipeline { original, .. } => {
                if let Some(original) = original {
                    writeln!(f, "--- PIPELINE")?;
                    write!(f, "{:?}", original.as_ref())?;
                    let indent = 2;
                    writeln!(f, "{:indent$}--- END PIPELINE", "")
                } else {
                    writeln!(f, "PIPELINE")
                }
            }
            Rename { .. } => write!(f, "RENAME"),
            Drop { .. } => write!(f, "DROP"),
        }
    }
}
