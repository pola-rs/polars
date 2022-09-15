use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;

#[derive(Clone, Debug)]
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
        // used for formatting
        #[cfg_attr(feature = "serde", serde(skip))]
        fmt_str: &'static str,
    },
    FastProjection {
        columns: Arc<Vec<Arc<str>>>,
    },
    DropNulls {
        subset: Arc<Vec<String>>,
    },
    Rechunk,
}

impl PartialEq for FunctionNode {
    fn eq(&self, other: &Self) -> bool {
        use FunctionNode::*;
        match (self, other) {
            (FastProjection { columns: l }, FastProjection { columns: r }) => l == r,
            (DropNulls { subset: l }, DropNulls { subset: r }) => l == r,
            (Rechunk, Rechunk) => true,
            _ => false,
        }
    }
}

impl FunctionNode {
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
            FastProjection { columns } => {
                let schema = columns
                    .iter()
                    .map(|name| {
                        let name = name.as_ref();
                        input_schema
                            .get_field(name)
                            .ok_or_else(|| PolarsError::NotFound(name.to_string().into()))
                    })
                    .collect::<PolarsResult<Schema>>()?;
                Ok(Cow::Owned(Arc::new(schema)))
            }
            DropNulls { .. } => Ok(Cow::Borrowed(input_schema)),
            Rechunk => Ok(Cow::Borrowed(input_schema)),
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            FastProjection { .. } | DropNulls { .. } | Rechunk => true,
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionNode::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            FastProjection { .. } | DropNulls { .. } | Rechunk => true,
        }
    }

    pub(crate) fn evaluate(&self, mut df: DataFrame) -> PolarsResult<DataFrame> {
        use FunctionNode::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            FastProjection { columns } => df.select(columns.as_slice()),
            DropNulls { subset } => df.drop_nulls(Some(subset.as_slice())),
            Rechunk => {
                df.as_single_chunk_par();
                Ok(df)
            }
        }
    }
}

impl Display for FunctionNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionNode::*;
        match self {
            Opaque { fmt_str, .. } => write!(f, "{}", fmt_str),
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
        }
    }
}
