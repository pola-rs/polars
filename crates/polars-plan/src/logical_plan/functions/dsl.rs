use crate::logical_plan::expr_expansion::rewrite_projections;
use super::*;

// Except for Opaque functions, this only has the DSL name of the function.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DslFunction {
    Opaque(FunctionNode),
    Explode{
        columns: Vec<Expr>
    },
    Melt{
        args: MeltArgs
    },
    RowIndex {
        name: Arc<str>,
        offset: Option<IdxSize>,
    },
}


impl DslFunction {
    pub(crate) fn into_function_node(self, schema: &Schema) -> PolarsResult<FunctionNode> {
        let function = match self {
            DslFunction::Explode{columns} => {
                let columns = rewrite_projections(columns, schema, &[])?;
                // columns to string
                let columns = columns
                    .iter()
                    .map(|e| {
                        if let Expr::Column(name) = e {
                            Ok(name.clone())
                        } else {
                            polars_bail!(InvalidOperation: "expected column expression")
                        }
                    })
                    .collect::<PolarsResult<Arc<[Arc<str>]>>>()?;
                FunctionNode::Explode {
                    columns,
                    schema: Default::default()
                }
            },
            DslFunction::Melt {args} => FunctionNode::Melt {args: Arc::new(args), schema: Default::default()},
            DslFunction::Opaque(func) => func,
            DslFunction::RowIndex {name, offset} => FunctionNode::RowIndex {name, offset, schema: Default::default()}
        };
        Ok(function)
    }

}

impl Debug for DslFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for DslFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use DslFunction::*;
        match self {
            Opaque(inner) => write!(f, "{inner}"),
            Explode { .. } => write!(f, "EXPLODE"),
            Melt { .. } => write!(f, "MELT"),
            RowIndex { .. } => write!(f, "WITH ROW INDEX"),
            // DropNulls { subset } => {
            //     write!(f, "DROP_NULLS by: ")?;
            //     let subset = subset.as_ref();
            //     fmt_column_delimited(f, subset, "[", "]")
            // },
            // Rechunk => write!(f, "RECHUNK"),
            // Count { .. } => write!(f, "FAST COUNT(*)"),
            // Unnest { columns } => {
            //     write!(f, "UNNEST by:")?;
            //     let columns = columns.as_ref();
            //     fmt_column_delimited(f, columns, "[", "]")
            // },
            // #[cfg(feature = "merge_sorted")]
            // MergeSorted { .. } => write!(f, "MERGE SORTED"),
            // Pipeline { original, .. } => {
            //     if let Some(original) = original {
            //         writeln!(f, "--- STREAMING")?;
            //         write!(f, "{:?}", original.as_ref())?;
            //         let indent = 2;
            //         writeln!(f, "{:indent$}--- END STREAMING", "")
            //     } else {
            //         writeln!(f, "STREAMING")
            //     }
            // },
            // Rename { .. } => write!(f, "RENAME"),
        }
    }
}


impl From<FunctionNode> for DslFunction {
    fn from(value: FunctionNode) -> Self {
        DslFunction::Opaque(value)
    }
}