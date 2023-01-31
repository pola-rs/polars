use super::*;
use crate::logical_plan::projection::is_regex_projection;

/// Specialized expressions for Categorical dtypes.
pub struct MetaNameSpace(pub(crate) Expr);

impl MetaNameSpace {
    /// Pop latest expression and return the input(s) of the popped expression.
    pub fn pop(self) -> Vec<Expr> {
        let mut arena = Arena::with_capacity(8);
        let node = to_aexpr(self.0, &mut arena);
        let ae = arena.get(node);
        let mut inputs = Vec::with_capacity(2);
        ae.nodes(&mut inputs);
        inputs
            .iter()
            .map(|node| node_to_expr(*node, &arena))
            .collect()
    }

    /// Get the root column names.
    pub fn root_names(&self) -> Vec<Arc<str>> {
        expr_to_leaf_column_names(&self.0)
    }

    /// Get the output name of this expression.
    pub fn output_name(&self) -> PolarsResult<Arc<str>> {
        expr_output_name(&self.0)
    }

    /// Undo any renaming operation like `alias`, `keep_name`.
    pub fn undo_aliases(mut self) -> Expr {
        self.0.mutate().apply(|e| match e {
            Expr::Alias(input, _)
            | Expr::KeepName(input)
            | Expr::RenameAlias { expr: input, .. } => {
                // remove this node
                *e = *input.clone();

                // continue iteration
                true
            }
            // continue iteration
            _ => true,
        });

        self.0
    }

    pub fn has_multiple_outputs(&self) -> bool {
        self.0.into_iter().any(|e| match e {
            Expr::Wildcard | Expr::Columns(_) | Expr::DtypeColumn(_) => true,
            Expr::Column(name) => is_regex_projection(name),
            _ => false,
        })
    }

    pub fn is_regex_projection(&self) -> bool {
        self.0.into_iter().any(|e| match e {
            Expr::Column(name) => is_regex_projection(name),
            _ => false,
        })
    }
}
