use std::fmt::Display;

use super::*;
use crate::plans::conversion::is_regex_projection;
use crate::plans::tree_format::TreeFmtVisitor;
use crate::plans::visitor::{AexprNode, TreeWalker};
use crate::prelude::tree_format::TreeFmtVisitorDisplay;

/// Specialized expressions for Categorical dtypes.
pub struct MetaNameSpace(pub(crate) Expr);

impl MetaNameSpace {
    /// Pop latest expression and return the input(s) of the popped expression.
    pub fn pop(self, _schema: Option<&Schema>) -> PolarsResult<Vec<Expr>> {
        let mut out = Vec::new();
        self.0.nodes_owned(&mut out);
        Ok(out)
    }

    /// Get the root column names.
    pub fn root_names(&self) -> Vec<PlSmallStr> {
        expr_to_leaf_column_names(&self.0)
    }

    /// A projection that only takes a column or a column + alias.
    pub fn is_simple_projection(&self, schema: Option<&Schema>) -> bool {
        let schema = match schema {
            None => &Default::default(),
            Some(s) => s,
        };
        let mut arena = Arena::with_capacity(8);
        let mut ctx = ExprToIRContext::new_no_verification(&mut arena, schema);
        to_expr_ir(self.0.clone(), &mut ctx)
            .map(|expr| aexpr_is_simple_projection(expr.node(), &arena))
            .unwrap_or(false)
    }

    /// Get the output name of this expression.
    pub fn output_name(&self) -> PolarsResult<PlSmallStr> {
        expr_output_name(&self.0)
    }

    /// Undo any renaming operation like `alias`, `keep_name`.
    pub fn undo_aliases(self) -> Expr {
        self.0.map_expr(|e| match e {
            Expr::Alias(input, _)
            | Expr::KeepName(input)
            | Expr::RenameAlias { expr: input, .. } => Arc::unwrap_or_clone(input),
            e => e,
        })
    }

    /// Indicate if this expression expands to multiple expressions.
    pub fn has_multiple_outputs(&self) -> bool {
        self.0.into_iter().any(|e| {
            matches!(
                e,
                Expr::Selector(_)
                    | Expr::Function {
                        function: FunctionExpr::StructExpr(StructFunction::SelectFields(_)),
                        ..
                    }
            )
        })
    }

    /// Indicate if this expression is a basic (non-regex) column.
    pub fn is_column(&self) -> bool {
        match &self.0 {
            Expr::Column(name) => !is_regex_projection(name),
            _ => false,
        }
    }

    /// Indicate if this expression only selects columns; the presence of any
    /// transform operations will cause the check to return `false`, though
    /// aliasing of the selected columns is optionally allowed.
    pub fn is_column_selection(&self, allow_aliasing: bool) -> bool {
        self.0.into_iter().all(|e| match e {
            Expr::Column(_) | Expr::Selector(_) => true,
            Expr::Alias(_, _) | Expr::KeepName(_) | Expr::RenameAlias { .. } => allow_aliasing,
            _ => false,
        })
    }

    /// Indicate if this expression represents a literal value (optionally aliased).
    pub fn is_literal(&self, allow_aliasing: bool) -> bool {
        self.0.into_iter().all(|e| match e {
            Expr::Literal(_) => true,
            Expr::Alias(_, _) => allow_aliasing,
            Expr::Cast {
                expr,
                dtype,
                options: CastOptions::Strict,
            } if matches!(dtype.as_literal(), Some(DataType::Datetime(_, _))) && matches!(&**expr, Expr::Literal(LiteralValue::Scalar(sc)) if matches!(sc.as_any_value(), AnyValue::Datetime(..))) => true,
            _ => false,
        })
    }

    /// Indicate if this expression expands to multiple expressions with regex expansion.
    pub fn is_regex_projection(&self) -> bool {
        self.0
            .into_iter()
            .any(|e| matches!(e, Expr::Selector(Selector::Matches(_))))
    }

    /// Get a hold to an implementor of the `Display` trait that will format as
    /// the expression as a tree
    pub fn into_tree_formatter(
        self,
        display_as_dot: bool,
        schema: Option<&Schema>,
    ) -> PolarsResult<impl Display> {
        let schema = match schema {
            None => &Default::default(),
            Some(s) => s,
        };
        let mut arena = Default::default();
        let mut ctx = ExprToIRContext::new_no_verification(&mut arena, schema);
        let node = to_expr_ir(self.0, &mut ctx)?.node();
        let mut visitor = TreeFmtVisitor::default();
        if display_as_dot {
            visitor.display = TreeFmtVisitorDisplay::DisplayDot;
        }
        AexprNode::new(node).visit(&mut visitor, &arena)?;
        Ok(visitor)
    }
}
