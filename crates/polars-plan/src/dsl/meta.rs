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
        to_expr_ir(&self.0, &mut ctx)
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

    /// Indicate if this expression expands to multiple expressions with regex expansion.
    pub fn is_regex_projection(&self) -> bool {
        self.0
            .into_iter()
            .any(|e| matches!(e, Expr::Selector(Selector::Matches(_))))
    }

    /// Get a hold to an implementor of the `Display` trait that will format
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
        let node = to_expr_ir(&self.0, &mut ctx)?.node();
        let mut visitor = TreeFmtVisitor::default();
        if display_as_dot {
            visitor.display = TreeFmtVisitorDisplay::DisplayDot;
        }
        AexprNode::new(node).visit(&mut visitor, &arena)?;
        Ok(visitor)
    }
}

// Properties
impl MetaNameSpace {
    fn into_expr_ir(self) -> PolarsResult<(ExprIR, Arena<AExpr>)> {
        let mut expr_arena = Arena::new();
        let schema = Default::default();
        let mut ctx = ExprToIRContext::new_no_verification(&mut expr_arena, &schema);
        let e_ir = to_expr_ir(&self.0, &mut ctx)?;

        Ok((e_ir, expr_arena))
    }

    /// Resolves the expression as `select` does on a frame with `schema`, which must contain every
    /// column it reads, or on its own without a schema.
    fn into_expr_irs(self, schema: Option<&Schema>) -> PolarsResult<(Vec<ExprIR>, Arena<AExpr>)> {
        let Some(schema) = schema else {
            let (e_ir, expr_arena) = self.into_expr_ir()?;
            return Ok((vec![e_ir], expr_arena));
        };
        let plan = DslBuilder::from_existing_df(DataFrame::empty_with_schema(schema))
            .project(vec![self.0], Default::default())
            .build()
            .to_alp()?;
        let exprs = match plan.lp_arena.get(plan.lp_top) {
            IR::Select { expr, .. } => expr.clone(),
            // A selector that matches no column.
            _ => Vec::new(),
        };
        Ok((exprs, plan.expr_arena))
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

    pub fn is_length_preserving(self, schema: Option<&Schema>) -> PolarsResult<bool> {
        let (exprs, expr_arena) = self.into_expr_irs(schema)?;
        Ok(exprs.iter().all(|e| e.is_length_preserving(&expr_arena)))
    }

    pub fn is_scalar(self, schema: Option<&Schema>) -> PolarsResult<bool> {
        let (exprs, expr_arena) = self.into_expr_irs(schema)?;
        Ok(exprs.iter().all(|e| e.is_scalar(&expr_arena)))
    }

    pub fn is_known_length(self, schema: Option<&Schema>) -> PolarsResult<bool> {
        let (exprs, expr_arena) = self.into_expr_irs(schema)?;
        Ok(exprs.iter().all(|e| e.is_known_length(&expr_arena)))
    }

    /// Indicate if this expression as a whole is row-separable (see [`is_row_separable_rec`]).
    pub fn is_row_separable(self, schema: Option<&Schema>) -> PolarsResult<bool> {
        let has_schema = schema.is_some();
        let (exprs, expr_arena) = self.into_expr_irs(schema)?;
        Ok(exprs.iter().all(|e| {
            is_row_separable_rec(e.node(), &expr_arena)
                && (has_schema || !has_unresolved_set_argument(e.node(), &expr_arena))
        }))
    }

    /// Indicate if this expression yields one scalar value that does not depend on any
    /// input frame (see [`is_input_independent_scalar_rec`]).
    pub fn is_input_independent_scalar(self) -> PolarsResult<bool> {
        let (e_ir, expr_arena) = self.into_expr_ir()?;
        Ok(is_input_independent_scalar_rec(e_ir.node(), &expr_arena))
    }
}

/// Whether an `is_in` or `replace` has a non-literal argument, which only the schema can resolve
/// into a set (`implode()`) or a value per row.
fn has_unresolved_set_argument(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    expr_arena.iter(node).any(|(_, ae)| match ae {
        #[cfg(feature = "is_in")]
        AExpr::Function {
            function: IRFunctionExpr::Boolean(IRBooleanFunction::IsIn { .. }),
            input,
            ..
        } => !matches!(expr_arena.get(input[1].node()), AExpr::Literal(_)),
        #[cfg(feature = "replace")]
        AExpr::Function {
            function: IRFunctionExpr::Replace | IRFunctionExpr::ReplaceStrict { .. },
            input,
            ..
        } => input[1..3]
            .iter()
            .any(|e| !matches!(expr_arena.get(e.node()), AExpr::Literal(_))),
        _ => false,
    })
}
