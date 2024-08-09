use std::fmt::Display;
use std::ops::BitAnd;

use super::*;
use crate::plans::conversion::is_regex_projection;
use crate::plans::ir::tree_format::TreeFmtVisitor;
use crate::plans::visitor::{AexprNode, TreeWalker};

/// Specialized expressions for Categorical dtypes.
pub struct MetaNameSpace(pub(crate) Expr);

impl MetaNameSpace {
    /// Pop latest expression and return the input(s) of the popped expression.
    pub fn pop(self) -> PolarsResult<Vec<Expr>> {
        let mut arena = Arena::with_capacity(8);
        let node = to_aexpr(self.0, &mut arena)?;
        let ae = arena.get(node);
        let mut inputs = Vec::with_capacity(2);
        ae.nodes(&mut inputs);
        Ok(inputs
            .iter()
            .map(|node| node_to_expr(*node, &arena))
            .collect())
    }

    /// Get the root column names.
    pub fn root_names(&self) -> Vec<Arc<str>> {
        expr_to_leaf_column_names(&self.0)
    }

    /// A projection that only takes a column or a column + alias.
    pub fn is_simple_projection(&self) -> bool {
        let mut arena = Arena::with_capacity(8);
        to_aexpr(self.0.clone(), &mut arena)
            .map(|node| aexpr_is_simple_projection(node, &arena))
            .unwrap_or(false)
    }

    /// Get the output name of this expression.
    pub fn output_name(&self) -> PolarsResult<Arc<str>> {
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
        self.0.into_iter().any(|e| match e {
            Expr::Selector(_) | Expr::Wildcard | Expr::Columns(_) | Expr::DtypeColumn(_) => true,
            Expr::IndexColumn(idxs) => idxs.len() > 1,
            Expr::Column(name) => is_regex_projection(name),
            _ => false,
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
            Expr::Column(_)
            | Expr::Columns(_)
            | Expr::DtypeColumn(_)
            | Expr::Exclude(_, _)
            | Expr::Nth(_)
            | Expr::IndexColumn(_)
            | Expr::Selector(_)
            | Expr::Wildcard => true,
            Expr::Alias(_, _) | Expr::KeepName(_) | Expr::RenameAlias { .. } if allow_aliasing => {
                true
            },
            _ => false,
        })
    }

    /// Indicate if this expression expands to multiple expressions with regex expansion.
    pub fn is_regex_projection(&self) -> bool {
        self.0.into_iter().any(|e| match e {
            Expr::Column(name) => is_regex_projection(name),
            _ => false,
        })
    }

    pub fn _selector_add(self, other: Expr) -> PolarsResult<Expr> {
        if let Expr::Selector(mut s) = self.0 {
            if let Expr::Selector(s_other) = other {
                s = s + s_other;
            } else {
                s = s + Selector::Root(Box::new(other))
            }
            Ok(Expr::Selector(s))
        } else {
            polars_bail!(ComputeError: "expected selector, got {:?}", self.0)
        }
    }

    pub fn _selector_and(self, other: Expr) -> PolarsResult<Expr> {
        if let Expr::Selector(mut s) = self.0 {
            if let Expr::Selector(s_other) = other {
                s = s.bitand(s_other);
            } else {
                s = s.bitand(Selector::Root(Box::new(other)))
            }
            Ok(Expr::Selector(s))
        } else {
            polars_bail!(ComputeError: "expected selector, got {:?}", self.0)
        }
    }

    pub fn _selector_sub(self, other: Expr) -> PolarsResult<Expr> {
        if let Expr::Selector(mut s) = self.0 {
            if let Expr::Selector(s_other) = other {
                s = s - s_other;
            } else {
                s = s - Selector::Root(Box::new(other))
            }
            Ok(Expr::Selector(s))
        } else {
            polars_bail!(ComputeError: "expected selector, got {:?}", self.0)
        }
    }

    pub fn _selector_xor(self, other: Expr) -> PolarsResult<Expr> {
        if let Expr::Selector(mut s) = self.0 {
            if let Expr::Selector(s_other) = other {
                s = s ^ s_other;
            } else {
                s = s ^ Selector::Root(Box::new(other))
            }
            Ok(Expr::Selector(s))
        } else {
            polars_bail!(ComputeError: "expected selector, got {:?}", self.0)
        }
    }

    pub fn _into_selector(self) -> Expr {
        if let Expr::Selector(_) = self.0 {
            self.0
        } else {
            Expr::Selector(Selector::new(self.0))
        }
    }

    /// Get a hold to an implementor of the `Display` trait that will format as
    /// the expression as a tree
    pub fn into_tree_formatter(self) -> PolarsResult<impl Display> {
        let mut arena = Default::default();
        let node = to_aexpr(self.0, &mut arena)?;
        let mut visitor = TreeFmtVisitor::default();

        AexprNode::new(node).visit(&mut visitor, &arena)?;

        Ok(visitor)
    }
}
