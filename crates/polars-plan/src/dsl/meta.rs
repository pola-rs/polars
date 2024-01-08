use std::fmt::Display;
use std::ops::BitAnd;

use super::*;
use crate::dsl::selector::Selector;
use crate::logical_plan::projection::is_regex_projection;
use crate::logical_plan::tree_format::TreeFmtVisitor;
use crate::logical_plan::visitor::{AexprNode, TreeWalker};

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

    /// A projection that only takes a column or a column + alias.
    pub fn is_simple_projection(&self) -> bool {
        let mut arena = Arena::with_capacity(8);
        let node = to_aexpr(self.0.clone(), &mut arena);
        aexpr_is_simple_projection(node, &arena)
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
            },
            // continue iteration
            _ => true,
        });

        self.0
    }

    /// Indicate if this expression expands to multiple expressions.
    pub fn has_multiple_outputs(&self) -> bool {
        self.0.into_iter().any(|e| match e {
            Expr::Selector(_) | Expr::Wildcard | Expr::Columns(_) | Expr::DtypeColumn(_) => true,
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
            polars_bail!(ComputeError: "expected selector, got {}", self.0)
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
            polars_bail!(ComputeError: "expected selector, got {}", self.0)
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
            polars_bail!(ComputeError: "expected selector, got {}", self.0)
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
        let node = to_aexpr(self.0, &mut arena);
        let mut visitor = TreeFmtVisitor::new();
        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor))?;
        Ok(visitor)
    }
}
