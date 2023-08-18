use std::fmt::{Display, Formatter, UpperExp};

use polars_core::error::*;

use crate::logical_plan::visitor::{VisitRecursion, Visitor};
use crate::prelude::visitor::AexprNode;
use crate::prelude::*;

/// Hack UpperExpr trait to get a kind of formatting that doesn't traverse the nodes.
/// So we can format with {foo:E}
impl UpperExp for AExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            AExpr::Explode(_) => "explode",
            AExpr::Alias(_, name) => return write!(f, "alias({})", name.as_ref()),
            AExpr::Column(name) => return write!(f, "col({})", name.as_ref()),
            AExpr::Literal(lv) => return write!(f, "lit({lv:?})"),
            AExpr::BinaryExpr { op, .. } => return write!(f, "binary: {}", op),
            AExpr::Cast {
                data_type, strict, ..
            } => {
                return if *strict {
                    write!(f, "strict cast({})", data_type)
                } else {
                    write!(f, "cast({})", data_type)
                }
            },
            AExpr::Sort { options, .. } => {
                return write!(
                    f,
                    "sort: {}{}{}",
                    options.descending as u8, options.nulls_last as u8, options.multithreaded as u8
                )
            },
            AExpr::Take { .. } => "take",
            AExpr::SortBy { descending, .. } => {
                write!(f, "sort_by:")?;
                for i in descending {
                    write!(f, "{}", *i as u8)?;
                }
                return Ok(());
            },
            AExpr::Filter { .. } => "filter",
            AExpr::Agg(a) => {
                let s: &str = a.into();
                return write!(f, "{}", s.to_lowercase());
            },
            AExpr::Ternary { .. } => "ternary",
            AExpr::AnonymousFunction { options, .. } => {
                return write!(f, "anonymous_function: {}", options.fmt_str)
            },
            AExpr::Function { function, .. } => return write!(f, "function: {function}"),
            AExpr::Window { .. } => "window",
            AExpr::Wildcard => "*",
            AExpr::Slice { .. } => "slice",
            AExpr::Count => "count",
            AExpr::Nth(v) => return write!(f, "nth({})", v),
        };

        write!(f, "{s}")
    }
}

pub(crate) struct TreeFmtVisitor {
    levels: Vec<Vec<String>>,
    depth: u32,
    width: u32,
}

impl TreeFmtVisitor {
    pub(crate) fn new() -> Self {
        Self {
            levels: vec![],
            depth: 0,
            width: 0,
        }
    }
}

impl Visitor for TreeFmtVisitor {
    type Node = AexprNode;

    /// Invoked before any children of `node` are visited.
    fn pre_visit(&mut self, node: &Self::Node) -> PolarsResult<VisitRecursion> {
        let ae = node.to_aexpr();
        let repr = format!("{:E}", ae);

        if self.levels.len() <= self.depth as usize {
            self.levels.push(vec![])
        }

        // the post-visit ensures the width of this node is known
        let row = self.levels.get_mut(self.depth as usize).unwrap();

        // set default values to ensure we format at the right width
        row.resize(self.width as usize + 1, "".to_string());
        row[self.width as usize] = repr;

        // we will enter depth first, we enter child so depth increases
        self.depth += 1;

        Ok(VisitRecursion::Continue)
    }

    fn post_visit(&mut self, _node: &Self::Node) -> PolarsResult<VisitRecursion> {
        // because we traverse depth first
        // every post-visit increases the width as we finished a depth-first branch
        self.width += 1;

        // we finished this branch so we decrease in depth, back the caller node
        self.depth -= 1;
        Ok(VisitRecursion::Continue)
    }
}

fn format_levels(f: &mut Formatter<'_>, levels: &[Vec<String>]) -> std::fmt::Result {
    let n_cols = levels.iter().map(|v| v.len()).max().unwrap();

    let mut col_widths = vec![0usize; n_cols];

    for (i, col_width) in col_widths.iter_mut().enumerate() {
        *col_width = levels
            .iter()
            .map(|row| row.get(i).map(|s| s.as_str()).unwrap_or("").chars().count())
            .max()
            .unwrap();
    }

    const COL_SPACING: usize = 4;

    for (row_count, row) in levels.iter().enumerate() {
        // write vertical bars
        if row_count != 0 {
            writeln!(f)?;
            for ((col_i, col_name), col_width) in row.iter().enumerate().zip(&col_widths) {
                let mut col_spacing = COL_SPACING;
                if col_i > 0 {
                    col_spacing *= 2;
                }

                let mut remaining = col_width + col_spacing;
                let half = (*col_width + col_spacing) / 2;

                // left_half
                for _ in 0..half {
                    remaining -= 1;
                    write!(f, " ")?;
                }
                // bar
                remaining -= 1;
                let val = if col_name.is_empty() { ' ' } else { '|' };
                write!(f, "{}", val)?;

                for _ in 0..remaining {
                    write!(f, " ")?
                }
            }
            write!(f, "\n\n")?;
        }

        // write column names and spacing
        for (col_repr, col_width) in row.iter().zip(&col_widths) {
            for _ in 0..COL_SPACING {
                write!(f, " ")?
            }
            write!(f, "{}", col_repr)?;
            let remaining = *col_width - col_repr.chars().count();
            for _ in 0..remaining + COL_SPACING {
                write!(f, " ")?
            }
        }
        writeln!(f)?;
    }

    Ok(())
}

impl Display for TreeFmtVisitor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        format_levels(f, &self.levels)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::logical_plan::visitor::TreeWalker;

    #[test]
    fn test_tree_fmt_visit() {
        let e = (col("foo") * lit(2) + lit(3) + lit(43)).sum();
        let mut arena = Default::default();
        let node = to_aexpr(e, &mut arena);

        let mut visitor = TreeFmtVisitor::new();

        AexprNode::with_context(node, &mut arena, |ae_node| ae_node.visit(&mut visitor)).unwrap();
        let expected: &[&[&str]] = &[
            &["sum"],
            &["binary: +"],
            &["lit(43)", "binary: +"],
            &["", "lit(3)", "binary: *"],
            &["", "", "lit(2)", "col(foo)"],
        ];

        assert_eq!(visitor.levels, expected);
    }
}
