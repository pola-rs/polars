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

    let row_idx_width = levels.len().to_string().len() + 1;
    let col_idx_width = n_cols.to_string().len();
    let space = " ";
    let dash = "─";

    for (i, col_width) in col_widths.iter_mut().enumerate() {
        *col_width = levels
            .iter()
            .map(|row| row.get(i).map(|s| s.as_str()).unwrap_or("").chars().count())
            .max()
            .map(|n| if n < col_idx_width { col_idx_width } else { n })
            .unwrap();
    }

    const COL_SPACING: usize = 2;

    for (row_count, row) in levels.iter().enumerate() {
        if row_count == 0 {
            // write the col numbers
            writeln!(f)?;
            write!(f, "{space:>row_idx_width$}  ")?;
            for (col_i, (_, col_width)) in
                levels.last().unwrap().iter().zip(&col_widths).enumerate()
            {
                let mut col_spacing = COL_SPACING;
                if col_i > 0 {
                    col_spacing *= 2;
                }
                let half = (col_spacing + 4) / 2;
                let remaining = col_spacing + 4 - half;

                // left_half
                write!(f, "{space:^half$}")?;
                // col num
                write!(f, "{col_i:^col_width$}")?;

                write!(f, "{space:^remaining$}")?;
            }
            writeln!(f)?;

            // write the horizontal line
            write!(f, "{space:>row_idx_width$} ┌")?;
            for (col_i, (_, col_width)) in
                levels.last().unwrap().iter().zip(&col_widths).enumerate()
            {
                let mut col_spacing = COL_SPACING;
                if col_i > 0 {
                    col_spacing *= 2;
                }
                write!(f, "{dash:─^width$}", width = col_width + col_spacing + 4)?;
            }
            write!(f, "\n{space:>row_idx_width$} │\n")?;
        } else {
            // write connecting lines
            write!(f, "{space:>row_idx_width$} │")?;
            let mut last_empty = true;
            let mut before = "";
            for ((col_i, col_name), col_width) in row.iter().enumerate().zip(&col_widths) {
                let mut col_spacing = COL_SPACING;
                if col_i > 0 {
                    col_spacing *= 2;
                }

                let half = (*col_width + col_spacing + 4) / 2;
                let remaining = col_width + col_spacing + 4 - half - 1;
                if last_empty {
                    // left_half
                    write!(f, "{space:^half$}")?;
                    // bar
                    if col_name.is_empty() {
                        write!(f, " ")?;
                    } else {
                        write!(f, "│")?;
                        last_empty = false;
                        before = "│";
                    }
                } else {
                    // left_half
                    write!(f, "{dash:─^half$}")?;
                    // bar
                    write!(f, "╮")?;
                    before = "╮"
                }
                if (col_i == row.len() - 1) | col_name.is_empty() {
                    write!(f, "{space:^remaining$}")?;
                } else {
                    if before == "│" {
                        write!(f, " ╰")?;
                    } else {
                        write!(f, "──")?;
                    }
                    write!(f, "{dash:─^width$}", width = remaining - 2)?;
                }
            }
            writeln!(f)?;
            // write vertical bars x 2
            for _ in 0..2 {
                write!(f, "{space:>row_idx_width$} │")?;
                for ((col_i, col_name), col_width) in row.iter().enumerate().zip(&col_widths) {
                    let mut col_spacing = COL_SPACING;
                    if col_i > 0 {
                        col_spacing *= 2;
                    }

                    let half = (*col_width + col_spacing + 4) / 2;
                    let remaining = col_width + col_spacing + 4 - half - 1;

                    // left_half
                    write!(f, "{space:^half$}")?;
                    // bar
                    let val = if col_name.is_empty() { ' ' } else { '│' };
                    write!(f, "{}", val)?;

                    write!(f, "{space:^remaining$}")?;
                }
                writeln!(f)?;
            }
        }

        // write the top of the boxes
        write!(f, "{space:>row_idx_width$} │")?;
        for (col_i, (col_repr, col_width)) in row.iter().zip(&col_widths).enumerate() {
            let mut col_spacing = COL_SPACING;
            if col_i > 0 {
                col_spacing *= 2;
            }
            let char_count = col_repr.chars().count() + 4;
            let half = (*col_width + col_spacing + 4 - char_count) / 2;
            let remaining = col_width + col_spacing + 4 - half - char_count;

            write!(f, "{space:^half$}")?;

            if !col_repr.is_empty() {
                write!(f, "╭")?;
                write!(f, "{dash:─^width$}", width = char_count - 2)?;
                write!(f, "╮")?;
            } else {
                write!(f, "    ")?;
            }
            write!(f, "{space:^remaining$}")?;
        }
        writeln!(f)?;

        // write column names and spacing
        write!(f, "{row_count:>row_idx_width$} │")?;
        for (col_i, (col_repr, col_width)) in row.iter().zip(&col_widths).enumerate() {
            let mut col_spacing = COL_SPACING;
            if col_i > 0 {
                col_spacing *= 2;
            }
            let char_count = col_repr.chars().count() + 4;
            let half = (*col_width + col_spacing + 4 - char_count) / 2;
            let remaining = col_width + col_spacing + 4 - half - char_count;

            write!(f, "{space:^half$}")?;

            if !col_repr.is_empty() {
                write!(f, "│ {} │", col_repr)?;
            } else {
                write!(f, "    ")?;
            }
            write!(f, "{space:^remaining$}")?;
        }
        writeln!(f)?;

        // write the bottom of the boxes
        write!(f, "{space:>row_idx_width$} │")?;
        for (col_i, (col_repr, col_width)) in row.iter().zip(&col_widths).enumerate() {
            let mut col_spacing = COL_SPACING;
            if col_i > 0 {
                col_spacing *= 2;
            }
            let char_count = col_repr.chars().count() + 4;
            let half = (*col_width + col_spacing + 4 - char_count) / 2;
            let remaining = col_width + col_spacing + 4 - half - char_count;

            write!(f, "{space:^half$}")?;

            if !col_repr.is_empty() {
                write!(f, "╰")?;
                write!(f, "{dash:─^width$}", width = char_count - 2)?;
                write!(f, "╯")?;
            } else {
                write!(f, "    ")?;
            }
            write!(f, "{space:^remaining$}")?;
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
