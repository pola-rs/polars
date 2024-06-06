use std::fmt;
use std::path::PathBuf;

use super::format::ExprIRSliceDisplay;
use crate::constants::UNLIMITED_CACHE;
use crate::prelude::alp::format::ColumnsDisplay;
use crate::prelude::*;

pub struct IRDotDisplay<'a> {
    is_streaming: bool,
    lp: IRPlanRef<'a>,
}

const INDENT: &str = "  ";

#[derive(Clone, Copy)]
enum DotNode {
    Plain(usize),
    Cache(usize),
}

impl fmt::Display for DotNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DotNode::Plain(n) => write!(f, "p{n}"),
            DotNode::Cache(n) => write!(f, "c{n}"),
        }
    }
}

#[inline(always)]
fn write_label<'a, 'b>(
    f: &'b mut fmt::Formatter<'a>,
    id: DotNode,
    mut w: impl FnMut(&mut EscapeLabel<'a, 'b>) -> fmt::Result,
) -> fmt::Result {
    write!(f, "{INDENT}{id}[label=\"")?;

    let mut escaped = EscapeLabel(f);
    w(&mut escaped)?;
    let EscapeLabel(f) = escaped;

    writeln!(f, "\"]")?;

    Ok(())
}

impl<'a> IRDotDisplay<'a> {
    pub fn new(lp: IRPlanRef<'a>) -> Self {
        if let Some(streaming_lp) = lp.extract_streaming_plan() {
            return Self::new_streaming(streaming_lp);
        }

        Self {
            is_streaming: false,
            lp,
        }
    }

    fn new_streaming(lp: IRPlanRef<'a>) -> Self {
        Self {
            is_streaming: true,
            lp,
        }
    }

    fn with_root(&self, root: Node) -> Self {
        Self {
            is_streaming: false,
            lp: self.lp.with_root(root),
        }
    }

    fn display_expr(&self, expr: &'a ExprIR) -> ExprIRDisplay<'a> {
        expr.display(self.lp.expr_arena)
    }

    fn display_exprs(&self, exprs: &'a [ExprIR]) -> ExprIRSliceDisplay<'a, ExprIR> {
        ExprIRSliceDisplay {
            exprs,
            expr_arena: self.lp.expr_arena,
        }
    }

    fn _format(
        &self,
        f: &mut fmt::Formatter<'_>,
        parent: Option<DotNode>,
        last: &mut usize,
    ) -> std::fmt::Result {
        use fmt::Write;

        let root = self.lp.root();

        let mut parent = parent;
        if self.is_streaming {
            *last += 1;
            let streaming_node = DotNode::Plain(*last);

            if let Some(parent) = parent {
                writeln!(f, "{INDENT}{parent} -- {streaming_node}")?;
                write_label(f, streaming_node, |f| f.write_str("STREAMING"))?;
            }

            parent = Some(streaming_node);
        }
        let parent = parent;

        let id = if let IR::Cache { id, .. } = root {
            DotNode::Cache(*id)
        } else {
            *last += 1;
            DotNode::Plain(*last)
        };

        if let Some(parent) = parent {
            writeln!(f, "{INDENT}{parent} -- {id}")?;
        }

        use IR::*;
        match root {
            Union { inputs, .. } => {
                for input in inputs {
                    self.with_root(*input)._format(f, Some(id), last)?;
                }

                write_label(f, id, |f| f.write_str("UNION"))?;
            },
            HConcat { inputs, .. } => {
                for input in inputs {
                    self.with_root(*input)._format(f, Some(id), last)?;
                }

                write_label(f, id, |f| f.write_str("HCONCAT"))?;
            },
            Cache {
                input, cache_hits, ..
            } => {
                self.with_root(*input)._format(f, Some(id), last)?;

                if *cache_hits == UNLIMITED_CACHE {
                    write_label(f, id, |f| f.write_str("CACHE"))?;
                } else {
                    write_label(f, id, |f| write!(f, "CACHE: {cache_hits} times"))?;
                };
            },
            Filter { predicate, input } => {
                self.with_root(*input)._format(f, Some(id), last)?;

                let pred = self.display_expr(predicate);
                write_label(f, id, |f| write!(f, "FILTER BY {pred}"))?;
            },
            #[cfg(feature = "python")]
            PythonScan { predicate, options } => {
                let predicate = predicate.as_ref().map(|e| self.display_expr(e));
                let with_columns = NumColumns(options.with_columns.as_ref().map(|s| s.as_slice()));
                let total_columns = options.schema.len();
                let predicate = OptionExprIRDisplay(predicate);

                write_label(f, id, |f| {
                    write!(
                        f,
                        "PYTHON SCAN\nπ {with_columns}/{total_columns};\nσ {predicate}"
                    )
                })?
            },
            Select {
                expr,
                input,
                schema,
                ..
            } => {
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "π {}/{}", expr.len(), schema.len()))?;
            },
            Sort {
                input, by_column, ..
            } => {
                let by_column = self.display_exprs(by_column);
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "SORT BY {by_column}"))?;
            },
            GroupBy {
                input, keys, aggs, ..
            } => {
                let keys = self.display_exprs(keys);
                let aggs = self.display_exprs(aggs);
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "AGG {aggs}\nBY\n{keys}"))?;
            },
            HStack { input, exprs, .. } => {
                let exprs = self.display_exprs(exprs);
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "WITH COLUMNS {exprs}"))?;
            },
            Reduce { input, exprs, .. } => {
                let exprs = self.display_exprs(exprs);
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "REDUCE {exprs}"))?;
            },
            Slice { input, offset, len } => {
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| write!(f, "SLICE offset: {offset}; len: {len}"))?;
            },
            Distinct { input, options, .. } => {
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| {
                    f.write_str("DISTINCT")?;

                    if let Some(subset) = &options.subset {
                        f.write_str(" BY ")?;

                        let mut subset = subset.iter();

                        if let Some(fst) = subset.next() {
                            f.write_str(fst)?;
                            for name in subset {
                                write!(f, ", \"{name}\"")?;
                            }
                        } else {
                            f.write_str("None")?;
                        }
                    }

                    Ok(())
                })?;
            },
            DataFrameScan {
                schema,
                projection,
                selection,
                ..
            } => {
                let num_columns = NumColumns(projection.as_ref().map(|p| p.as_ref().as_ref()));
                let selection = selection.as_ref().map(|e| self.display_expr(e));
                let selection = OptionExprIRDisplay(selection);
                let total_columns = schema.len();

                write_label(f, id, |f| {
                    write!(f, "TABLE\nπ {num_columns}/{total_columns};\nσ {selection}")
                })?;
            },
            Scan {
                paths,
                file_info,
                predicate,
                scan_type,
                file_options: options,
                output_schema: _,
            } => {
                let name: &str = scan_type.into();
                let path = PathsDisplay(paths.as_ref());
                let with_columns = options.with_columns.as_ref().map(|cols| cols.as_slice());
                let with_columns = NumColumns(with_columns);
                let total_columns = file_info.schema.len();
                let predicate = predicate.as_ref().map(|e| self.display_expr(e));
                let predicate = OptionExprIRDisplay(predicate);

                write_label(f, id, |f| {
                    write!(
                        f,
                        "{name} SCAN {path}\nπ {with_columns}/{total_columns};\nσ {predicate}",
                    )
                })?;
            },
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                self.with_root(*input_left)._format(f, Some(id), last)?;
                self.with_root(*input_right)._format(f, Some(id), last)?;

                let left_on = self.display_exprs(left_on);
                let right_on = self.display_exprs(right_on);

                write_label(f, id, |f| {
                    write!(
                        f,
                        "JOIN {}\nleft: {left_on};\nright: {right_on}",
                        options.args.how
                    )
                })?;
            },
            MapFunction {
                input, function, ..
            } => {
                if let Some(streaming_lp) = function.to_streaming_lp() {
                    Self::new_streaming(streaming_lp)._format(f, Some(id), last)?;
                } else {
                    self.with_root(*input)._format(f, Some(id), last)?;
                    write_label(f, id, |f| write!(f, "{function}"))?;
                }
            },
            ExtContext { input, .. } => {
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| f.write_str("EXTERNAL_CONTEXT"))?;
            },
            Sink { input, payload, .. } => {
                self.with_root(*input)._format(f, Some(id), last)?;

                write_label(f, id, |f| {
                    f.write_str(match payload {
                        SinkType::Memory => "SINK (MEMORY)",
                        SinkType::File { .. } => "SINK (FILE)",
                        #[cfg(feature = "cloud")]
                        SinkType::Cloud { .. } => "SINK (CLOUD)",
                    })
                })?;
            },
            SimpleProjection { input, columns } => {
                let num_columns = columns.as_ref().len();
                let total_columns = self.lp.lp_arena.get(*input).schema(self.lp.lp_arena).len();

                let columns = ColumnsDisplay(columns.as_ref());
                self.with_root(*input)._format(f, Some(id), last)?;
                write_label(f, id, |f| {
                    write!(f, "simple π {num_columns}/{total_columns}\n[{columns}]")
                })?;
            },
            Invalid => write_label(f, id, |f| f.write_str("INVALID"))?,
        }

        Ok(())
    }
}

// A few utility structures for formatting
struct PathsDisplay<'a>(&'a [PathBuf]);
struct NumColumns<'a>(Option<&'a [String]>);
struct OptionExprIRDisplay<'a>(Option<ExprIRDisplay<'a>>);

impl fmt::Display for PathsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.len() {
            0 => Ok(()),
            1 => self.0[0].display().fmt(f),
            _ => write!(
                f,
                "{} files: first file: {}",
                self.0.len(),
                self.0[0].display()
            ),
        }
    }
}

impl fmt::Display for NumColumns<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("*"),
            Some(columns) => columns.len().fmt(f),
        }
    }
}

impl fmt::Display for OptionExprIRDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("None"),
            Some(expr) => expr.fmt(f),
        }
    }
}

/// Utility structure to write to a [`fmt::Formatter`] whilst escaping the output as a label name
struct EscapeLabel<'a, 'b>(&'b mut fmt::Formatter<'a>);

impl<'a, 'b> fmt::Write for EscapeLabel<'a, 'b> {
    fn write_str(&mut self, mut s: &str) -> fmt::Result {
        loop {
            let mut char_indices = s.char_indices();

            // This escapes quotes and new lines
            // @NOTE: I am aware this does not work for \" and such. I am ignoring that fact as we
            // are not really using such strings.
            let f = char_indices
                .find_map(|(i, c)| {
                    (|| match c {
                        '"' => {
                            self.0.write_str(&s[..i])?;
                            self.0.write_str(r#"\""#)?;
                            Ok(Some(i + 1))
                        },
                        '\n' => {
                            self.0.write_str(&s[..i])?;
                            self.0.write_str(r#"\n"#)?;
                            Ok(Some(i + 1))
                        },
                        _ => Ok(None),
                    })()
                    .transpose()
                })
                .transpose()?;

            let Some(at) = f else {
                break;
            };

            s = &s[at..];
        }

        self.0.write_str(s)?;

        Ok(())
    }
}

impl fmt::Display for IRDotDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "graph  polars_query {{")?;

        let mut last = 0;
        self._format(f, None, &mut last)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}
