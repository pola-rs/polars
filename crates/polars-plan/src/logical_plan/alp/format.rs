use std::borrow::Cow;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use crate::prelude::*;

struct ExprDisplay<'a> {
    node: Node,
    output_name: &'a OutputName,
    expr_arena: &'a Arena<AExpr>,
}

struct ExprVecDisplay<'a, T: AsExpr> {
    exprs: &'a [T],
    expr_arena: &'a Arena<AExpr>,
}

pub struct IRDisplay<'a> {
    pub root: Node,
    pub ir_arena: &'a Arena<IR>,
    pub expr_arena: &'a Arena<AExpr>,
}

trait AsExpr {
    fn node(&self) -> Node;
    fn output_name(&self) -> &OutputName;
}

impl AsExpr for Node {
    fn node(&self) -> Node {
        *self
    }
    fn output_name(&self) -> &OutputName {
        &OutputName::None
    }
}

impl AsExpr for ExprIR {
    fn node(&self) -> Node {
        self.node()
    }
    fn output_name(&self) -> &OutputName {
        self.output_name_inner()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_scan(
    f: &mut Formatter,
    name: &str,
    path: &[PathBuf],
    indent: usize,
    n_columns: i64,
    total_columns: usize,
    predicate: &Option<ExprDisplay<'_>>,
    n_rows: Option<usize>,
) -> fmt::Result {
    if indent != 0 {
        writeln!(f)?;
    }
    let path_fmt = match path.len() {
        1 => path[0].to_string_lossy(),
        0 => "".into(),
        _ => Cow::Owned(format!(
            "{} files: first file: {}",
            path.len(),
            path[0].to_string_lossy()
        )),
    };

    write!(f, "{:indent$}{name} SCAN {path_fmt}", "")?;
    if n_columns > 0 {
        write!(
            f,
            "\n{:indent$}PROJECT {n_columns}/{total_columns} COLUMNS",
            "",
        )?;
    } else {
        write!(f, "\n{:indent$}PROJECT */{total_columns} COLUMNS", "")?;
    }
    if let Some(predicate) = predicate {
        write!(f, "\n{:indent$}SELECTION: {predicate}", "")?;
    }
    if let Some(n_rows) = n_rows {
        write!(f, "\n{:indent$}N_ROWS: {n_rows}", "")?;
    }
    Ok(())
}

impl<'a> IRDisplay<'a> {
    fn _format(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
        if indent != 0 {
            writeln!(f)?;
        }
        let sub_indent = indent + 2;
        use IR::*;
        match self.root() {
            #[cfg(feature = "python")]
            PythonScan { options, predicate } => {
                let total_columns = options.schema.len();
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|s| s.len() as i64)
                    .unwrap_or(-1);

                let predicate = predicate.as_ref().map(|p| self.display_expr(p));

                write_scan(
                    f,
                    "PYTHON",
                    &[],
                    sub_indent,
                    n_columns,
                    total_columns,
                    &predicate,
                    options.n_rows,
                )
            },
            Union { inputs, options } => {
                let name = if let Some(slice) = options.slice {
                    format!("SLICED UNION: {slice:?}")
                } else {
                    "UNION".to_string()
                };

                // 3 levels of indentation
                // - 0 => UNION ... END UNION
                // - 1 => PLAN 0, PLAN 1, ... PLAN N
                // - 2 => actual formatting of plans
                let sub_sub_indent = sub_indent + 2;
                write!(f, "{:indent$}{name}", "")?;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    self.with_root(*plan)._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END {name}", "")
            },
            HConcat { inputs, .. } => {
                let sub_sub_indent = sub_indent + 2;
                write!(f, "{:indent$}HCONCAT", "")?;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    self.with_root(*plan)._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END HCONCAT", "")
            },
            Cache {
                input,
                id,
                cache_hits,
            } => {
                write!(
                    f,
                    "{:indent$}CACHE[id: {:x}, cache_hits: {}]",
                    "", *id, *cache_hits
                )?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Scan {
                paths,
                file_info,
                predicate,
                scan_type,
                file_options,
                ..
            } => {
                let n_columns = file_options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len() as i64)
                    .unwrap_or(-1);

                let predicate = predicate.as_ref().map(|p| self.display_expr(p));

                write_scan(
                    f,
                    scan_type.into(),
                    paths,
                    sub_indent,
                    n_columns,
                    file_info.schema.len(),
                    &predicate,
                    file_options.n_rows,
                )
            },
            Filter { predicate, input } => {
                let predicate = self.display_expr(predicate);
                // this one is writeln because we don't increase indent (which inserts a line)
                writeln!(f, "{:indent$}FILTER {predicate} FROM", "")?;
                self.with_root(*input)._format(f, indent)
            },
            DataFrameScan {
                schema,
                projection,
                selection,
                ..
            } => {
                let total_columns = schema.len();
                let n_columns = if let Some(columns) = projection {
                    columns.len().to_string()
                } else {
                    "*".to_string()
                };
                let selection = match selection {
                    Some(s) => Cow::Owned(self.display_expr(s).to_string()),
                    None => Cow::Borrowed("None"),
                };
                write!(
                    f,
                    "{:indent$}DF {:?}; PROJECT {}/{} COLUMNS; SELECTION: {}",
                    "",
                    schema.iter_names().take(4).collect::<Vec<_>>(),
                    n_columns,
                    total_columns,
                    selection,
                )
            },
            Select { expr, input, .. } => {
                // @NOTE: Maybe there should be a clear delimiter here?
                let default_exprs = self.display_expr_vec(expr.default_exprs());
                let cse_exprs = self.display_expr_vec(expr.cse_exprs());

                write!(f, "{:indent$} SELECT {default_exprs}, {cse_exprs} FROM", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Sort {
                input, by_column, ..
            } => {
                let by_column = self.display_expr_vec(by_column);
                write!(f, "{:indent$}SORT BY {by_column}", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            GroupBy {
                input, keys, aggs, ..
            } => {
                let aggs = self.display_expr_vec(aggs);
                let keys = self.display_expr_vec(keys);

                write!(f, "{:indent$}AGGREGATE", "")?;
                write!(f, "\n{:indent$}\t{aggs} BY {keys} FROM", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let left_on = self.display_expr_vec(left_on);
                let right_on = self.display_expr_vec(right_on);

                let how = &options.args.how;
                write!(f, "{:indent$}{how} JOIN:", "")?;
                write!(f, "\n{:indent$}LEFT PLAN ON: {left_on}", "")?;
                self.with_root(*input_left)._format(f, sub_indent)?;
                write!(f, "\n{:indent$}RIGHT PLAN ON: {right_on}", "")?;
                self.with_root(*input_right)._format(f, sub_indent)?;
                write!(f, "\n{:indent$}END {how} JOIN", "")
            },
            HStack { input, exprs, .. } => {
                // @NOTE: Maybe there should be a clear delimiter here?
                let default_exprs = self.display_expr_vec(exprs.default_exprs());
                let cse_exprs = self.display_expr_vec(exprs.cse_exprs());

                write!(f, "{:indent$} WITH_COLUMNS:", "",)?;
                write!(f, "\n{:indent$} {default_exprs}, {cse_exprs} ", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Distinct { input, options } => {
                write!(
                    f,
                    "{:indent$}UNIQUE[maintain_order: {:?}, keep_strategy: {:?}] BY {:?}",
                    "", options.maintain_order, options.keep_strategy, options.subset
                )?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Slice { input, offset, len } => {
                write!(f, "{:indent$}SLICE[offset: {offset}, len: {len}]", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            MapFunction {
                input, function, ..
            } => {
                let function_fmt = format!("{function}");
                write!(f, "{:indent$}{function_fmt}", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            ExtContext { input, .. } => {
                write!(f, "{:indent$}EXTERNAL_CONTEXT", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Sink { input, payload, .. } => {
                let name = match payload {
                    SinkType::Memory => "SINK (memory)",
                    SinkType::File { .. } => "SINK (file)",
                    #[cfg(feature = "cloud")]
                    SinkType::Cloud { .. } => "SINK (cloud)",
                };
                write!(f, "{:indent$}{name}", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            SimpleProjection { input, .. } => {
                write!(f, "{:indent$}SIMPLE_PROJECTION ", "")?;
                self.with_root(*input)._format(f, sub_indent)
            },
            Invalid => write!(f, "{:indent$}INVALID", ""),
        }
    }
}

impl<'a> IRDisplay<'a> {
    fn display_expr(&self, root: &'a ExprIR) -> ExprDisplay<'a> {
        ExprDisplay {
            node: root.node(),
            output_name: root.output_name_inner(),
            expr_arena: self.expr_arena,
        }
    }

    fn display_expr_vec(&self, exprs: &'a [ExprIR]) -> ExprVecDisplay<'a, ExprIR> {
        ExprVecDisplay {
            exprs,
            expr_arena: self.expr_arena,
        }
    }

    fn root(&self) -> &IR {
        self.ir_arena.get(self.root)
    }

    fn with_root(&self, root: Node) -> Self {
        Self {
            root,
            ir_arena: self.ir_arena,
            expr_arena: self.expr_arena,
        }
    }
}

impl<'a> ExprDisplay<'a> {
    pub(crate) fn with_vec<T: AsExpr>(&self, exprs: &'a [T]) -> ExprVecDisplay<'a, T> {
        ExprVecDisplay {
            exprs,
            expr_arena: self.expr_arena,
        }
    }

    pub(crate) fn with_root<T: AsExpr>(&self, root: &'a T) -> Self {
        Self {
            node: root.node(),
            output_name: root.output_name(),
            expr_arena: self.expr_arena,
        }
    }
}

impl<'a> Display for IRDisplay<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self._format(f, 0)
    }
}

impl<'a, T: AsExpr> Display for ExprVecDisplay<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Display items in slice delimited by a comma

        use std::fmt::Write;

        let mut iter = self.exprs.iter();

        f.write_char('[')?;
        if let Some(fst) = iter.next() {
            let fst = ExprDisplay {
                node: fst.node(),
                output_name: fst.output_name(),
                expr_arena: self.expr_arena,
            };
            write!(f, "{fst}")?;
        }

        for expr in iter {
            let expr = ExprDisplay {
                node: expr.node(),
                output_name: expr.output_name(),
                expr_arena: self.expr_arena,
            };
            write!(f, ", {expr}")?;
        }

        f.write_char(']')?;

        Ok(())
    }
}

impl<'a> Display for ExprDisplay<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let root = self.expr_arena.get(self.node);

        use AExpr::*;
        match root {
            Window {
                function,
                partition_by,
                options,
            } => {
                let function = self.with_root(function);
                let partition_by = self.with_vec(partition_by);
                match options {
                    #[cfg(feature = "dynamic_group_by")]
                    WindowType::Rolling(options) => {
                        write!(
                            f,
                            "{function}.rolling(by='{}', offset={}, period={})",
                            options.index_column, options.offset, options.period
                        )
                    },
                    _ => {
                        write!(f, "{function}.over({partition_by})")
                    },
                }
            },
            Nth(i) => write!(f, "nth({i})"),
            Len => write!(f, "len()"),
            Explode(expr) => {
                let expr = self.with_root(expr);
                write!(f, "{expr}.explode()")
            },
            Alias(expr, name) => {
                let expr = self.with_root(expr);
                write!(f, "{expr}.alias(\"{name}\")")
            },
            Column(name) => write!(f, "col(\"{name}\")"),
            Literal(v) => {
                match v {
                    LiteralValue::String(v) => {
                        // dot breaks with debug fmt due to \"
                        write!(f, "String({v})")
                    },
                    _ => {
                        write!(f, "{v:?}")
                    },
                }
            },
            BinaryExpr { left, op, right } => {
                let left = self.with_root(left);
                let right = self.with_root(right);
                write!(f, "[({left}) {op:?} ({right})]")
            },
            Sort { expr, options } => {
                let expr = self.with_root(expr);
                if options.descending {
                    write!(f, "{expr}.sort(desc)")
                } else {
                    write!(f, "{expr}.sort(asc)")
                }
            },
            SortBy {
                expr,
                by,
                sort_options,
            } => {
                let expr = self.with_root(expr);
                let by = self.with_vec(by);
                write!(f, "{expr}.sort_by(by={by}, sort_option={sort_options:?})",)
            },
            Filter { input, by } => {
                let input = self.with_root(input);
                let by = self.with_root(by);

                write!(f, "{input}.filter({by})")
            },
            Gather {
                expr,
                idx,
                returns_scalar,
            } => {
                let expr = self.with_root(expr);
                let idx = self.with_root(idx);
                expr.fmt(f)?;

                if *returns_scalar {
                    write!(f, ".get({idx})")
                } else {
                    write!(f, ".gather({idx})")
                }
            },
            Agg(agg) => {
                use AAggExpr::*;
                match agg {
                    Min {
                        input,
                        propagate_nans,
                    } => {
                        self.with_root(input).fmt(f)?;
                        if *propagate_nans {
                            write!(f, ".nan_min()")
                        } else {
                            write!(f, ".min()")
                        }
                    },
                    Max {
                        input,
                        propagate_nans,
                    } => {
                        self.with_root(input).fmt(f)?;
                        if *propagate_nans {
                            write!(f, ".nan_max()")
                        } else {
                            write!(f, ".max()")
                        }
                    },
                    Median(expr) => write!(f, "{}.median()", self.with_root(expr)),
                    Mean(expr) => write!(f, "{}.mean()", self.with_root(expr)),
                    First(expr) => write!(f, "{}.first()", self.with_root(expr)),
                    Last(expr) => write!(f, "{}.last()", self.with_root(expr)),
                    Implode(expr) => write!(f, "{}.list()", self.with_root(expr)),
                    NUnique(expr) => write!(f, "{}.n_unique()", self.with_root(expr)),
                    Sum(expr) => write!(f, "{}.sum()", self.with_root(expr)),
                    AggGroups(expr) => write!(f, "{}.groups()", self.with_root(expr)),
                    Count(expr, _) => write!(f, "{}.count()", self.with_root(expr)),
                    Var(expr, _) => write!(f, "{}.var()", self.with_root(expr)),
                    Std(expr, _) => write!(f, "{}.std()", self.with_root(expr)),
                    Quantile { expr, .. } => write!(f, "{}.quantile()", self.with_root(expr)),
                }
            },
            Cast {
                expr,
                data_type,
                strict,
            } => {
                self.with_root(expr).fmt(f)?;
                if *strict {
                    write!(f, ".strict_cast({data_type:?})")
                } else {
                    write!(f, ".cast({data_type:?})")
                }
            },
            Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.with_root(predicate);
                let truthy = self.with_root(truthy);
                let falsy = self.with_root(falsy);
                write!(f, ".when({predicate}).then({truthy}).otherwise({falsy})",)
            },
            Function {
                input, function, ..
            } => {
                let fst = self.with_root(&input[0]);
                fst.fmt(f)?;
                if input.len() >= 2 {
                    write!(f, ".{function}({})", self.with_vec(&input[1..]))
                } else {
                    write!(f, ".{function}()")
                }
            },
            AnonymousFunction { input, options, .. } => {
                let fst = self.with_root(&input[0]);
                fst.fmt(f)?;
                if input.len() >= 2 {
                    write!(f, ".{}({})", options.fmt_str, self.with_vec(&input[1..]))
                } else {
                    write!(f, ".{}()", options.fmt_str)
                }
            },
            Slice {
                input,
                offset,
                length,
            } => {
                let input = self.with_root(input);
                let offset = self.with_root(offset);
                let length = self.with_root(length);

                write!(f, "{input}.slice(offset={offset}, length={length})")
            },
            Wildcard => write!(f, "*"),
        }?;

        match self.output_name {
            OutputName::None => {},
            OutputName::LiteralLhs(_) => {},
            OutputName::ColumnLhs(_) => {},
            OutputName::Alias(name) => write!(f, r#".alias("{name}")"#)?,
        }

        Ok(())
    }
}
