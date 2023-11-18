use std::borrow::Cow;
use std::fmt;
use std::fmt::{Debug, Display, Formatter, Write};
use std::path::PathBuf;

use crate::prelude::*;

#[allow(clippy::too_many_arguments)]
fn write_scan<P: Display>(
    f: &mut Formatter,
    name: &str,
    path: &[PathBuf],
    indent: usize,
    n_columns: i64,
    total_columns: usize,
    predicate: &Option<P>,
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

    write!(f, "{:indent$}{} SCAN {}", "", name, path_fmt)?;
    if n_columns > 0 {
        write!(
            f,
            "\n{:indent$}PROJECT {n_columns}/{total_columns} COLUMNS",
            "",
        )?;
    } else {
        write!(f, "\n{:indent$}PROJECT */{total_columns} COLUMNS", "",)?;
    }
    if let Some(predicate) = predicate {
        write!(f, "\n{:indent$}SELECTION: {predicate}", "")?;
    }
    if let Some(n_rows) = n_rows {
        write!(f, "\n{:indent$}N_ROWS: {n_rows}", "")?;
    }
    Ok(())
}

impl LogicalPlan {
    fn _format(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
        if indent != 0 {
            writeln!(f)?;
        }
        let sub_indent = indent + 2;
        use LogicalPlan::*;
        match self {
            #[cfg(feature = "python")]
            PythonScan { options } => {
                let total_columns = options.schema.len();
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|s| s.len() as i64)
                    .unwrap_or(-1);

                write_scan(
                    f,
                    "PYTHON",
                    &[],
                    sub_indent,
                    n_columns,
                    total_columns,
                    &options.predicate,
                    options.n_rows,
                )
            },
            Union { inputs, options } => {
                let mut name = String::new();
                let name = if let Some(slice) = options.slice {
                    write!(name, "SLICED UNION: {:?}", slice)?;
                    name.as_str()
                } else {
                    "UNION"
                };
                // 3 levels of indentation
                // - 0 => UNION ... END UNION
                // - 1 => PLAN 0, PLAN 1, ... PLAN N
                // - 2 => actual formatting of plans
                let sub_sub_indent = sub_indent + 2;
                write!(f, "{:indent$}{}", "", name)?;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    plan._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END {}", "", name)
            },
            Cache { input, id, count } => {
                write!(f, "{:indent$}CACHE[id: {:x}, count: {}]", "", *id, *count)?;
                input._format(f, sub_indent)
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
                write_scan(
                    f,
                    scan_type.into(),
                    paths,
                    sub_indent,
                    n_columns,
                    file_info.schema.len(),
                    predicate,
                    file_options.n_rows,
                )
            },
            Selection { predicate, input } => {
                // this one is writeln because we don't increase indent (which inserts a line)
                writeln!(f, "{:indent$}FILTER {predicate:?} FROM", "")?;
                input._format(f, indent)
            },
            DataFrameScan {
                schema,
                projection,
                selection,
                ..
            } => {
                let total_columns = schema.len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = projection {
                    n_columns = format!("{}", columns.len());
                }
                let selection = match selection {
                    Some(s) => Cow::Owned(format!("{s:?}")),
                    None => Cow::Borrowed("None"),
                };
                write!(
                    f,
                    "{:indent$}DF {:?}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    "",
                    schema.iter_names().take(4).collect::<Vec<_>>(),
                    n_columns,
                    total_columns,
                    selection,
                )
            },
            Projection { expr, input, .. } => {
                write!(f, "{:indent$} SELECT {expr:?} FROM", "")?;
                input._format(f, sub_indent)
            },
            Sort {
                input, by_column, ..
            } => {
                write!(f, "{:indent$}SORT BY {by_column:?}", "")?;
                input._format(f, sub_indent)
            },
            Aggregate {
                input, keys, aggs, ..
            } => {
                write!(f, "{:indent$}AGGREGATE", "")?;
                write!(f, "\n{:indent$}\t{aggs:?} BY {keys:?} FROM", "")?;
                input._format(f, sub_indent)
            },
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let how = &options.args.how;
                write!(f, "{:indent$}{how} JOIN:", "")?;
                write!(f, "\n{:indent$}LEFT PLAN ON: {left_on:?}", "")?;
                input_left._format(f, sub_indent)?;
                write!(f, "\n{:indent$}RIGHT PLAN ON: {right_on:?}", "")?;
                input_right._format(f, sub_indent)?;
                write!(f, "\n{:indent$}END {} JOIN", "", how)
            },
            HStack { input, exprs, .. } => {
                write!(f, "{:indent$} WITH_COLUMNS:", "",)?;
                write!(f, "\n{:indent$} {exprs:?}", "")?;
                input._format(f, sub_indent)
            },
            Distinct { input, options } => {
                write!(f, "{:indent$}UNIQUE BY {:?}", "", options.subset)?;
                input._format(f, sub_indent)
            },
            Slice { input, offset, len } => {
                write!(f, "{:indent$}SLICE[offset: {offset}, len: {len}]", "")?;
                input._format(f, sub_indent)
            },
            MapFunction {
                input, function, ..
            } => {
                let function_fmt = format!("{function}");
                write!(f, "{:indent$}{function_fmt}", "")?;
                input._format(f, sub_indent)
            },
            Error { input, err } => write!(f, "{err:?}\n{input:?}"),
            ExtContext { input, .. } => {
                write!(f, "{:indent$}EXTERNAL_CONTEXT", "")?;
                input._format(f, sub_indent)
            },
            Sink { input, payload, .. } => {
                let name = match payload {
                    SinkType::Memory => "SINK (memory)",
                    SinkType::File { .. } => "SINK (file)",
                    #[cfg(feature = "cloud")]
                    SinkType::Cloud { .. } => "SINK (cloud)",
                };
                write!(f, "{:indent$}{}", "", name)?;
                input._format(f, sub_indent)
            },
        }
    }
}

impl Debug for LogicalPlan {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self._format(f, 0)
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Debug for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Expr::*;
        match self {
            Window {
                function,
                partition_by,
                ..
            } => write!(f, "{function:?}.over({partition_by:?})"),
            Nth(i) => write!(f, "nth({i})"),
            Count => write!(f, "count()"),
            Explode(expr) => write!(f, "{expr:?}.explode()"),
            Alias(expr, name) => write!(f, "{expr:?}.alias(\"{name}\")"),
            Column(name) => write!(f, "col(\"{name}\")"),
            Literal(v) => {
                match v {
                    LiteralValue::Utf8(v) => {
                        // dot breaks with debug fmt due to \"
                        write!(f, "Utf8({v})")
                    },
                    _ => {
                        write!(f, "{v:?}")
                    },
                }
            },
            BinaryExpr { left, op, right } => write!(f, "[({left:?}) {op:?} ({right:?})]"),
            Sort { expr, options } => match options.descending {
                true => write!(f, "{expr:?}.sort(desc)"),
                false => write!(f, "{expr:?}.sort(asc)"),
            },
            SortBy {
                expr,
                by,
                descending,
            } => {
                write!(f, "{expr:?}.sort_by(by={by:?}, descending={descending:?})",)
            },
            Filter { input, by } => {
                write!(f, "{input:?}.filter({by:?})")
            },
            Gather {
                expr,
                idx,
                returns_scalar,
            } => {
                if *returns_scalar {
                    write!(f, "{expr:?}.get({idx:?})")
                } else {
                    write!(f, "{expr:?}.take({idx:?})")
                }
            },
            SubPlan(lf, _) => {
                write!(f, ".subplan({lf:?})")
            },
            Agg(agg) => {
                use AggExpr::*;
                match agg {
                    Min {
                        input,
                        propagate_nans,
                    } => {
                        if *propagate_nans {
                            write!(f, "{input:?}.nan_min()")
                        } else {
                            write!(f, "{input:?}.min()")
                        }
                    },
                    Max {
                        input,
                        propagate_nans,
                    } => {
                        if *propagate_nans {
                            write!(f, "{input:?}.nan_max()")
                        } else {
                            write!(f, "{input:?}.max()")
                        }
                    },
                    Median(expr) => write!(f, "{expr:?}.median()"),
                    Mean(expr) => write!(f, "{expr:?}.mean()"),
                    First(expr) => write!(f, "{expr:?}.first()"),
                    Last(expr) => write!(f, "{expr:?}.last()"),
                    Implode(expr) => write!(f, "{expr:?}.list()"),
                    NUnique(expr) => write!(f, "{expr:?}.n_unique()"),
                    Sum(expr) => write!(f, "{expr:?}.sum()"),
                    AggGroups(expr) => write!(f, "{expr:?}.groups()"),
                    Count(expr) => write!(f, "{expr:?}.count()"),
                    Var(expr, _) => write!(f, "{expr:?}.var()"),
                    Std(expr, _) => write!(f, "{expr:?}.std()"),
                    Quantile { expr, .. } => write!(f, "{expr:?}.quantile()"),
                }
            },
            Cast {
                expr,
                data_type,
                strict,
            } => {
                if *strict {
                    write!(f, "{expr:?}.strict_cast({data_type:?})")
                } else {
                    write!(f, "{expr:?}.cast({data_type:?})")
                }
            },
            Ternary {
                predicate,
                truthy,
                falsy,
            } => write!(
                f,
                ".when({predicate:?}).then({truthy:?}).otherwise({falsy:?})",
            ),
            Function {
                input, function, ..
            } => {
                if input.len() >= 2 {
                    write!(f, "{:?}.{}({:?})", input[0], function, &input[1..])
                } else {
                    write!(f, "{:?}.{function}()", input[0])
                }
            },
            AnonymousFunction { input, options, .. } => {
                if input.len() >= 2 {
                    write!(f, "{:?}.{}({:?})", input[0], options.fmt_str, &input[1..])
                } else {
                    write!(f, "{:?}.{}()", input[0], options.fmt_str)
                }
            },
            Slice {
                input,
                offset,
                length,
            } => write!(f, "{input:?}.slice(offset={offset:?}, length={length:?})",),
            Wildcard => write!(f, "*"),
            Exclude(column, names) => write!(f, "{column:?}.exclude({names:?})"),
            KeepName(e) => write!(f, "{e:?}.name.keep()"),
            RenameAlias { expr, .. } => write!(f, ".rename_alias({expr:?})"),
            Columns(names) => write!(f, "cols({names:?})"),
            DtypeColumn(dt) => write!(f, "dtype_columns({dt:?})"),
            Selector(_) => write!(f, "SELECTOR"),
        }
    }
}

impl Debug for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Debug for LiteralValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use LiteralValue::*;

        match self {
            Binary(_) => write!(f, "[binary value]"),
            Range { low, high, .. } => write!(f, "range({low}, {high})"),
            Series(s) => {
                let name = s.name();
                if name.is_empty() {
                    write!(f, "Series")
                } else {
                    write!(f, "Series[{name}]")
                }
            },
            _ => {
                let av = self.to_anyvalue().unwrap();
                write!(f, "{av}")
            },
        }
    }
}
