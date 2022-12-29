use std::borrow::Cow;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::path::Path;

use crate::prelude::*;

fn write_scan<P: Display>(
    f: &mut fmt::Formatter,
    name: &str,
    path: &Path,
    indent: usize,
    n_columns: i64,
    total_columns: usize,
    predicate: &Option<P>,
) -> fmt::Result {
    writeln!(f, "{:indent$}{} SCAN {}", "", name, path.to_string_lossy(),)?;
    if n_columns > 0 {
        writeln!(
            f,
            "{:indent$}PROJECT {n_columns}/{total_columns} COLUMNS",
            "",
        )?;
    } else {
        writeln!(f, "{:indent$}PROJECT */{total_columns} COLUMNS", "",)?;
    }
    if let Some(predicate) = predicate {
        writeln!(f, "{:indent$}SELECTION: {predicate}", "")?;
    }
    Ok(())
}

impl LogicalPlan {
    fn _format(&self, f: &mut fmt::Formatter, mut indent: usize) -> fmt::Result {
        indent += 2;
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
                    Path::new(""),
                    indent,
                    n_columns,
                    total_columns,
                    &options.predicate,
                )
            }
            AnonymousScan {
                file_info,
                predicate,
                options,
                ..
            } => {
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len() as i64)
                    .unwrap_or(-1);
                write_scan(
                    f,
                    options.fmt_str,
                    Path::new(""),
                    indent,
                    n_columns,
                    file_info.schema.len(),
                    predicate,
                )
            }
            Union { inputs, .. } => {
                writeln!(f, "{:indent$}UNION:", "")?;
                for (i, plan) in inputs.iter().enumerate() {
                    writeln!(f, "{:indent$}PLAN {i}:", "")?;
                    plan._format(f, indent)?;
                }
                writeln!(f, "{:indent$}END UNION", "")
            }
            Cache { input, id, count } => {
                writeln!(f, "{:indent$}CACHE[id: {:x}, count: {}]", "", *id, *count)?;
                input._format(f, indent)
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                file_info,
                predicate,
                options,
                ..
            } => {
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len() as i64)
                    .unwrap_or(-1);
                write_scan(
                    f,
                    "PARQUET",
                    path,
                    indent,
                    n_columns,
                    file_info.schema.len(),
                    predicate,
                )
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                file_info,
                options,
                predicate,
                ..
            } => {
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len() as i64)
                    .unwrap_or(-1);
                write_scan(
                    f,
                    "IPC",
                    path,
                    indent,
                    n_columns,
                    file_info.schema.len(),
                    predicate,
                )
            }
            Selection { predicate, input } => {
                writeln!(f, "{:indent$}FILTER {predicate:?} FROM", "")?;
                input._format(f, indent)
            }
            Melt { input, .. } => {
                writeln!(f, "{:indent$}MELT", "")?;
                input._format(f, indent)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                options,
                file_info,
                predicate,
                ..
            } => {
                let n_columns = options
                    .with_columns
                    .as_ref()
                    .map(|columns| columns.len() as i64)
                    .unwrap_or(-1);
                write_scan(
                    f,
                    "CSV",
                    path,
                    indent,
                    n_columns,
                    file_info.schema.len(),
                    predicate,
                )
            }
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

                writeln!(
                    f,
                    "{:indent$}DF {:?}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    "",
                    schema.iter_names().take(4).collect::<Vec<_>>(),
                    n_columns,
                    total_columns,
                    selection,
                )
            }
            Projection { expr, input, .. } => {
                writeln!(f, "{:indent$} SELECT {expr:?} FROM", "")?;
                input._format(f, indent)
            }
            LocalProjection { expr, input, .. } => {
                writeln!(f, "{:indent$} LOCAL SELECT {expr:?} FROM", "")?;
                input._format(f, indent)
            }
            Sort {
                input, by_column, ..
            } => {
                writeln!(f, "{:indent$}SORT BY {by_column:?}", "")?;
                input._format(f, indent)
            }
            Explode { input, columns, .. } => {
                writeln!(f, "{:indent$}EXPLODE BY {columns:?}", "")?;
                input._format(f, indent)
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                writeln!(f, "{:indent$}Aggregate", "")?;
                writeln!(f, "{:indent$}\t{aggs:?} BY {keys:?} FROM", "")?;
                writeln!(f, "{:indent$}\t{input:?}", "")
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let how = &options.how;
                writeln!(f, "{:indent$}{how} JOIN:", "")?;
                writeln!(f, "{:indent$}LEFT PLAN ON: {left_on:?}", "")?;
                input_left._format(f, indent)?;
                writeln!(f, "{:indent$}RIGHT PLAN ON: {right_on:?}", "")?;
                input_right._format(f, indent)?;
                writeln!(f, "{:indent$}END {} JOIN", "", how)
            }
            HStack { input, exprs, .. } => {
                writeln!(f, "{:indent$} WITH_COLUMNS:", "",)?;
                writeln!(f, "{:indent$} {exprs:?}", "")?;
                input._format(f, indent)
            }
            Distinct { input, options } => {
                writeln!(f, "{:indent$}UNIQUE BY {:?}", "", options.subset)?;
                input._format(f, indent)
            }
            Slice { input, offset, len } => {
                writeln!(f, "{:indent$}SLICE[offset: {offset}, len: {len}]", "")?;
                input._format(f, indent)
            }
            MapFunction {
                input, function, ..
            } => {
                let function_fmt = format!("{function}");
                writeln!(f, "{:indent$}{function_fmt}", "")?;
                input._format(f, indent)
            }
            Error { input, err } => write!(f, "{err:?}\n{input:?}"),
            ExtContext { input, .. } => {
                writeln!(f, "{:indent$}EXTERNAL_CONTEXT", "")?;
                input._format(f, indent)
            }
            FileSink { input, .. } => {
                writeln!(f, "{:indent$}FILE_SINK", "")?;
                input._format(f, indent)
            }
        }
    }
}

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self._format(f, 0)
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
                    }
                    _ => {
                        write!(f, "{v:?}")
                    }
                }
            }
            BinaryExpr { left, op, right } => write!(f, "[({left:?}) {op:?} ({right:?})]"),
            Sort { expr, options } => match options.descending {
                true => write!(f, "{expr:?} DESC"),
                false => write!(f, "{expr:?} ASC"),
            },
            SortBy { expr, by, reverse } => {
                write!(f, "SORT {expr:?} BY {by:?} REVERSE ORDERING {reverse:?}",)
            }
            Filter { input, by } => {
                write!(f, "{input:?}\nFILTER WHERE {by:?}")
            }
            Take { expr, idx } => {
                write!(f, "TAKE {expr:?} AT {idx:?}")
            }
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
                    }
                    Max {
                        input,
                        propagate_nans,
                    } => {
                        if *propagate_nans {
                            write!(f, "{input:?}.nan_max()")
                        } else {
                            write!(f, "{input:?}.max()")
                        }
                    }
                    Median(expr) => write!(f, "{expr:?}.median()"),
                    Mean(expr) => write!(f, "{expr:?}.mean()"),
                    First(expr) => write!(f, "{expr:?}.first()"),
                    Last(expr) => write!(f, "{expr:?}.last()"),
                    List(expr) => write!(f, "{expr:?}.list()"),
                    NUnique(expr) => write!(f, "{expr:?}.n_unique()"),
                    Sum(expr) => write!(f, "{expr:?}.sum()"),
                    AggGroups(expr) => write!(f, "{expr:?}.groups()"),
                    Count(expr) => write!(f, "{expr:?}.count()"),
                    Var(expr, _) => write!(f, "{expr:?}.var()"),
                    Std(expr, _) => write!(f, "{expr:?}.var()"),
                    Quantile { expr, .. } => write!(f, "{expr:?}.quantile()"),
                }
            }
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
            }
            Ternary {
                predicate,
                truthy,
                falsy,
            } => write!(
                f,
                "\nWHEN {predicate:?}\nTHEN\n\t{truthy:?}\nOTHERWISE\n\t{falsy:?}",
            ),
            Function {
                input, function, ..
            } => {
                if input.len() >= 2 {
                    write!(f, "{:?}.{}({:?})", input[0], function, &input[1..])
                } else {
                    write!(f, "{:?}.{function}()", input[0])
                }
            }
            AnonymousFunction { input, options, .. } => {
                if input.len() >= 2 {
                    write!(f, "{:?}.{}({:?})", input[0], options.fmt_str, &input[1..])
                } else {
                    write!(f, "{:?}.{}()", input[0], options.fmt_str)
                }
            }
            Slice {
                input,
                offset,
                length,
            } => write!(f, "{input:?}.slice(offset={offset:?}, length={length:?})",),
            Wildcard => write!(f, "*"),
            Exclude(column, names) => write!(f, "{column:?}, EXCEPT {names:?}"),
            KeepName(e) => write!(f, "KEEP NAME {e:?}"),
            RenameAlias { expr, .. } => write!(f, "RENAME_ALIAS {expr:?}"),
            Columns(names) => write!(f, "COLUMNS({names:?})"),
            DtypeColumn(dt) => write!(f, "COLUMN OF DTYPE: {dt:?}"),
        }
    }
}

impl Debug for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use Operator::*;
        let s = match self {
            Eq => "==",
            NotEq => "!=",
            Lt => "<",
            LtEq => "<=",
            Gt => ">",
            GtEq => ">=",
            Plus => "+",
            Minus => "-",
            Multiply => "*",
            Divide => "/",
            TrueDivide => "/",
            FloorDivide => "//",
            Modulus => "%",
            And => "&",
            Or => "|",
            Xor => "^",
        };
        write!(f, "{s}")
    }
}

impl Debug for LiteralValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use LiteralValue::*;

        match self {
            Null => write!(f, "null"),
            Boolean(b) => write!(f, "{b}"),
            Utf8(s) => write!(f, "{s}"),
            #[cfg(feature = "dtype-binary")]
            Binary(_) => write!(f, "[binary value]"),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => write!(f, "{v}u8"),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => write!(f, "{v}u16"),
            UInt32(v) => write!(f, "{v}u32"),
            UInt64(v) => write!(f, "{v}u64"),
            #[cfg(feature = "dtype-i8")]
            Int8(v) => write!(f, "{v}i8"),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => write!(f, "{v}i16"),
            Int32(v) => write!(f, "{v}i32"),
            Int64(v) => write!(f, "{v}i64"),
            Float32(v) => write!(f, "{v}f32"),
            Float64(v) => write!(f, "{v}f64"),
            Range { low, high, .. } => write!(f, "range({low}, {high})"),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            DateTime(nd, _) => {
                write!(f, "{nd}")
            }
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            Duration(du, _) => {
                write!(f, "{du}")
            }
            Series(s) => {
                let name = s.name();
                if name.is_empty() {
                    write!(f, "Series")
                } else {
                    write!(f, "Series[{name}]")
                }
            }
        }
    }
}
