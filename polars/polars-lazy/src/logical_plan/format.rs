use crate::prelude::*;
use std::borrow::Cow;
use std::fmt;
use std::fmt::{Debug, Formatter};

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LogicalPlan::*;
        match self {
            Union { inputs, .. } => write!(f, "UNION {:?}", inputs),
            Cache { input } => write!(f, "CACHE {:?}", input),
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                options,
                ..
            } => {
                let total_columns = schema.len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "PARQUET SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
                )
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                options,
                predicate,
                ..
            } => {
                let total_columns = schema.len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "IPC SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
                )
            }
            Selection { predicate, input } => {
                write!(f, "FILTER {:?}\nFROM\n{:?}", predicate, input)
            }
            Melt { input, .. } => {
                write!(f, "MELT {:?}", input)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                options,
                schema,
                predicate,
                ..
            } => {
                let total_columns = schema.len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "CSV SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
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
                    Some(s) => Cow::Owned(format!("{:?}", s)),
                    None => Cow::Borrowed("None"),
                };

                write!(
                    f,
                    "DATAFRAME(in-memory): {:?};\n\tproject {}/{} columns\t|\tdetails: {:?};\n\
                    \tselection: {:?}\n\n",
                    schema.iter_names().take(4).collect::<Vec<_>>(),
                    n_columns,
                    total_columns,
                    projection,
                    selection,
                )
            }
            Projection { expr, input, .. } => {
                write!(
                    f,
                    "SELECT {:?} COLUMNS: {:?}
FROM
{:?}",
                    expr.len(),
                    expr,
                    input
                )
            }
            LocalProjection { expr, input, .. } => {
                write!(
                    f,
                    "LOCAL SELECT {:?} COLUMNS \nFROM\n{:?}",
                    expr.len(),
                    input
                )
            }
            Sort {
                input, by_column, ..
            } => write!(f, "SORT {:?} BY {:?}", input, by_column),
            Explode { input, columns, .. } => {
                write!(f, "EXPLODE COLUMN(S) {:?} OF {:?}", columns, input)
            }
            Aggregate {
                input, keys, aggs, ..
            } => write!(f, "Aggregate\n\t{:?} BY {:?} FROM {:?}", aggs, keys, input),
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } => write!(
                f,
                "JOIN\n\t({:?})\nWITH\n\t({:?})\nON (left: {:?} right: {:?})",
                input_left, input_right, left_on, right_on
            ),
            HStack { input, exprs, .. } => {
                write!(f, "{:?}\nWITH COLUMNS {:?}", input, exprs)
            }
            Distinct { input, .. } => write!(f, "DISTINCT {:?}", input),
            Slice { input, offset, len } => {
                write!(f, "{:?}\nSLICE[offset: {}, len: {}]", input, offset, len)
            }
            Udf { input, options, .. } => write!(f, "{} \n{:?}", options.fmt_str, input),
            Error { input, err } => write!(f, "{:?}\n{:?}", err, input),
        }
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
            } => write!(f, "{:?}.over({:?})", function, partition_by),
            Nth(i) => write!(f, "nth({})", i),
            Count => write!(f, "count()"),
            IsUnique(expr) => write!(f, "{:?}.unique()", expr),
            Explode(expr) => write!(f, "{:?}.explode()", expr),
            Duplicated(expr) => write!(f, "{:?}.is_duplicate()", expr),
            Reverse(expr) => write!(f, "{:?}.reverse()", expr),
            Alias(expr, name) => write!(f, "{:?}.alias(\"{}\")", expr, name),
            Column(name) => write!(f, "col(\"{}\")", name),
            Literal(v) => {
                match v {
                    LiteralValue::Utf8(v) => {
                        // dot breaks with debug fmt due to \"
                        write!(f, "Utf8({})", v)
                    }
                    _ => {
                        write!(f, "{:?}", v)
                    }
                }
            }
            BinaryExpr { left, op, right } => write!(f, "[({:?}) {:?} ({:?})]", left, op, right),
            Not(expr) => write!(f, "not({:?})", expr),
            IsNull(expr) => write!(f, "{:?}.is_null()", expr),
            IsNotNull(expr) => write!(f, "{:?}.is_not_null()", expr),
            Sort { expr, options } => match options.descending {
                true => write!(f, "{:?} DESC", expr),
                false => write!(f, "{:?} ASC", expr),
            },
            SortBy { expr, by, reverse } => {
                write!(
                    f,
                    "SORT {:?} BY {:?} REVERSE ORDERING {:?}",
                    expr, by, reverse
                )
            }
            Filter { input, by } => {
                write!(f, "{:?}\nFILTER WHERE {:?}", input, by)
            }
            Take { expr, idx } => {
                write!(f, "TAKE {:?} AT {:?}", expr, idx)
            }
            Agg(agg) => {
                use AggExpr::*;
                match agg {
                    Min(expr) => write!(f, "{:?}.min()", expr),
                    Max(expr) => write!(f, "{:?}.max()", expr),
                    Median(expr) => write!(f, "{:?}.median()", expr),
                    Mean(expr) => write!(f, "{:?}.mean()", expr),
                    First(expr) => write!(f, "{:?}.first()", expr),
                    Last(expr) => write!(f, "{:?}.last()", expr),
                    List(expr) => write!(f, "{:?}.list()", expr),
                    NUnique(expr) => write!(f, "{:?}.n_unique()", expr),
                    Sum(expr) => write!(f, "{:?}.sum()", expr),
                    AggGroups(expr) => write!(f, "{:?}.groups()", expr),
                    Count(expr) => write!(f, "{:?}.count()", expr),
                    Var(expr) => write!(f, "{:?}.var()", expr),
                    Std(expr) => write!(f, "{:?}.var()", expr),
                    Quantile { expr, .. } => write!(f, "{:?}.quantile()", expr),
                }
            }
            Cast {
                expr, data_type, ..
            } => write!(f, "{:?}.cast({:?})", expr, data_type),
            Ternary {
                predicate,
                truthy,
                falsy,
            } => write!(
                f,
                "\nWHEN {:?}\n\t{:?}\nOTHERWISE\n\t{:?}",
                predicate, truthy, falsy
            ),
            Function { input, options, .. } => {
                if input.len() >= 2 {
                    write!(f, "{:?}.{}({:?})", input[0], options.fmt_str, &input[1..])
                } else {
                    write!(f, "{:?}.{}()", input[0], options.fmt_str)
                }
            }
            Shift { input, periods, .. } => write!(f, "SHIFT {:?} by {}", input, periods),
            Slice {
                input,
                offset,
                length,
            } => write!(
                f,
                "{:?}.slice(offset={:?}, length={:?})",
                input, offset, length
            ),
            Wildcard => write!(f, "*"),
            Exclude(column, names) => write!(f, "{:?}, EXCEPT {:?}", column, names),
            KeepName(e) => write!(f, "KEEP NAME {:?}", e),
            RenameAlias { expr, .. } => write!(f, "RENAME_ALIAS {:?}", expr),
            Columns(names) => write!(f, "COLUMNS({:?})", names),
            DtypeColumn(dt) => write!(f, "COLUMN OF DTYPE: {:?}", dt),
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
            Divide => "//",
            TrueDivide => "/",
            Modulus => "%",
            And => "&",
            Or => "|",
            Xor => "^",
        };
        write!(f, "{}", s)
    }
}

impl Debug for LiteralValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        use LiteralValue::*;

        match self {
            Null => write!(f, "null"),
            Boolean(b) => write!(f, "{}", b),
            Utf8(s) => write!(f, "{}", s),
            #[cfg(feature = "dtype-u8")]
            UInt8(v) => write!(f, "{}u8", v),
            #[cfg(feature = "dtype-u16")]
            UInt16(v) => write!(f, "{}u16", v),
            UInt32(v) => write!(f, "{}u32", v),
            UInt64(v) => write!(f, "{}u64", v),
            #[cfg(feature = "dtype-i8")]
            Int8(v) => write!(f, "{}i8", v),
            #[cfg(feature = "dtype-i16")]
            Int16(v) => write!(f, "{}i16", v),
            Int32(v) => write!(f, "{}i32", v),
            Int64(v) => write!(f, "{}i64", v),
            Float32(v) => write!(f, "{}f32", v),
            Float64(v) => write!(f, "{}f64", v),
            Range { low, high, .. } => write!(f, "range({}, {})", low, high),
            #[cfg(all(feature = "temporal", feature = "dtype-datetime"))]
            DateTime(nd, _) => {
                write!(f, "{}", nd)
            }
            #[cfg(all(feature = "temporal", feature = "dtype-duration"))]
            Duration(du, _) => {
                write!(f, "{}", du)
            }
            Series(s) => {
                let name = s.name();
                if name.is_empty() {
                    write!(f, "Series")
                } else {
                    write!(f, "Series[{}]", name)
                }
            }
        }
    }
}
