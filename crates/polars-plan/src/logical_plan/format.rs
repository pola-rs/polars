use std::fmt;

use crate::prelude::*;

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Expr::*;
        match self {
            Window {
                function,
                partition_by,
                order_by,
                options,
            } => match options {
                #[cfg(feature = "dynamic_group_by")]
                WindowType::Rolling(options) => {
                    write!(
                        f,
                        "{:?}.rolling(by='{}', offset={}, period={})",
                        function, options.index_column, options.offset, options.period
                    )
                },
                _ => {
                    if let Some((order_by, _)) = order_by {
                        write!(f, "{function:?}.over(partition_by: {partition_by:?}, order_by: {order_by:?})")
                    } else {
                        write!(f, "{function:?}.over({partition_by:?})")
                    }
                },
            },
            Nth(i) => write!(f, "nth({i})"),
            Len => write!(f, "len()"),
            Explode(expr) => write!(f, "{expr:?}.explode()"),
            Alias(expr, name) => write!(f, "{expr:?}.alias(\"{name}\")"),
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
            BinaryExpr { left, op, right } => write!(f, "[({left:?}) {op:?} ({right:?})]"),
            Sort { expr, options } => {
                if options.descending {
                    write!(f, "{expr:?}.sort(desc)")
                } else {
                    write!(f, "{expr:?}.sort(asc)")
                }
            },
            SortBy {
                expr,
                by,
                sort_options,
            } => {
                write!(
                    f,
                    "{expr:?}.sort_by(by={by:?}, sort_option={sort_options:?})",
                )
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
                    write!(f, "{expr:?}.gather({idx:?})")
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
                    Count(expr, _) => write!(f, "{expr:?}.count()"),
                    Var(expr, _) => write!(f, "{expr:?}.var()"),
                    Std(expr, _) => write!(f, "{expr:?}.std()"),
                    Quantile { expr, .. } => write!(f, "{expr:?}.quantile()"),
                }
            },
            Cast {
                expr,
                data_type,
                options,
            } => {
                if options.strict() {
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
                    write!(f, "{:?}.{function}({:?})", input[0], &input[1..])
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
            IndexColumn(idxs) => write!(f, "index_columns({idxs:?})"),
            Selector(_) => write!(f, "selector"),
            #[cfg(feature = "dtype-struct")]
            Field(names) => write!(f, ".field({names:?})"),
        }
    }
}
