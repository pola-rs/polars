use polars_core::prelude::{PolarsError, PolarsResult};
use polars_lazy::dsl::{lit, Expr};
use sqlparser::ast::{
    Expr as SqlExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr, JoinConstraint,
    Value as SqlValue,
};

use crate::sql_expr::{apply_window_spec, parse_sql_expr};

/// Convert a SQL function to a Polars Expr
pub(crate) enum SQLFunctionExpr<'a> {
    // ----
    // Math functions
    // ----
    /// SQL 'abs' function
    Abs(&'a SQLFunction),
    /// SQL 'acos' function
    Acos(&'a SQLFunction),
    /// SQL 'asin' function
    Asin(&'a SQLFunction),
    /// SQL 'atan' function
    Atan(&'a SQLFunction),
    /// SQL 'ceil' function
    Ceil(&'a SQLFunction),
    /// SQL 'exp' function
    Exp(&'a SQLFunction),
    /// SQL 'floor' function
    Floor(&'a SQLFunction),
    /// SQL 'ln' function
    Ln(&'a SQLFunction),
    /// SQL 'log2' function
    Log2(&'a SQLFunction),
    /// SQL 'log10' function
    Log10(&'a SQLFunction),
    /// SQL 'log' function
    Log(&'a SQLFunction),

    /// SQL 'pow' function
    Pow(&'a SQLFunction),
    // ----
    // String functions
    // ----
    /// SQL 'lower' function
    Lower(&'a SQLFunction),
    /// SQL 'upper' function
    Upper(&'a SQLFunction),
    /// SQL 'ltrim' function
    LTrim(&'a SQLFunction),
    /// SQL 'rtrim' function
    RTrim(&'a SQLFunction),
    /// SQL 'starts_with' function
    StartsWith(&'a SQLFunction),
    /// SQL 'ends_with' function
    EndsWith(&'a SQLFunction),
    // ----
    // Aggregate functions
    // ----
    /// SQL 'count' function
    Count(CountExpr<'a>),
    /// SQL 'sum' function
    Sum(&'a SQLFunction),
    /// SQL 'min' function
    Min(&'a SQLFunction),
    /// SQL 'max' function
    Max(&'a SQLFunction),
    /// SQL 'avg' function
    Avg(&'a SQLFunction),
    /// SQL 'stddev' function
    StdDev(&'a SQLFunction),
    /// SQL 'variance' function
    Variance(&'a SQLFunction),
    /// SQL 'first' function
    First(&'a SQLFunction),
    /// SQL 'last' function
    Last(&'a SQLFunction),
    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function
    ArrayLength(&'a SQLFunction),
    /// SQL 'array_min' function
    ArrayMin(&'a SQLFunction),
    /// SQL 'array_max' function
    ArrayMax(&'a SQLFunction),
    /// SQL 'array_sum' function
    ArraySum(&'a SQLFunction),
    /// SQL 'array_mean' function
    ArrayMean(&'a SQLFunction),
    /// SQL 'array_reverse' function
    ArrayReverse(&'a SQLFunction),
    /// SQL 'array_unique' function
    ArrayUnique(&'a SQLFunction),
    /// SQL 'explode' function
    Explode(&'a SQLFunction),
    /// SQL 'slice' function
    Slice(&'a SQLFunction),
    /// SQL 'array_get' function
    ArrayGet(&'a SQLFunction),
    /// SQL 'array_contains' function
    ArrayContains(&'a SQLFunction),
}

impl<'a> From<&'a SQLFunction> for SQLFunctionExpr<'a> {
    fn from(function: &'a SQLFunction) -> Self {
        let function_name = function.name.0[0].value.to_lowercase();
        match function_name.as_str() {
            // ----
            // Math functions
            // ----
            "abs" => Self::Abs(function),
            "acos" => Self::Acos(function),
            "asin" => Self::Asin(function),
            "atan" => Self::Atan(function),
            "ceil" | "ceiling" => Self::Ceil(function),
            "exp" => Self::Exp(function),
            "floor" => Self::Floor(function),
            "ln" => Self::Ln(function),
            "log2" => Self::Log2(function),
            "log10" => Self::Log10(function),
            "log" => Self::Log(function),
            "pow" => Self::Pow(function),
            // ----
            // String functions
            // ----
            "lower" => Self::Lower(function),
            "upper" => Self::Upper(function),
            "ltrim" => Self::LTrim(function),
            "rtrim" => Self::RTrim(function),
            "starts_with" => Self::StartsWith(function),
            "ends_with" => Self::EndsWith(function),
            // ----
            // Aggregate functions
            // ----
            "count" => Self::Count(CountExpr(function)),
            "sum" => Self::Sum(function),
            "min" => Self::Min(function),
            "max" => Self::Max(function),
            "avg" => Self::Avg(function),
            "stddev" | "stddev_samp" => Self::StdDev(function),
            "variance" | "var_samp" => Self::Variance(function),
            "first" => Self::First(function),
            "last" => Self::Last(function),
            // ----
            // Array functions
            // ----
            "array_length" => Self::ArrayLength(function),
            "array_min" => Self::ArrayMin(function),
            "array_max" => Self::ArrayMax(function),
            "array_sum" => Self::ArraySum(function),
            "array_mean" => Self::ArrayMean(function),
            "array_reverse" => Self::ArrayReverse(function),
            "array_unique" => Self::ArrayUnique(function),
            "explode" => Self::Explode(function),
            "slice" => Self::Slice(function),
            "array_get" => Self::ArrayGet(function),
            "array_contains" => Self::ArrayContains(function),
            other => unimplemented!("{other}"),
        }
    }
}

impl TryFrom<SQLFunctionExpr<'_>> for Expr {
    type Error = PolarsError;
    fn try_from(function_expr: SQLFunctionExpr) -> Result<Self, Self::Error> {
        use SQLFunctionExpr::*;
        match function_expr {
            // ----
            // Math functions
            // ----
            Abs(function) => unary(function, Expr::abs),
            Acos(function) => unary(function, Expr::arccos),
            Asin(function) => unary(function, Expr::arcsin),
            Atan(function) => unary(function, Expr::arctan),
            Ceil(function) => unary(function, Expr::ceil),
            Exp(function) => unary(function, Expr::exp),
            Floor(function) => unary(function, Expr::floor),
            Ln(function) => unary(function, |e| e.log(std::f64::consts::E)),
            Log2(function) => unary(function, |e| e.log(2.0)),
            Log10(function) => unary(function, |e| e.log(10.0)),
            Log(function) => binary(function, Expr::log),
            Pow(function) => binary(function, |e: Expr, p: Expr| e.pow(p)),
            // ----
            // String functions
            // ----
            Lower(function) => unary(function, |e| e.str().to_lowercase()),
            Upper(function) => unary(function, |e| e.str().to_uppercase()),
            LTrim(function) => match function.args.len() {
                1 => unary(function, |e| e.str().lstrip(None)),
                2 => binary(function, |e, s| e.str().lstrip(Some(s))),
                _ => panic!(
                    "Invalid number of arguments for LTrim: {}",
                    function.args.len()
                ),
            },
            RTrim(function) => match function.args.len() {
                1 => unary(function, |e| e.str().rstrip(None)),
                2 => binary(function, |e, s| e.str().rstrip(Some(s))),
                _ => panic!(
                    "Invalid number of arguments for RTrim: {}",
                    function.args.len()
                ),
            },
            StartsWith(function) => binary(function, |e, s| e.str().starts_with(s)),
            EndsWith(function) => binary(function, |e, s| e.str().ends_with(s)),

            // ----
            // Aggregate functions
            // ----
            Count(count_expr) => count_expr.try_into(),
            Sum(function) => unary(function, Expr::sum),
            Min(function) => unary(function, Expr::min),
            Max(function) => unary(function, Expr::max),
            Avg(function) => unary(function, Expr::mean),
            StdDev(function) => unary(function, |e| e.std(1)),
            Variance(function) => unary(function, |e| e.var(1)),
            First(function) => unary(function, Expr::first),
            Last(function) => unary(function, Expr::last),
            // ----
            // Array functions
            // ----
            ArrayLength(function) => unary(function, |e| e.arr().lengths()),
            ArrayMin(function) => unary(function, |e| e.arr().min()),
            ArrayMax(function) => unary(function, |e| e.arr().max()),
            ArraySum(function) => unary(function, |e| e.arr().sum()),
            ArrayMean(function) => unary(function, |e| e.arr().mean()),
            ArrayReverse(function) => unary(function, |e| e.arr().reverse()),
            ArrayUnique(function) => unary(function, |e| e.arr().unique()),
            Explode(function) => unary(function, |e| e.explode()),
            Slice(function) => trinary(function, |e, s: Expr, l: Expr| e.slice(s, l)),
            ArrayGet(function) => binary(function, |e, i| e.arr().get(i)),
            ArrayContains(function) => binary(function, |e, i: Expr| e.arr().contains(i)),
        }
    }
}

fn unary<'a>(function: &'a SQLFunction, f: impl Fn(Expr) -> Expr) -> Result<Expr, PolarsError> {
    let args = extract_args(function);
    if let FunctionArgExpr::Expr(sql_expr) = args[0] {
        let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &function.over)?;
        Ok(f(expr))
    } else {
        not_supported_error(function.name.0[0].value.as_str(), &args)
    }
}

fn binary<'a, T: FromSqlExpr>(
    function: &'a SQLFunction,
    f: impl Fn(Expr, T) -> Expr,
) -> Result<Expr, PolarsError> {
    let args = extract_args(function);

    if let FunctionArgExpr::Expr(sql_expr) = args[0] {
        let expr1 = apply_window_spec(parse_sql_expr(sql_expr)?, &function.over)?;
        if let FunctionArgExpr::Expr(sql_expr) = args[1] {
            let expr2 = T::from_sql_expr(sql_expr)?;
            Ok(f(expr1, expr2))
        } else {
            not_supported_error(function.name.0[0].value.as_str(), &args)
        }
    } else {
        not_supported_error(function.name.0[0].value.as_str(), &args)
    }
}

fn trinary<'a, Arg1: FromSqlExpr, Arg2: FromSqlExpr>(
    function: &'a SQLFunction,
    f: impl Fn(Expr, Arg1, Arg2) -> Expr,
) -> Result<Expr, PolarsError> {
    let args = extract_args(function);

    if let [FunctionArgExpr::Expr(sql_expr1), FunctionArgExpr::Expr(sql_expr2), FunctionArgExpr::Expr(sql_expr3)] =
        &args[..]
    {
        let expr1 = apply_window_spec(parse_sql_expr(sql_expr1)?, &function.over)?;
        let arg1 = Arg1::from_sql_expr(sql_expr2)?;
        let arg2 = Arg2::from_sql_expr(sql_expr3)?;
        Ok(f(expr1, arg1, arg2))
    } else {
        not_supported_error(function.name.0[0].value.as_str(), &args)
    }
}

// CountExpr is extracted because it has many special cases
pub(crate) struct CountExpr<'a>(&'a SQLFunction);

impl TryFrom<CountExpr<'_>> for Expr {
    type Error = PolarsError;
    fn try_from(count_expr: CountExpr) -> Result<Self, Self::Error> {
        let args = extract_args(count_expr.0);
        Ok(match (args.len(), count_expr.0.distinct) {
            // count()
            (0, false) => lit(1i32).count(),
            // count(distinct)
            (0, true) => return not_supported_error("count", &args),
            (1, false) => match args[0] {
                // count(col)
                FunctionArgExpr::Expr(sql_expr) => {
                    let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &count_expr.0.over)?;
                    expr.count()
                }
                // count(*)
                FunctionArgExpr::Wildcard => lit(1i32).count(),
                // count(tbl.*) is not supported
                _ => return not_supported_error("count", &args),
            },
            (1, true) => {
                // count(distinct col)
                if let FunctionArgExpr::Expr(sql_expr) = args[0] {
                    let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &count_expr.0.over)?;
                    expr.n_unique()
                } else {
                    // count(distinct *) or count(distinct tbl.*) is not supported
                    return not_supported_error("count", &args);
                }
            }
            _ => return not_supported_error("count", &args),
        })
    }
}

fn not_supported_error(
    function_name: &str,
    args: &Vec<&sqlparser::ast::FunctionArgExpr>,
) -> PolarsResult<Expr> {
    Err(PolarsError::ComputeError(
        format!(
            "Function {:?} with args {:?} was not supported in polars-sql yet!",
            function_name, args
        )
        .into(),
    ))
}

fn extract_args(sql_function: &SQLFunction) -> Vec<&FunctionArgExpr> {
    sql_function
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Named { arg, .. } => arg,
            FunctionArg::Unnamed(arg) => arg,
        })
        .collect()
}

trait FromSqlExpr {
    fn from_sql_expr(expr: &SqlExpr) -> Result<Self, PolarsError>
    where
        Self: Sized;
}

impl FromSqlExpr for f64 {
    fn from_sql_expr(expr: &SqlExpr) -> Result<Self, PolarsError>
    where
        Self: Sized,
    {
        match expr {
            SqlExpr::Value(v) => match v {
                SqlValue::Number(s, _) => s.parse::<f64>().map_err(|_| {
                    PolarsError::ComputeError(format!("Can't parse literal {:?}", s).into())
                }),
                _ => Err(PolarsError::ComputeError(
                    format!("Can't parse literal {:?}", v).into(),
                )),
            },
            _ => Err(PolarsError::ComputeError(
                format!("Can't parse literal {:?}", expr).into(),
            )),
        }
    }
}

impl FromSqlExpr for String {
    fn from_sql_expr(expr: &SqlExpr) -> Result<Self, PolarsError>
    where
        Self: Sized,
    {
        match expr {
            SqlExpr::Value(v) => match v {
                SqlValue::SingleQuotedString(s) => Ok(s.clone()),
                _ => Err(PolarsError::ComputeError(
                    format!("Can't parse literal {:?}", v).into(),
                )),
            },
            _ => Err(PolarsError::ComputeError(
                format!("Can't parse literal {:?}", expr).into(),
            )),
        }
    }
}

impl FromSqlExpr for Expr {
    fn from_sql_expr(expr: &SqlExpr) -> Result<Self, PolarsError>
    where
        Self: Sized,
    {
        parse_sql_expr(expr)
    }
}
