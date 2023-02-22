use polars_core::prelude::{PolarsError, PolarsResult};
use polars_lazy::dsl::{lit, Expr};
use sqlparser::ast::{
    Expr as SqlExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr, Value as SqlValue,
    WindowSpec,
};

use crate::sql_expr::parse_sql_expr;

pub(crate) struct SqlFunctionVisitor<'a>(pub(crate) &'a SQLFunction);

/// SQL functions that are supported by Polars
pub(crate) enum PolarsSqlFunctions {
    // ----
    // Math functions
    // ----
    /// SQL 'abs' function
    Abs,
    /// SQL 'acos' function
    Acos,
    /// SQL 'asin' function
    Asin,
    /// SQL 'atan' function
    Atan,
    /// SQL 'ceil' function
    Ceil,
    /// SQL 'exp' function
    Exp,
    /// SQL 'floor' function
    Floor,
    /// SQL 'ln' function
    Ln,
    /// SQL 'log2' function
    Log2,
    /// SQL 'log10' function
    Log10,
    /// SQL 'log' function
    Log,

    /// SQL 'pow' function
    Pow,
    // ----
    // String functions
    // ----
    /// SQL 'lower' function
    Lower,
    /// SQL 'upper' function
    Upper,
    /// SQL 'ltrim' function
    LTrim,
    /// SQL 'rtrim' function
    RTrim,
    /// SQL 'starts_with' function
    StartsWith,
    /// SQL 'ends_with' function
    EndsWith,
    // ----
    // Aggregate functions
    // ----
    /// SQL 'count' function
    Count,
    /// SQL 'sum' function
    Sum,
    /// SQL 'min' function
    Min,
    /// SQL 'max' function
    Max,
    /// SQL 'avg' function
    Avg,
    /// SQL 'stddev' function
    StdDev,
    /// SQL 'variance' function
    Variance,
    /// SQL 'first' function
    First,
    /// SQL 'last' function
    Last,
    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function
    ArrayLength,
    /// SQL 'array_lower' function
    ArrayMin,
    /// SQL 'array_upper' function
    ArrayMax,
    /// SQL 'array_sum' function
    ArraySum,
    /// SQL 'array_mean' function
    ArrayMean,
    /// SQL 'array_reverse' function
    ArrayReverse,
    /// SQL 'array_unique' function
    ArrayUnique,
    /// SQL 'unnest' function
    Explode,
    /// SQL 'array_get' function
    ArrayGet,
    /// SQL 'array_contains' function
    ArrayContains,
}

impl TryFrom<&'_ SQLFunction> for PolarsSqlFunctions {
    type Error = PolarsError;
    fn try_from(function: &'_ SQLFunction) -> Result<Self, Self::Error> {
        let function_name = function.name.0[0].value.to_lowercase();
        Ok(match function_name.as_str() {
            // ----
            // Math functions
            // ----
            "abs" => Self::Abs,
            "acos" => Self::Acos,
            "asin" => Self::Asin,
            "atan" => Self::Atan,
            "ceil" | "ceiling" => Self::Ceil,
            "exp" => Self::Exp,
            "floor" => Self::Floor,
            "ln" => Self::Ln,
            "log2" => Self::Log2,
            "log10" => Self::Log10,
            "log" => Self::Log,
            "pow" => Self::Pow,
            // ----
            // String functions
            // ----
            "lower" => Self::Lower,
            "upper" => Self::Upper,
            "ltrim" => Self::LTrim,
            "rtrim" => Self::RTrim,
            "starts_with" => Self::StartsWith,
            "ends_with" => Self::EndsWith,
            // ----
            // Aggregate functions
            // ----
            "count" => Self::Count,
            "sum" => Self::Sum,
            "min" => Self::Min,
            "max" => Self::Max,
            "avg" => Self::Avg,
            "stddev" | "stddev_samp" => Self::StdDev,
            "variance" | "var_samp" => Self::Variance,
            "first" => Self::First,
            "last" => Self::Last,
            // ----
            // Array functions
            // ----
            "array_length" => Self::ArrayLength,
            "array_lower" => Self::ArrayMin,
            "array_upper" => Self::ArrayMax,
            "array_sum" => Self::ArraySum,
            "array_mean" => Self::ArrayMean,
            "array_reverse" => Self::ArrayReverse,
            "array_unique" => Self::ArrayUnique,
            "unnest" => Self::Explode,
            "array_get" => Self::ArrayGet,
            "array_contains" => Self::ArrayContains,
            other => {
                return Err(PolarsError::InvalidOperation(
                    format!("Unsupported SQL function: {}", other).into(),
                ))
            }
        })
    }
}
impl SqlFunctionVisitor<'_> {
    pub(crate) fn visit_function(&self) -> PolarsResult<Expr> {
        let function = self.0;

        let function_name: PolarsSqlFunctions = function.try_into()?;
        use PolarsSqlFunctions::*;
        match function_name {
            // ----
            // Math functions
            // ----
            Abs => self.visit_unary(Expr::abs),
            Acos => self.visit_unary(Expr::arccos),
            Asin => self.visit_unary(Expr::arcsin),
            Atan => self.visit_unary(Expr::arctan),
            Ceil => self.visit_unary(Expr::ceil),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Ln => self.visit_unary(|e| e.log(std::f64::consts::E)),
            Log2 => self.visit_unary(|e| e.log(2.0)),
            Log10 => self.visit_unary(|e| e.log(10.0)),
            Log => self.visit_binary(Expr::log),
            Pow => self.visit_binary::<Expr>(Expr::pow),
            // ----
            // String functions
            // ----
            Lower => self.visit_unary(|e| e.str().to_lowercase()),
            Upper => self.visit_unary(|e| e.str().to_uppercase()),
            LTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().lstrip(None)),
                2 => self.visit_binary(|e, s| e.str().lstrip(Some(s))),
                _ => panic!(
                    "Invalid number of arguments for LTrim: {}",
                    function.args.len()
                ),
            },
            RTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().rstrip(None)),
                2 => self.visit_binary(|e, s| e.str().rstrip(Some(s))),
                _ => panic!(
                    "Invalid number of arguments for RTrim: {}",
                    function.args.len()
                ),
            },
            StartsWith => self.visit_binary(|e, s| e.str().starts_with(s)),
            EndsWith => self.visit_binary(|e, s| e.str().ends_with(s)),
            // ----
            // Aggregate functions
            // ----
            Count => self.visit_count(),
            Sum => self.visit_unary(Expr::sum),
            Min => self.visit_unary(Expr::min),
            Max => self.visit_unary(Expr::max),
            Avg => self.visit_unary(Expr::mean),
            StdDev => self.visit_unary(|e| e.std(1)),
            Variance => self.visit_unary(|e| e.var(1)),
            First => self.visit_unary(Expr::first),
            Last => self.visit_unary(Expr::last),
            // ----
            // Array functions
            // ----
            ArrayLength => self.visit_unary(|e| e.arr().lengths()),
            ArrayMin => self.visit_unary(|e| e.arr().min()),
            ArrayMax => self.visit_unary(|e| e.arr().max()),
            ArraySum => self.visit_unary(|e| e.arr().sum()),
            ArrayMean => self.visit_unary(|e| e.arr().mean()),
            ArrayReverse => self.visit_unary(|e| e.arr().reverse()),
            ArrayUnique => self.visit_unary(|e| e.arr().unique()),
            Explode => self.visit_unary(|e| e.explode()),
            ArrayContains => self.visit_binary::<Expr>(|e, s| e.arr().contains(s)),
            ArrayGet => self.visit_binary(|e, i| e.arr().get(i)),
        }
    }

    fn visit_unary(&self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let function = self.0;
        let args = extract_args(function);
        if let FunctionArgExpr::Expr(sql_expr) = args[0] {
            let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &function.over)?;
            Ok(f(expr))
        } else {
            not_supported_error(function.name.0[0].value.as_str(), &args)
        }
    }

    fn visit_binary<Arg: FromSqlExpr>(&self, f: impl Fn(Expr, Arg) -> Expr) -> PolarsResult<Expr> {
        let function = self.0;
        let args = extract_args(function);
        if let FunctionArgExpr::Expr(sql_expr) = args[0] {
            let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &function.over)?;
            if let FunctionArgExpr::Expr(sql_expr) = args[1] {
                let expr2 = Arg::from_sql_expr(sql_expr)?;
                Ok(f(expr, expr2))
            } else {
                not_supported_error(function.name.0[0].value.as_str(), &args)
            }
        } else {
            not_supported_error(function.name.0[0].value.as_str(), &args)
        }
    }

    fn visit_count(&self) -> PolarsResult<Expr> {
        let args = extract_args(self.0);
        Ok(match (args.len(), self.0.distinct) {
            // count()
            (0, false) => lit(1i32).count(),
            // count(distinct)
            (0, true) => return not_supported_error("count", &args),
            (1, false) => match args[0] {
                // count(col)
                FunctionArgExpr::Expr(sql_expr) => {
                    let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &self.0.over)?;
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
                    let expr = apply_window_spec(parse_sql_expr(sql_expr)?, &self.0.over)?;
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

fn apply_window_spec(expr: Expr, window_spec: &Option<WindowSpec>) -> PolarsResult<Expr> {
    Ok(match &window_spec {
        Some(window_spec) => {
            // Process for simple window specification, partition by first
            let partition_by = window_spec
                .partition_by
                .iter()
                .map(parse_sql_expr)
                .collect::<PolarsResult<Vec<_>>>()?;
            expr.over(partition_by)
            // Order by and Row range may not be supported at the moment
        }
        None => expr,
    })
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

pub(crate) trait FromSqlExpr {
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
