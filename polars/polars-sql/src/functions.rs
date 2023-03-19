use polars_core::prelude::{polars_bail, polars_err, PolarsError, PolarsResult};
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
    /// ```sql
    /// SELECT ABS(column_1) from df;
    /// ```
    Abs,
    /// SQL 'acos' function
    /// ```sql
    /// SELECT ACOS(column_1) from df;
    /// ```
    Acos,
    /// SQL 'asin' function
    /// ```sql
    /// SELECT ASIN(column_1) from df;
    /// ```
    Asin,
    /// SQL 'atan' function
    /// ```sql
    /// SELECT ATAN(column_1) from df;
    /// ```
    Atan,
    /// SQL 'ceil' function
    /// ```sql
    /// SELECT CEIL(column_1) from df;
    /// ```
    Ceil,
    /// SQL 'exp' function
    /// ```sql
    /// SELECT EXP(column_1) from df;
    /// ```
    Exp,
    /// SQL 'floor' function
    /// ```sql
    /// SELECT FLOOR(column_1) from df;
    /// ```
    Floor,
    /// SQL 'ln' function
    /// ```sql
    /// SELECT LN(column_1) from df;
    /// ```
    Ln,
    /// SQL 'log2' function
    /// ```sql
    /// SELECT LOG2(column_1) from df;
    /// ```
    Log2,
    /// SQL 'log10' function
    /// ```sql
    /// SELECT LOG10(column_1) from df;
    /// ```
    Log10,
    /// SQL 'log' function
    /// ```sql
    /// SELECT LOG(column_1, 10) from df;
    /// ```
    Log,
    /// SQL 'pow' function
    /// ```sql
    /// SELECT POW(column_1, 2) from df;
    /// ```
    Pow,
    // ----
    // String functions
    // ----
    /// SQL 'lower' function
    /// ```sql
    /// SELECT LOWER(column_1) from df;
    /// ```
    Lower,
    /// SQL 'upper' function
    /// ```sql
    /// SELECT UPPER(column_1) from df;
    /// ```
    Upper,
    /// SQL 'ltrim' function
    /// ```sql
    /// SELECT LTRIM(column_1) from df;
    /// ```
    LTrim,
    /// SQL 'rtrim' function
    /// ```sql
    /// SELECT RTRIM(column_1) from df;
    /// ```
    RTrim,
    /// SQL 'starts_with' function
    /// ```sql
    /// SELECT STARTS_WITH(column_1, 'a') from df;
    /// SELECT column_2 from df WHERE STARTS_WITH(column_1, 'a');
    /// ```
    StartsWith,
    /// SQL 'ends_with' function
    /// ```sql
    /// SELECT ENDS_WITH(column_1, 'a') from df;
    /// SELECT column_2 from df WHERE ENDS_WITH(column_1, 'a');
    /// ```
    EndsWith,
    // ----
    // Aggregate functions
    // ----
    /// SQL 'count' function
    /// ```sql
    /// SELECT COUNT(column_1) from df;
    /// SELECT COUNT(*) from df;
    /// SELECT COUNT(DISTINCT column_1) from df;
    /// SELECT COUNT(DISTINCT *) from df;
    /// ```
    Count,
    /// SQL 'sum' function
    /// ```sql
    /// SELECT SUM(column_1) from df;
    /// ```
    Sum,
    /// SQL 'min' function
    /// ```sql
    /// SELECT MIN(column_1) from df;
    /// ```
    Min,
    /// SQL 'max' function
    /// ```sql
    /// SELECT MAX(column_1) from df;
    /// ```
    Max,
    /// SQL 'avg' function
    /// ```sql
    /// SELECT AVG(column_1) from df;
    /// ```
    Avg,
    /// SQL 'stddev' function
    /// ```sql
    /// SELECT STDDEV(column_1) from df;
    /// ```
    StdDev,
    /// SQL 'variance' function
    /// ```sql
    /// SELECT VARIANCE(column_1) from df;
    /// ```
    Variance,
    /// SQL 'first' function
    /// ```sql
    /// SELECT FIRST(column_1) from df;
    /// ```
    First,
    /// SQL 'last' function
    /// ```sql
    /// SELECT LAST(column_1) from df;
    /// ```
    Last,
    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function
    /// ```sql
    /// SELECT ARRAY_LENGTH(column_1) from df;
    /// ```
    ArrayLength,
    /// SQL 'array_lower' function
    /// Returns the minimum value in an array; equivalent to `array_min`
    /// ```sql
    /// SELECT ARRAY_LOWER(column_1) from df;
    /// ```
    ArrayMin,
    /// SQL 'array_upper' function
    /// Returns the maximum value in an array; equivalent to `array_max`
    /// ```sql
    /// SELECT ARRAY_UPPER(column_1) from df;
    /// ```
    ArrayMax,
    /// SQL 'array_sum' function
    /// Returns the sum of all values in an array
    /// ```sql
    /// SELECT ARRAY_SUM(column_1) from df;
    /// ```
    ArraySum,
    /// SQL 'array_mean' function
    /// Returns the mean of all values in an array
    /// ```sql
    /// SELECT ARRAY_MEAN(column_1) from df;
    /// ```
    ArrayMean,
    /// SQL 'array_reverse' function
    /// Returns the array with the elements in reverse order
    /// ```sql
    /// SELECT ARRAY_REVERSE(column_1) from df;
    /// ```
    ArrayReverse,
    /// SQL 'array_unique' function
    /// Returns the array with the unique elements
    /// ```sql
    /// SELECT ARRAY_UNIQUE(column_1) from df;
    /// ```
    ArrayUnique,
    /// SQL 'unnest' function
    /// unnest/explods an array column into multiple rows
    /// ```sql
    /// SELECT unnest(column_1) from df;
    /// ```
    Explode,
    /// SQL 'array_get' function
    /// Returns the value at the given index in the array
    /// ```sql
    /// SELECT ARRAY_GET(column_1, 1) from df;
    /// ```
    ArrayGet,
    /// SQL 'array_contains' function
    /// Returns true if the array contains the value
    /// ```sql
    /// SELECT ARRAY_CONTAINS(column_1, 'foo') from df;
    /// ```
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
            other => polars_bail!(InvalidOperation: "unsupported SQL function: {}", other),
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
            // parse the inner sql expr -- e.g. SUM(a) -> a
            let expr = parse_sql_expr(sql_expr)?;
            // apply the function on the inner expr -- e.g. SUM(a) -> SUM
            let expr = f(expr);
            // apply the window spec if present
            apply_window_spec(expr, &function.over)
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

fn not_supported_error(function_name: &str, args: &Vec<&FunctionArgExpr>) -> PolarsResult<Expr> {
    polars_bail!(
        InvalidOperation:
        "function `{}` with args {:?} is not supported in polars-sql",
        function_name, args
    );
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
    fn from_sql_expr(expr: &SqlExpr) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl FromSqlExpr for f64 {
    fn from_sql_expr(expr: &SqlExpr) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SqlExpr::Value(v) => match v {
                SqlValue::Number(s, _) => s
                    .parse()
                    .map_err(|_| polars_err!(ComputeError: "can't parse literal {:?}", s)),
                _ => polars_bail!(ComputeError: "can't parse literal {:?}", v),
            },
            _ => polars_bail!(ComputeError: "can't parse literal {:?}", expr),
        }
    }
}

impl FromSqlExpr for String {
    fn from_sql_expr(expr: &SqlExpr) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SqlExpr::Value(v) => match v {
                SqlValue::SingleQuotedString(s) => Ok(s.clone()),
                _ => polars_bail!(ComputeError: "can't parse literal {:?}", v),
            },
            _ => polars_bail!(ComputeError: "can't parse literal {:?}", expr),
        }
    }
}

impl FromSqlExpr for Expr {
    fn from_sql_expr(expr: &SqlExpr) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        parse_sql_expr(expr)
    }
}
