use polars_core::prelude::{polars_bail, polars_err, PolarsError, PolarsResult};
use polars_lazy::dsl::Expr;
use polars_plan::dsl::count;
use polars_plan::logical_plan::LiteralValue;
use sqlparser::ast::{
    Expr as SqlExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr, Value as SqlValue,
    WindowSpec, WindowType,
};

use crate::sql_expr::parse_sql_expr;
use crate::SQLContext;

pub(crate) struct SqlFunctionVisitor<'a> {
    pub(crate) func: &'a SQLFunction,
    pub(crate) ctx: &'a SQLContext,
}

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
    /// SQL 'cos' function
    /// ```sql
    /// SELECT COS(column_1) from df;
    /// ```
    Cos,
    /// SQL 'cot' function
    /// ```sql
    /// SELECT COT(column_1) from df;
    /// ```
    Cot,
    /// SQL 'sin' function
    /// ```sql
    /// SELECT SIN(column_1) from df;
    /// ```
    Sin,
    /// SQL 'tan' function
    /// ```sql
    /// SELECT TAN(column_1) from df;
    /// ```
    Tan,
    /// SQL 'cosd' function
    /// ```sql
    /// SELECT COSD(column_1) from df;
    /// ```
    CosD,
    /// SQL 'cotd' function
    /// ```sql
    /// SELECT COTD(column_1) from df;
    /// ```
    CotD,
    /// SQL 'sind' function
    /// ```sql
    /// SELECT SIND(column_1) from df;
    /// ```
    SinD,
    /// SQL 'tand' function
    /// ```sql
    /// SELECT TAND(column_1) from df;
    /// ```
    TanD,
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
    /// SQL 'acosd' function
    /// ```sql
    /// SELECT ACOSD(column_1) from df;
    /// ```
    AcosD,
    /// SQL 'asind' function
    /// ```sql
    /// SELECT ASIND(column_1) from df;
    /// ```
    AsinD,
    /// SQL 'atand' function
    /// ```sql
    /// SELECT ATAND(column_1) from df;
    /// ```
    AtanD,
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
    /// SQL 'log1p' function
    /// ```sql
    /// SELECT LOG1P(column_1) from df;
    /// ```
    Log1p,
    /// SQL 'pow' function
    /// ```sql
    /// SELECT POW(column_1, 2) from df;
    /// ```
    Pow,
    /// SQL 'round' function
    /// ```sql
    /// SELECT ROUND(column_1, 3) from df;
    /// ```
    Round,
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
    /// SQL 'degrees' function
    /// ```sql
    /// SELECT DEGREES(column_1) from df;
    /// ```
    Degrees,
    /// SQL 'RADIANS' function
    /// ```sql
    /// SELECT radians(column_1) from df;
    /// ```
    Radians,
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

impl PolarsSqlFunctions {
    pub(crate) fn keywords() -> &'static [&'static str] {
        &[
            "abs",
            "acos",
            "acosd",
            "array_contains",
            "array_get",
            "array_length",
            "array_lower",
            "array_mean",
            "array_reverse",
            "array_sum",
            "array_unique",
            "array_upper",
            "asin",
            "asind",
            "atan",
            "atand",
            "avg",
            "ceil",
            "ceiling",
            "cos",
            "cosd",
            "cot",
            "cotd",
            "count",
            "degrees",
            "ends_with",
            "exp",
            "first",
            "floor",
            "last",
            "len",
            "length",
            "ln",
            "log",
            "log10",
            "log1p",
            "log2",
            "lower",
            "ltrim",
            "max",
            "min",
            "pow",
            "radians",
            "round",
            "rtrim",
            "sin",
            "sind",
            "starts_with",
            "stddev",
            "sum",
            "tan",
            "tan",
            "tand",
            "tand",
            "unnest",
            "upper",
            "variance",
        ]
    }
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
            "cos" => Self::Cos,
            "cot" => Self::Cot,
            "sin" => Self::Sin,
            "tan" => Self::Tan,
            "cosd" => Self::CosD,
            "cotd" => Self::CotD,
            "sind" => Self::SinD,
            "tand" => Self::TanD,
            "acos" => Self::Acos,
            "asin" => Self::Asin,
            "atan" => Self::Atan,
            "acosd" => Self::AcosD,
            "asind" => Self::AsinD,
            "atand" => Self::AtanD,
            "degrees" => Self::Degrees,
            "radians" => Self::Radians,
            "ceil" | "ceiling" => Self::Ceil,
            "exp" => Self::Exp,
            "floor" => Self::Floor,
            "ln" => Self::Ln,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "log1p" => Self::Log1p,
            "log2" => Self::Log2,
            "pow" => Self::Pow,
            "round" => Self::Round,
            // ----
            // String functions
            // ----
            "ends_with" => Self::EndsWith,
            "lower" => Self::Lower,
            "ltrim" => Self::LTrim,
            "rtrim" => Self::RTrim,
            "starts_with" => Self::StartsWith,
            "upper" => Self::Upper,
            // ----
            // Aggregate functions
            // ----
            "avg" => Self::Avg,
            "count" => Self::Count,
            "first" => Self::First,
            "last" => Self::Last,
            "max" => Self::Max,
            "min" => Self::Min,
            "stddev" | "stddev_samp" => Self::StdDev,
            "sum" => Self::Sum,
            "variance" | "var_samp" => Self::Variance,
            // ----
            // Array functions
            // ----
            "array_contains" => Self::ArrayContains,
            "array_get" => Self::ArrayGet,
            "array_length" => Self::ArrayLength,
            "array_lower" => Self::ArrayMin,
            "array_mean" => Self::ArrayMean,
            "array_reverse" => Self::ArrayReverse,
            "array_sum" => Self::ArraySum,
            "array_unique" => Self::ArrayUnique,
            "array_upper" => Self::ArrayMax,
            "unnest" => Self::Explode,
            other => polars_bail!(InvalidOperation: "unsupported SQL function: {}", other),
        })
    }
}

impl SqlFunctionVisitor<'_> {
    pub(crate) fn visit_function(&self) -> PolarsResult<Expr> {
        let function = self.func;

        let function_name: PolarsSqlFunctions = function.try_into()?;
        use PolarsSqlFunctions::*;

        match function_name {
            // ----
            // Math functions
            // ----
            Abs => self.visit_unary(Expr::abs),
            Cos => self.visit_unary(Expr::cos),
            Cot => self.visit_unary(Expr::cot),
            Sin => self.visit_unary(Expr::sin),
            Tan => self.visit_unary(Expr::tan),
            CosD => self.visit_unary(|e| e.radians().cos()),
            CotD => self.visit_unary(|e| e.radians().cot()),
            SinD => self.visit_unary(|e| e.radians().sin()),
            TanD => self.visit_unary(|e| e.radians().tan()),
            Acos => self.visit_unary(Expr::arccos),
            Asin => self.visit_unary(Expr::arcsin),
            Atan => self.visit_unary(Expr::arctan),
            AcosD => self.visit_unary(|e| e.arccos().degrees()),
            AsinD => self.visit_unary(|e| e.arcsin().degrees()),
            AtanD => self.visit_unary(|e| e.arctan().degrees()),
            Degrees => self.visit_unary(Expr::degrees),
            Radians => self.visit_unary(Expr::radians),
            Ceil => self.visit_unary(Expr::ceil),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Ln => self.visit_unary(|e| e.log(std::f64::consts::E)),
            Log => self.visit_binary(Expr::log),
            Log10 => self.visit_unary(|e| e.log(10.0)),
            Log1p => self.visit_unary(Expr::log1p),
            Log2 => self.visit_unary(|e| e.log(2.0)),
            Pow => self.visit_binary::<Expr>(Expr::pow),
            Round => match function.args.len() {
                1 => self.visit_unary(|e| e.round(0)),
                2 => self.try_visit_binary(|e, decimals| {
                    Ok(e.round(match decimals {
                        Expr::Literal(LiteralValue::Int64(n)) => n as u32,
                        _ => {
                            polars_bail!(InvalidOperation: "Invalid 'decimals' for Round: {}", function.args[1]);
                        }
                    }))
                }),
                _ => {
                    polars_bail!(InvalidOperation:"Invalid number of arguments for Round: {}",function.args.len());
                },
            },
            // ----
            // String functions
            // ----
            EndsWith => self.visit_binary(|e, s| e.str().ends_with(s)),
            Lower => self.visit_unary(|e| e.str().to_lowercase()),
            LTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().lstrip(None)),
                2 => self.visit_binary(|e, s| e.str().lstrip(Some(s))),
                _ => polars_bail!(InvalidOperation:
                    "Invalid number of arguments for LTrim: {}",
                    function.args.len()
                ),
            },
            RTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().rstrip(None)),
                2 => self.visit_binary(|e, s| e.str().rstrip(Some(s))),
                _ => polars_bail!(InvalidOperation:
                    "Invalid number of arguments for RTrim: {}",
                    function.args.len()
                ),
            },
            StartsWith => self.visit_binary(|e, s| e.str().starts_with(s)),
            Upper => self.visit_unary(|e| e.str().to_uppercase()),
            // ----
            // Aggregate functions
            // ----
            Avg => self.visit_unary(Expr::mean),
            Count => self.visit_count(),
            First => self.visit_unary(Expr::first),
            Last => self.visit_unary(Expr::last),
            Max => self.visit_unary_with_opt_cumulative(Expr::max, Expr::cummax),
            Min => self.visit_unary_with_opt_cumulative(Expr::min, Expr::cummin),
            StdDev => self.visit_unary(|e| e.std(1)),
            Sum => self.visit_unary_with_opt_cumulative(Expr::sum, Expr::cumsum),
            Variance => self.visit_unary(|e| e.var(1)),
            // ----
            // Array functions
            // ----
            ArrayContains => self.visit_binary::<Expr>(|e, s| e.list().contains(s)),
            ArrayGet => self.visit_binary(|e, i| e.list().get(i)),
            ArrayLength => self.visit_unary(|e| e.list().lengths()),
            ArrayMax => self.visit_unary(|e| e.list().max()),
            ArrayMean => self.visit_unary(|e| e.list().mean()),
            ArrayMin => self.visit_unary(|e| e.list().min()),
            ArrayReverse => self.visit_unary(|e| e.list().reverse()),
            ArraySum => self.visit_unary(|e| e.list().sum()),
            ArrayUnique => self.visit_unary(|e| e.list().unique()),
            Explode => self.visit_unary(|e| e.explode()),
        }
    }

    fn visit_unary(&self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        self.visit_unary_no_window(f)
            .and_then(|e| self.apply_window_spec(e, &self.func.over))
    }

    /// Some functions have cumulative equivalents that can be applied to window specs
    /// e.g. SUM(a) OVER (ORDER BY b DESC) -> CUMSUM(a, false)
    /// visit_unary_with_cumulative_window will take in a function & a cumulative function
    /// if there is a cumulative window spec, it will apply the cumulative function,
    /// otherwise it will apply the function
    fn visit_unary_with_opt_cumulative(
        &self,
        f: impl Fn(Expr) -> Expr,
        cumulative_f: impl Fn(Expr, bool) -> Expr,
    ) -> PolarsResult<Expr> {
        match self.func.over.as_ref() {
            Some(WindowType::WindowSpec(spec)) => {
                self.apply_cumulative_window(f, cumulative_f, spec)
            }
            Some(WindowType::NamedWindow(named_window)) => polars_bail!(
                InvalidOperation: "Named windows are not supported yet. Got {:?}",
                named_window
            ),
            _ => self.visit_unary(f),
        }
    }
    /// Window specs without partition bys are essentially cumulative functions
    /// e.g. SUM(a) OVER (ORDER BY b DESC) -> CUMSUM(a, false)
    fn apply_cumulative_window(
        &self,
        f: impl Fn(Expr) -> Expr,
        cumulative_f: impl Fn(Expr, bool) -> Expr,
        WindowSpec {
            partition_by,
            order_by,
            ..
        }: &WindowSpec,
    ) -> PolarsResult<Expr> {
        if !order_by.is_empty() && partition_by.is_empty() {
            let (order_by, desc): (Vec<Expr>, Vec<bool>) = order_by
                .iter()
                .map(|o| {
                    let expr = parse_sql_expr(&o.expr, self.ctx)?;
                    Ok(match o.asc {
                        Some(b) => (expr, !b),
                        None => (expr, false),
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()?
                .into_iter()
                .unzip();
            self.visit_unary_no_window(|e| cumulative_f(e.sort_by(&order_by, &desc), false))
        } else {
            self.visit_unary(f)
        }
    }

    fn visit_unary_no_window(&self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let function = self.func;

        let args = extract_args(function);
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx)?;
                // apply the function on the inner expr -- e.g. SUM(a) -> SUM
                Ok(f(expr))
            }
            _ => self.not_supported_error(),
        }
    }

    fn visit_binary<Arg: FromSqlExpr>(&self, f: impl Fn(Expr, Arg) -> Expr) -> PolarsResult<Expr> {
        self.try_visit_binary(|e, a| Ok(f(e, a)))
    }

    fn try_visit_binary<Arg: FromSqlExpr>(
        &self,
        f: impl Fn(Expr, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let function = self.func;
        let args = extract_args(function);
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr), FunctionArgExpr::Expr(sql_expr2)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx)?;
                let expr2 = Arg::from_sql_expr(sql_expr2, self.ctx)?;
                f(expr, expr2)
            }
            _ => self.not_supported_error(),
        }
    }

    fn visit_count(&self) -> PolarsResult<Expr> {
        let args = extract_args(self.func);
        match (self.func.distinct, args.as_slice()) {
            // count()
            (false, []) => Ok(count()),
            // count(column_name)
            (false, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr =
                    self.apply_window_spec(parse_sql_expr(sql_expr, self.ctx)?, &self.func.over)?;
                Ok(expr.count())
            }
            // count(*)
            (false, [FunctionArgExpr::Wildcard]) => Ok(count()),
            // count(distinct column_name)
            (true, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr =
                    self.apply_window_spec(parse_sql_expr(sql_expr, self.ctx)?, &self.func.over)?;
                Ok(expr.n_unique())
            }
            _ => self.not_supported_error(),
        }
    }

    fn apply_window_spec(
        &self,
        expr: Expr,
        window_type: &Option<WindowType>,
    ) -> PolarsResult<Expr> {
        Ok(match &window_type {
            Some(WindowType::WindowSpec(window_spec)) => {
                if window_spec.partition_by.is_empty() {
                    let exprs = window_spec
                        .order_by
                        .iter()
                        .map(|o| {
                            let e = parse_sql_expr(&o.expr, self.ctx)?;
                            match o.asc {
                                Some(b) => Ok(e.sort(!b)),
                                None => Ok(e),
                            }
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;
                    expr.over(exprs)
                } else {
                    // Process for simple window specification, partition by first
                    let partition_by = window_spec
                        .partition_by
                        .iter()
                        .map(|p| parse_sql_expr(p, self.ctx))
                        .collect::<PolarsResult<Vec<_>>>()?;
                    expr.over(partition_by)
                }
            }
            Some(WindowType::NamedWindow(named_window)) => polars_bail!(
                InvalidOperation: "Named windows are not supported yet. Got: {:?}",
                named_window
            ),
            None => expr,
        })
    }

    fn not_supported_error(&self) -> PolarsResult<Expr> {
        polars_bail!(
            InvalidOperation:
            "No function matches the given name and arguments: `{}`",
            self.func.to_string()
        );
    }
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
    fn from_sql_expr(expr: &SqlExpr, ctx: &SQLContext) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl FromSqlExpr for f64 {
    fn from_sql_expr(expr: &SqlExpr, _ctx: &SQLContext) -> PolarsResult<Self>
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
    fn from_sql_expr(expr: &SqlExpr, _: &SQLContext) -> PolarsResult<Self>
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
    fn from_sql_expr(expr: &SqlExpr, ctx: &SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        parse_sql_expr(expr, ctx)
    }
}
