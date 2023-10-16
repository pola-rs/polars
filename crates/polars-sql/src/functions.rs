use polars_core::prelude::{polars_bail, polars_err, PolarsResult};
use polars_lazy::dsl::Expr;
use polars_plan::dsl::{coalesce, count, when};
use polars_plan::logical_plan::LiteralValue;
use polars_plan::prelude::lit;
use polars_plan::prelude::LiteralValue::Null;
use sqlparser::ast::{
    Expr as SqlExpr, Function as SQLFunction, FunctionArg, FunctionArgExpr, Value as SqlValue,
    WindowSpec, WindowType,
};

use crate::sql_expr::parse_sql_expr;
use crate::SQLContext;

pub(crate) struct SqlFunctionVisitor<'a> {
    pub(crate) func: &'a SQLFunction,
    pub(crate) ctx: &'a mut SQLContext,
}

/// SQL functions that are supported by Polars
pub(crate) enum PolarsSqlFunctions {
    // ----
    // Math functions
    // ----
    /// SQL 'abs' function
    /// Returns the absolute value of the input column
    /// ```sql
    /// SELECT ABS(column_1) from df;
    /// ```
    Abs,
    /// SQL 'cos' function
    /// Compute the cosine sine of the input column (in radians)
    /// ```sql
    /// SELECT COS(column_1) from df;
    /// ```
    Cos,
    /// SQL 'cot' function
    /// Compute the cotangent of the input column (in radians)
    /// ```sql
    /// SELECT COT(column_1) from df;
    /// ```
    Cot,
    /// SQL 'sin' function
    /// Compute the sine of the input column (in radians)
    /// ```sql
    /// SELECT SIN(column_1) from df;
    /// ```
    Sin,
    /// SQL 'tan' function
    /// Compute the tangent of the input column (in radians)
    /// ```sql
    /// SELECT TAN(column_1) from df;
    /// ```
    Tan,
    /// SQL 'cosd' function
    /// Compute the cosine sine of the input column (in degrees)
    /// ```sql
    /// SELECT COSD(column_1) from df;
    /// ```
    CosD,
    /// SQL 'cotd' function
    /// Compute cotangent of the input column (in degrees)
    /// ```sql
    /// SELECT COTD(column_1) from df;
    /// ```
    CotD,
    /// SQL 'sind' function
    /// Compute the sine of the input column (in degrees)
    /// ```sql
    /// SELECT SIND(column_1) from df;
    /// ```
    SinD,
    /// SQL 'tand' function
    /// Compute the tangent of the input column (in degrees)
    /// ```sql
    /// SELECT TAND(column_1) from df;
    /// ```
    TanD,
    /// SQL 'acos' function
    /// Compute inverse cosinus of the input column (in radians)
    /// ```sql
    /// SELECT ACOS(column_1) from df;
    /// ```
    Acos,
    /// SQL 'asin' function
    /// Compute inverse sine of the input column (in radians)
    /// ```sql
    /// SELECT ASIN(column_1) from df;
    /// ```
    Asin,
    /// SQL 'atan' function
    /// Compute inverse tangent of the input column (in radians)
    /// ```sql
    /// SELECT ATAN(column_1) from df;
    /// ```
    Atan,
    /// SQL 'atan2' function
    /// Compute the inverse tangent of column_2/column_1 (in radians).
    /// ```sql
    /// SELECT ATAN2(column_1, column_2) from df;
    /// ```
    Atan2,
    /// SQL 'acosd' function
    /// Compute inverse cosinus of the input column (in degrees)
    /// ```sql
    /// SELECT ACOSD(column_1) from df;
    /// ```
    AcosD,
    /// SQL 'asind' function
    /// Compute inverse sine of the input column (in degrees)
    /// ```sql
    /// SELECT ASIND(column_1) from df;
    /// ```
    AsinD,
    /// SQL 'atand' function
    /// Compute inverse tangent of the input column (in degrees)
    /// ```sql
    /// SELECT ATAND(column_1) from df;
    /// ```
    AtanD,
    /// SQL 'atan2d' function
    /// Compute the inverse tangent of column_2/column_1 (in degrees).
    /// ```sql
    /// SELECT ATAN2D(column_1) from df;
    /// ```
    Atan2D,
    /// SQL 'ceil' function
    /// Returns the nearest integer closest from zero.
    /// ```sql
    /// SELECT CEIL(column_1) from df;
    /// ```
    Ceil,
    /// SQL 'exp' function
    /// Computes the exponential of the given value.
    /// ```sql
    /// SELECT EXP(column_1) from df;
    /// ```
    Exp,
    /// SQL 'floor' function
    /// Returns the nearest integer away from zero.
    ///   0.5 will be rounded
    /// ```sql
    /// SELECT FLOOR(column_1) from df;
    /// ```
    Floor,
    /// SQL 'pi' function
    /// Returns a (very good) approximation of 𝜋
    /// ```sql
    /// SELECT PI() from df;
    /// ```
    Pi,
    /// SQL 'ln' function
    /// Computes the natural logarithm of the given value
    /// ```sql
    /// SELECT LN(column_1) from df;
    /// ```
    Ln,
    /// SQL 'log2' function
    /// Computes the logarithm of the given value in base 2
    /// ```sql
    /// SELECT LOG2(column_1) from df;
    /// ```
    Log2,
    /// SQL 'log10' function
    /// Computes the logarithm of the given value in base 10
    /// ```sql
    /// SELECT LOG10(column_1) from df;
    /// ```
    Log10,
    /// SQL 'log' function
    /// Computes the `base` logarithm of the given value
    /// ```sql
    /// SELECT LOG(column_1, 10) from df;
    /// ```
    Log,
    /// SQL 'log1p' function
    /// Computes the natural logarithm of the “given value plus one”
    /// ```sql
    /// SELECT LOG1P(column_1) from df;
    /// ```
    Log1p,
    /// SQL 'pow' function
    /// Returns the value to the power of `exponent`
    /// ```sql
    /// SELECT POW(column_1, 2) from df;
    /// ```
    Pow,
    /// SQL 'sqrt' function
    /// Returns the square root (√) of a number
    /// ```sql
    /// SELECT SQRT(column_1) from df;
    /// ```
    Sqrt,
    /// SQL 'cbrt' function
    /// Returns the cube root (∛) of a number
    /// ```sql
    /// SELECT CBRT(column_1) from df;
    /// ```
    Cbrt,
    /// SQL 'round' function
    /// Round a number to `x` decimals (default: 0) away from zero.
    ///   .5 is rounded away from zero.
    /// ```sql
    /// SELECT ROUND(column_1, 3) from df;
    /// ```
    Round,
    /// SQL 'degrees' function
    /// Convert between radians and degrees
    /// ```sql
    /// SELECT DEGREES(column_1) from df;
    /// ```
    Degrees,
    /// SQL 'RADIANS' function
    /// Convert between degrees and radians
    /// ```sql
    /// SELECT radians(column_1) from df;
    /// ```
    Radians,

    // ----
    // String functions
    // ----
    /// SQL 'ends_with' function
    /// Returns True if the value ends with the second argument.
    /// ```sql
    /// SELECT ENDS_WITH(column_1, 'a') from df;
    /// SELECT column_2 from df WHERE ENDS_WITH(column_1, 'a');
    /// ```
    EndsWith,
    /// SQL 'initcap' function
    /// Returns the value with the first letter capitalized
    /// ```sql
    /// SELECT INITCAP(column_1) from df;
    /// ```
    InitCap,
    /// SQL 'left' function
    /// Returns the `length` first characters
    /// ```sql
    /// SELECT LEFT(column_1, 3) from df;
    /// ```
    Left,
    /// SQL 'length' function (characters)
    /// Returns the character length of the string
    /// ```sql
    /// SELECT LENGTH(column_1) from df;
    /// ```
    Length,
    /// SQL 'lower' function
    /// Returns an lowercased column
    /// ```sql
    /// SELECT LOWER(column_1) from df;
    /// ```
    Lower,
    /// SQL 'ltrim' function
    /// Strip whitespaces from the left
    /// ```sql
    /// SELECT LTRIM(column_1) from df;
    /// ```
    LTrim,
    /// SQL 'octet_length' function (bytes)
    /// ```sql
    /// SELECT OCTET_LENGTH(column_1) from df;
    /// ```
    OctetLength,
    /// SQL 'regexp_like' function
    /// True if `pattern` matches the value (optional: `flags`).
    /// ```sql
    /// SELECT REGEXP_LIKE(column_1, 'xyz', 'i') from df;
    /// ```
    RegexpLike,
    /// SQL 'rtrim' function
    /// Strip whitespaces from the right
    /// ```sql
    /// SELECT RTRIM(column_1) from df;
    /// ```
    RTrim,
    /// SQL 'starts_with' function
    /// Returns True if the value starts with the second argument.
    /// ```sql
    /// SELECT STARTS_WITH(column_1, 'a') from df;
    /// SELECT column_2 from df WHERE STARTS_WITH(column_1, 'a');
    /// ```
    StartsWith,
    /// SQL 'substr' function
    /// Returns a portion of the data (first character = 0) in the range
    ///   \[start, start + length]
    /// ```sql
    /// SELECT SUBSTR(column_1, 3, 5) from df;
    /// ```
    Substring,
    /// SQL 'upper' function
    /// Returns an uppercased column
    /// ```sql
    /// SELECT UPPER(column_1) from df;
    /// ```
    Upper,
    /// SQL 'nullif' function
    /// Returns NULL if two expressions are equal, otherwise returns the first
    /// ```sql
    /// SELECT NULLIF(column_1, column_2) from df;
    /// ```
    NullIf,
    /// SQL 'coalesce' function
    /// Returns the first non-null value in the provided values/columns
    /// ```sql
    /// SELECT COALESCE(column_1, ...) from df;
    /// ```
    Coalesce,

    // ----
    // Aggregate functions
    // ----
    /// SQL 'count' function
    /// Returns the amount of elements in the grouping
    /// ```sql
    /// SELECT COUNT(column_1) from df;
    /// SELECT COUNT(*) from df;
    /// SELECT COUNT(DISTINCT column_1) from df;
    /// SELECT COUNT(DISTINCT *) from df;
    /// ```
    Count,
    /// SQL 'sum' function
    /// Returns the sum of all the elements in the grouping
    /// ```sql
    /// SELECT SUM(column_1) from df;
    /// ```
    Sum,
    /// SQL 'min' function
    /// Returns the smallest (minimum) of all the elements in the grouping
    /// ```sql
    /// SELECT MIN(column_1) from df;
    /// ```
    Min,
    /// SQL 'max' function
    /// Returns the greatest (maximum) of all the elements in the grouping
    /// ```sql
    /// SELECT MAX(column_1) from df;
    /// ```
    Max,
    /// SQL 'avg' function
    /// Returns the average (mean) of all the elements in the grouping
    /// ```sql
    /// SELECT AVG(column_1) from df;
    /// ```
    Avg,
    /// SQL 'stddev' function
    /// Returns the standard deviation of all the elements in the grouping
    /// ```sql
    /// SELECT STDDEV(column_1) from df;
    /// ```
    StdDev,
    /// SQL 'variance' function
    /// Returns the variance of all the elements in the grouping
    /// ```sql
    /// SELECT VARIANCE(column_1) from df;
    /// ```
    Variance,
    /// SQL 'first' function
    /// Returns the first element of the grouping
    /// ```sql
    /// SELECT FIRST(column_1) from df;
    /// ```
    First,
    /// SQL 'last' function
    /// Returns the last element of the grouping
    /// ```sql
    /// SELECT LAST(column_1) from df;
    /// ```
    Last,

    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function
    /// Returns the length of the array
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
    /// Unnest/explodes an array column into multiple rows
    /// ```sql
    /// SELECT unnest(column_1) from df;
    /// ```
    Explode,
    /// SQL 'array_to_string' function
    /// Takes all elements of the array and joins them into one string
    /// ```sql
    /// SELECT ARRAY_TO_STRING(column_1, ', ') from df;
    /// ```
    ArrayToString,
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
    Udf(String),
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
            "atan2",
            "atan2d",
            "atand",
            "avg",
            "cbrt",
            "ceil",
            "ceiling",
            "coalesce",
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
            "nullif",
            "octet_length",
            "pi",
            "pow",
            "power",
            "radians",
            "round",
            "rtrim",
            "sin",
            "sind",
            "sqrt",
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

impl PolarsSqlFunctions {
    fn try_from_sql(function: &'_ SQLFunction, ctx: &'_ SQLContext) -> PolarsResult<Self> {
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
            "atan2" => Self::Atan2,
            "acosd" => Self::AcosD,
            "asind" => Self::AsinD,
            "atand" => Self::AtanD,
            "atan2d" => Self::Atan2D,
            "degrees" => Self::Degrees,
            "radians" => Self::Radians,
            "ceil" | "ceiling" => Self::Ceil,
            "exp" => Self::Exp,
            "floor" => Self::Floor,
            "pi" => Self::Pi,
            "ln" => Self::Ln,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "log1p" => Self::Log1p,
            "log2" => Self::Log2,
            "pow" | "power" => Self::Pow,
            "sqrt" => Self::Sqrt,
            "cbrt" => Self::Cbrt,
            "round" => Self::Round,

            // ----
            // Comparison functions
            // ----
            "nullif" => Self::NullIf,
            "coalesce" => Self::Coalesce,

            // ----
            // String functions
            // ----
            "ends_with" => Self::EndsWith,
            "initcap" => Self::InitCap,
            "length" => Self::Length,
            "left" => Self::Left,
            "lower" => Self::Lower,
            "ltrim" => Self::LTrim,
            "octet_length" => Self::OctetLength,
            "regexp_like" => Self::RegexpLike,
            "rtrim" => Self::RTrim,
            "starts_with" => Self::StartsWith,
            "substr" => Self::Substring,
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
            "array_to_string" => Self::ArrayToString,
            "array_unique" => Self::ArrayUnique,
            "array_upper" => Self::ArrayMax,
            "unnest" => Self::Explode,

            other => {
                if ctx.function_registry.contains(other) {
                    Self::Udf(other.to_string())
                } else {
                    polars_bail!(InvalidOperation: "unsupported SQL function: {}", other);
                }
            },
        })
    }
}

impl SqlFunctionVisitor<'_> {
    pub(crate) fn visit_function(&mut self) -> PolarsResult<Expr> {
        let function = self.func;
        let function_name = PolarsSqlFunctions::try_from_sql(function, self.ctx)?;

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
            Atan2 => self.visit_binary(Expr::arctan2),
            AcosD => self.visit_unary(|e| e.arccos().degrees()),
            AsinD => self.visit_unary(|e| e.arcsin().degrees()),
            AtanD => self.visit_unary(|e| e.arctan().degrees()),
            Atan2D => self.visit_binary(|e, s| e.arctan2(s).degrees()),
            Degrees => self.visit_unary(Expr::degrees),
            Radians => self.visit_unary(Expr::radians),
            Ceil => self.visit_unary(Expr::ceil),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Pi => self.visit_nullary(Expr::pi),
            Ln => self.visit_unary(|e| e.log(std::f64::consts::E)),
            Log => self.visit_binary(Expr::log),
            Log10 => self.visit_unary(|e| e.log(10.0)),
            Log1p => self.visit_unary(Expr::log1p),
            Log2 => self.visit_unary(|e| e.log(2.0)),
            Pow => self.visit_binary::<Expr>(Expr::pow),
            Sqrt => self.visit_unary(Expr::sqrt),
            Cbrt => self.visit_unary(Expr::cbrt),
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
            // Comparison functions
            // ----
            NullIf => self.visit_binary(|l, r: Expr| when(l.clone().eq(r)).then(lit(LiteralValue::Null)).otherwise(l)),
            Coalesce => self.visit_variadic(coalesce),

            // ----
            // String functions
            // ----
            EndsWith => self.visit_binary(|e, s| e.str().ends_with(s)),
            InitCap => self.visit_unary(|e| e.str().to_titlecase()),
            Left => self.try_visit_binary(|e, length| {
                Ok(e.str().slice(0, match length {
                    Expr::Literal(LiteralValue::Int64(n)) => Some(n as u64),
                    _ => {
                        polars_bail!(InvalidOperation: "Invalid 'length' for Left: {}", function.args[1]);
                    }
                }))
            }),
            Length => self.visit_unary(|e| e.str().len_chars()),
            Lower => self.visit_unary(|e| e.str().to_lowercase()),
            LTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().strip_chars_start(lit(Null))),
                2 => self.visit_binary(|e, s| e.str().strip_chars_start(s)),
                _ => polars_bail!(InvalidOperation:
                    "Invalid number of arguments for LTrim: {}",
                    function.args.len()
                ),
            },
            OctetLength => self.visit_unary(|e| e.str().len_bytes()),
            RegexpLike => match function.args.len() {
                2 => self.visit_binary(|e, s| e.str().contains(s, true)),
                3 => self.try_visit_ternary(|e, pat, flags| {
                    Ok(e.str().contains(
                        match (pat, flags) {
                            (Expr::Literal(LiteralValue::Utf8(s)), Expr::Literal(LiteralValue::Utf8(f))) => {
                                if f.is_empty() { polars_bail!(InvalidOperation: "Invalid/empty 'flags' for RegexpLike: {}", function.args[2]); };
                                lit(format!("(?{}){}", f, s))
                            },
                            _ => {
                                polars_bail!(InvalidOperation: "Invalid arguments for RegexpLike: {}, {}", function.args[1], function.args[2]);
                            },
                        },
                        true))
                }),
                _ => polars_bail!(InvalidOperation:"Invalid number of arguments for RegexpLike: {}",function.args.len()),
            },
            RTrim => match function.args.len() {
                1 => self.visit_unary(|e| e.str().strip_chars_end(lit(Null))),
                2 => self.visit_binary(|e, s| e.str().strip_chars_end(s)),
                _ => polars_bail!(InvalidOperation:
                    "Invalid number of arguments for RTrim: {}",
                    function.args.len()
                ),
            },
            StartsWith => self.visit_binary(|e, s| e.str().starts_with(s)),
            Substring => match function.args.len() {
                2 => self.try_visit_binary(|e, start| {
                    Ok(e.str().slice(match start {
                        Expr::Literal(LiteralValue::Int64(n)) => n,
                        _ => {
                            polars_bail!(InvalidOperation: "Invalid 'start' for Substring: {}", function.args[1]);
                        }
                    }, None))
                }),
                3 => self.try_visit_ternary(|e, start, length| {
                    Ok(e.str().slice(
                        match start {
                            Expr::Literal(LiteralValue::Int64(n)) => n,
                            _ => {
                                polars_bail!(InvalidOperation: "Invalid 'start' for Substring: {}", function.args[1]);
                            }
                        }, match length {
                            Expr::Literal(LiteralValue::Int64(n)) => Some(n as u64),
                            _ => {
                                polars_bail!(InvalidOperation: "Invalid 'length' for Substring: {}", function.args[2]);
                            }
                        }))
                }),
                _ => polars_bail!(InvalidOperation:
                    "Invalid number of arguments for Substring: {}",
                    function.args.len()
                ),
            }
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
            ArrayLength => self.visit_unary(|e| e.list().len()),
            ArrayMax => self.visit_unary(|e| e.list().max()),
            ArrayMean => self.visit_unary(|e| e.list().mean()),
            ArrayMin => self.visit_unary(|e| e.list().min()),
            ArrayReverse => self.visit_unary(|e| e.list().reverse()),
            ArraySum => self.visit_unary(|e| e.list().sum()),
            ArrayToString => self.try_visit_binary(|e, s| {
                Ok(e.list().join(s))
            }),
            ArrayUnique => self.visit_unary(|e| e.list().unique()),
            Explode => self.visit_unary(|e| e.explode()),
            Udf(func_name) => self.visit_udf(&func_name)
        }
    }

    fn visit_udf(&mut self, func_name: &str) -> PolarsResult<Expr> {
        let function = self.func;

        let args = extract_args(function);
        let args = args
            .into_iter()
            .map(|arg| {
                if let FunctionArgExpr::Expr(e) = arg {
                    parse_sql_expr(e, self.ctx)
                } else {
                    polars_bail!(ComputeError: "Only expressions are supported in UDFs")
                }
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        if let Some(expr) = self.ctx.function_registry.get_udf(func_name)? {
            expr.call(args)
        } else {
            polars_bail!(ComputeError: "UDF {} not found", func_name)
        }
    }

    fn visit_unary(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        self.visit_unary_no_window(f)
            .and_then(|e| self.apply_window_spec(e, &self.func.over))
    }

    /// Some functions have cumulative equivalents that can be applied to window specs
    /// e.g. SUM(a) OVER (ORDER BY b DESC) -> CUMSUM(a, false)
    /// visit_unary_with_cumulative_window will take in a function & a cumulative function
    /// if there is a cumulative window spec, it will apply the cumulative function,
    /// otherwise it will apply the function
    fn visit_unary_with_opt_cumulative(
        &mut self,
        f: impl Fn(Expr) -> Expr,
        cumulative_f: impl Fn(Expr, bool) -> Expr,
    ) -> PolarsResult<Expr> {
        match self.func.over.as_ref() {
            Some(WindowType::WindowSpec(spec)) => {
                self.apply_cumulative_window(f, cumulative_f, spec)
            },
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
        &mut self,
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

    fn visit_unary_no_window(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let function = self.func;

        let args = extract_args(function);
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx)?;
                // apply the function on the inner expr -- e.g. SUM(a) -> SUM
                Ok(f(expr))
            },
            _ => self.not_supported_error(),
        }
    }

    fn visit_binary<Arg: FromSqlExpr>(
        &mut self,
        f: impl Fn(Expr, Arg) -> Expr,
    ) -> PolarsResult<Expr> {
        self.try_visit_binary(|e, a| Ok(f(e, a)))
    }

    fn try_visit_binary<Arg: FromSqlExpr>(
        &mut self,
        f: impl Fn(Expr, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let function = self.func;
        let args = extract_args(function);
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr1), FunctionArgExpr::Expr(sql_expr2)] => {
                let expr1 = parse_sql_expr(sql_expr1, self.ctx)?;
                let expr2 = Arg::from_sql_expr(sql_expr2, self.ctx)?;
                f(expr1, expr2)
            },
            _ => self.not_supported_error(),
        }
    }

    fn visit_variadic(&mut self, f: impl Fn(&[Expr]) -> Expr) -> PolarsResult<Expr> {
        self.try_visit_variadic(|e| Ok(f(e)))
    }

    fn try_visit_variadic(
        &mut self,
        f: impl Fn(&[Expr]) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let function = self.func;
        let args = extract_args(function);
        let mut expr_args = vec![];
        for arg in args {
            if let FunctionArgExpr::Expr(sql_expr) = arg {
                expr_args.push(parse_sql_expr(sql_expr, self.ctx)?);
            } else {
                return self.not_supported_error();
            };
        }
        f(&expr_args)
    }

    // fn visit_ternary<Arg: FromSqlExpr>(
    //     &self,
    //     f: impl Fn(Expr, Arg, Arg) -> Expr,
    // ) -> PolarsResult<Expr> {
    //     self.try_visit_ternary(|e, a1, a2| Ok(f(e, a1, a2)))
    // }

    fn try_visit_ternary<Arg: FromSqlExpr>(
        &mut self,
        f: impl Fn(Expr, Arg, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let function = self.func;
        let args = extract_args(function);
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr1), FunctionArgExpr::Expr(sql_expr2), FunctionArgExpr::Expr(sql_expr3)] =>
            {
                let expr1 = parse_sql_expr(sql_expr1, self.ctx)?;
                let expr2 = Arg::from_sql_expr(sql_expr2, self.ctx)?;
                let expr3 = Arg::from_sql_expr(sql_expr3, self.ctx)?;
                f(expr1, expr2, expr3)
            },
            _ => self.not_supported_error(),
        }
    }

    fn visit_nullary(&self, f: impl Fn() -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func);
        if !args.is_empty() {
            return self.not_supported_error();
        }
        Ok(f())
    }

    fn visit_count(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func);
        match (self.func.distinct, args.as_slice()) {
            // count()
            (false, []) => Ok(count()),
            // count(column_name)
            (false, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx)?;
                let expr = self.apply_window_spec(expr, &self.func.over)?;
                Ok(expr.count())
            },
            // count(*)
            (false, [FunctionArgExpr::Wildcard]) => Ok(count()),
            // count(distinct column_name)
            (true, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx)?;
                let expr = self.apply_window_spec(expr, &self.func.over)?;
                Ok(expr.n_unique())
            },
            _ => self.not_supported_error(),
        }
    }

    fn apply_window_spec(
        &mut self,
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
            },
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
    fn from_sql_expr(expr: &SqlExpr, ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl FromSqlExpr for f64 {
    fn from_sql_expr(expr: &SqlExpr, _ctx: &mut SQLContext) -> PolarsResult<Self>
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
    fn from_sql_expr(expr: &SqlExpr, _: &mut SQLContext) -> PolarsResult<Self>
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
    fn from_sql_expr(expr: &SqlExpr, ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        parse_sql_expr(expr, ctx)
    }
}
