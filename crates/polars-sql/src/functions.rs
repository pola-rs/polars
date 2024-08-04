use std::ops::Sub;

use polars_core::chunked_array::ops::{SortMultipleOptions, SortOptions};
use polars_core::export::regex;
use polars_core::prelude::{polars_bail, polars_err, DataType, PolarsResult, Schema, TimeUnit};
use polars_lazy::dsl::Expr;
#[cfg(feature = "list_eval")]
use polars_lazy::dsl::ListNameSpaceExtension;
use polars_plan::dsl::{coalesce, concat_str, len, max_horizontal, min_horizontal, when};
use polars_plan::plans::{typed_lit, LiteralValue};
use polars_plan::prelude::LiteralValue::Null;
use polars_plan::prelude::{col, cols, lit, StrptimeOptions};
use sqlparser::ast::{
    DateTimeField, DuplicateTreatment, Expr as SQLExpr, Function as SQLFunction, FunctionArg,
    FunctionArgExpr, FunctionArgumentClause, FunctionArgumentList, FunctionArguments, Ident,
    OrderByExpr, Value as SQLValue, WindowSpec, WindowType,
};

use crate::sql_expr::{adjust_one_indexed_param, parse_extract_date_part, parse_sql_expr};
use crate::SQLContext;

pub(crate) struct SQLFunctionVisitor<'a> {
    pub(crate) func: &'a SQLFunction,
    pub(crate) ctx: &'a mut SQLContext,
    pub(crate) active_schema: Option<&'a Schema>,
}

/// SQL functions that are supported by Polars
pub(crate) enum PolarsSQLFunctions {
    // ----
    // Math functions
    // ----
    /// SQL 'abs' function
    /// Returns the absolute value of the input column.
    /// ```sql
    /// SELECT ABS(column_1) FROM df;
    /// ```
    Abs,
    /// SQL 'ceil' function
    /// Returns the nearest integer closest from zero.
    /// ```sql
    /// SELECT CEIL(column_1) FROM df;
    /// ```
    Ceil,
    /// SQL 'div' function
    /// Returns the integer quotient of the division.
    /// ```sql
    /// SELECT DIV(column_1, 2) FROM df;
    /// ```
    Div,
    /// SQL 'exp' function
    /// Computes the exponential of the given value.
    /// ```sql
    /// SELECT EXP(column_1) FROM df;
    /// ```
    Exp,
    /// SQL 'floor' function
    /// Returns the nearest integer away from zero.
    ///   0.5 will be rounded
    /// ```sql
    /// SELECT FLOOR(column_1) FROM df;
    /// ```
    Floor,
    /// SQL 'pi' function
    /// Returns a (very good) approximation of 𝜋.
    /// ```sql
    /// SELECT PI() FROM df;
    /// ```
    Pi,
    /// SQL 'ln' function
    /// Computes the natural logarithm of the given value.
    /// ```sql
    /// SELECT LN(column_1) FROM df;
    /// ```
    Ln,
    /// SQL 'log2' function
    /// Computes the logarithm of the given value in base 2.
    /// ```sql
    /// SELECT LOG2(column_1) FROM df;
    /// ```
    Log2,
    /// SQL 'log10' function
    /// Computes the logarithm of the given value in base 10.
    /// ```sql
    /// SELECT LOG10(column_1) FROM df;
    /// ```
    Log10,
    /// SQL 'log' function
    /// Computes the `base` logarithm of the given value.
    /// ```sql
    /// SELECT LOG(column_1, 10) FROM df;
    /// ```
    Log,
    /// SQL 'log1p' function
    /// Computes the natural logarithm of "given value plus one".
    /// ```sql
    /// SELECT LOG1P(column_1) FROM df;
    /// ```
    Log1p,
    /// SQL 'pow' function
    /// Returns the value to the power of the given exponent.
    /// ```sql
    /// SELECT POW(column_1, 2) FROM df;
    /// ```
    Pow,
    /// SQL 'mod' function
    /// Returns the remainder of a numeric expression divided by another numeric expression.
    /// ```sql
    /// SELECT MOD(column_1, 2) FROM df;
    /// ```
    Mod,
    /// SQL 'sqrt' function
    /// Returns the square root (√) of a number.
    /// ```sql
    /// SELECT SQRT(column_1) FROM df;
    /// ```
    Sqrt,
    /// SQL 'cbrt' function
    /// Returns the cube root (∛) of a number.
    /// ```sql
    /// SELECT CBRT(column_1) FROM df;
    /// ```
    Cbrt,
    /// SQL 'round' function
    /// Round a number to `x` decimals (default: 0) away from zero.
    ///   .5 is rounded away from zero.
    /// ```sql
    /// SELECT ROUND(column_1, 3) FROM df;
    /// ```
    Round,
    /// SQL 'sign' function
    /// Returns the sign of the argument as -1, 0, or +1.
    /// ```sql
    /// SELECT SIGN(column_1) FROM df;
    /// ```
    Sign,

    // ----
    // Trig functions
    // ----
    /// SQL 'cos' function
    /// Compute the cosine sine of the input column (in radians).
    /// ```sql
    /// SELECT COS(column_1) FROM df;
    /// ```
    Cos,
    /// SQL 'cot' function
    /// Compute the cotangent of the input column (in radians).
    /// ```sql
    /// SELECT COT(column_1) FROM df;
    /// ```
    Cot,
    /// SQL 'sin' function
    /// Compute the sine of the input column (in radians).
    /// ```sql
    /// SELECT SIN(column_1) FROM df;
    /// ```
    Sin,
    /// SQL 'tan' function
    /// Compute the tangent of the input column (in radians).
    /// ```sql
    /// SELECT TAN(column_1) FROM df;
    /// ```
    Tan,
    /// SQL 'cosd' function
    /// Compute the cosine sine of the input column (in degrees).
    /// ```sql
    /// SELECT COSD(column_1) FROM df;
    /// ```
    CosD,
    /// SQL 'cotd' function
    /// Compute cotangent of the input column (in degrees).
    /// ```sql
    /// SELECT COTD(column_1) FROM df;
    /// ```
    CotD,
    /// SQL 'sind' function
    /// Compute the sine of the input column (in degrees).
    /// ```sql
    /// SELECT SIND(column_1) FROM df;
    /// ```
    SinD,
    /// SQL 'tand' function
    /// Compute the tangent of the input column (in degrees).
    /// ```sql
    /// SELECT TAND(column_1) FROM df;
    /// ```
    TanD,
    /// SQL 'acos' function
    /// Compute inverse cosinus of the input column (in radians).
    /// ```sql
    /// SELECT ACOS(column_1) FROM df;
    /// ```
    Acos,
    /// SQL 'asin' function
    /// Compute inverse sine of the input column (in radians).
    /// ```sql
    /// SELECT ASIN(column_1) FROM df;
    /// ```
    Asin,
    /// SQL 'atan' function
    /// Compute inverse tangent of the input column (in radians).
    /// ```sql
    /// SELECT ATAN(column_1) FROM df;
    /// ```
    Atan,
    /// SQL 'atan2' function
    /// Compute the inverse tangent of column_2/column_1 (in radians).
    /// ```sql
    /// SELECT ATAN2(column_1, column_2) FROM df;
    /// ```
    Atan2,
    /// SQL 'acosd' function
    /// Compute inverse cosinus of the input column (in degrees).
    /// ```sql
    /// SELECT ACOSD(column_1) FROM df;
    /// ```
    AcosD,
    /// SQL 'asind' function
    /// Compute inverse sine of the input column (in degrees).
    /// ```sql
    /// SELECT ASIND(column_1) FROM df;
    /// ```
    AsinD,
    /// SQL 'atand' function
    /// Compute inverse tangent of the input column (in degrees).
    /// ```sql
    /// SELECT ATAND(column_1) FROM df;
    /// ```
    AtanD,
    /// SQL 'atan2d' function
    /// Compute the inverse tangent of column_2/column_1 (in degrees).
    /// ```sql
    /// SELECT ATAN2D(column_1) FROM df;
    /// ```
    Atan2D,
    /// SQL 'degrees' function
    /// Convert between radians and degrees.
    /// ```sql
    /// SELECT DEGREES(column_1) FROM df;
    /// ```
    ///
    ///
    Degrees,
    /// SQL 'RADIANS' function
    /// Convert between degrees and radians.
    /// ```sql
    /// SELECT RADIANS(column_1) FROM df;
    /// ```
    Radians,

    // ----
    // Temporal functions
    // ----
    /// SQL 'date_part' function.
    /// Extracts a part of a date (or datetime) such as 'year', 'month', etc.
    /// ```sql
    /// SELECT DATE_PART('year', column_1) FROM df;
    /// SELECT DATE_PART('day', column_1) FROM df;
    DatePart,
    /// SQL 'strftime' function.
    /// Converts a datetime to a string using a format string.
    /// ```sql
    /// SELECT STRFTIME(column_1, '%d-%m-%Y %H:%M') FROM df;
    /// ```
    Strftime,

    // ----
    // String functions
    // ----
    /// SQL 'bit_length' function (bytes).
    /// ```sql
    /// SELECT BIT_LENGTH(column_1) FROM df;
    /// ```
    BitLength,
    /// SQL 'concat' function
    /// Returns all input expressions concatenated together as a string.
    /// ```sql
    /// SELECT CONCAT(column_1, column_2) FROM df;
    /// ```
    Concat,
    /// SQL 'concat_ws' function
    /// Returns all input expressions concatenated together
    /// (and interleaved with a separator) as a string.
    /// ```sql
    /// SELECT CONCAT_WS(':', column_1, column_2, column_3) FROM df;
    /// ```
    ConcatWS,
    /// SQL 'date' function.
    /// Converts a formatted string date to an actual Date type; ISO-8601 format is assumed
    /// unless a strftime-compatible formatting string is provided as the second parameter.
    /// ```sql
    /// SELECT DATE('2021-03-15') FROM df;
    /// SELECT DATE('2021-15-03', '%Y-d%-%m') FROM df;
    /// SELECT DATE('2021-03', '%Y-%m') FROM df;
    /// ```
    Date,
    /// SQL 'timestamp' function.
    /// Converts a formatted string datetime to an actual Datetime type; ISO-8601 format is
    /// assumed unless a strftime-compatible formatting string is provided as the second
    /// parameter.
    /// ```sql
    /// SELECT TIMESTAMP('2021-03-15 10:30:45') FROM df;
    /// SELECT TIMESTAMP('2021-15-03T00:01:02.333', '%Y-d%-%m %H:%M:%S') FROM df;
    /// ```
    Timestamp,
    /// SQL 'ends_with' function
    /// Returns True if the value ends with the second argument.
    /// ```sql
    /// SELECT ENDS_WITH(column_1, 'a') FROM df;
    /// SELECT column_2 from df WHERE ENDS_WITH(column_1, 'a');
    /// ```
    EndsWith,
    /// SQL 'initcap' function
    /// Returns the value with the first letter capitalized.
    /// ```sql
    /// SELECT INITCAP(column_1) FROM df;
    /// ```
    #[cfg(feature = "nightly")]
    InitCap,
    /// SQL 'left' function
    /// Returns the first (leftmost) `n` characters.
    /// ```sql
    /// SELECT LEFT(column_1, 3) FROM df;
    /// ```
    Left,
    /// SQL 'length' function (characters)
    /// Returns the character length of the string.
    /// ```sql
    /// SELECT LENGTH(column_1) FROM df;
    /// ```
    Length,
    /// SQL 'lower' function
    /// Returns an lowercased column.
    /// ```sql
    /// SELECT LOWER(column_1) FROM df;
    /// ```
    Lower,
    /// SQL 'ltrim' function
    /// Strip whitespaces from the left.
    /// ```sql
    /// SELECT LTRIM(column_1) FROM df;
    /// ```
    LTrim,
    /// SQL 'octet_length' function
    /// Returns the length of a given string in bytes.
    /// ```sql
    /// SELECT OCTET_LENGTH(column_1) FROM df;
    /// ```
    OctetLength,
    /// SQL 'regexp_like' function
    /// True if `pattern` matches the value (optional: `flags`).
    /// ```sql
    /// SELECT REGEXP_LIKE(column_1, 'xyz', 'i') FROM df;
    /// ```
    RegexpLike,
    /// SQL 'replace' function
    /// Replace a given substring with another string.
    /// ```sql
    /// SELECT REPLACE(column_1,'old','new') FROM df;
    /// ```
    Replace,
    /// SQL 'reverse' function
    /// Return the reversed string.
    /// ```sql
    /// SELECT REVERSE(column_1) FROM df;
    /// ```
    Reverse,
    /// SQL 'right' function
    /// Returns the last (rightmost) `n` characters.
    /// ```sql
    /// SELECT RIGHT(column_1, 3) FROM df;
    /// ```
    Right,
    /// SQL 'rtrim' function
    /// Strip whitespaces from the right.
    /// ```sql
    /// SELECT RTRIM(column_1) FROM df;
    /// ```
    RTrim,
    /// SQL 'starts_with' function
    /// Returns True if the value starts with the second argument.
    /// ```sql
    /// SELECT STARTS_WITH(column_1, 'a') FROM df;
    /// SELECT column_2 from df WHERE STARTS_WITH(column_1, 'a');
    /// ```
    StartsWith,
    /// SQL 'strpos' function
    /// Returns the index of the given substring in the target string.
    /// ```sql
    /// SELECT STRPOS(column_1,'xyz') FROM df;
    /// ```
    StrPos,
    /// SQL 'substr' function
    /// Returns a portion of the data (first character = 0) in the range.
    ///   \[start, start + length]
    /// ```sql
    /// SELECT SUBSTR(column_1, 3, 5) FROM df;
    /// ```
    Substring,
    /// SQL 'strptime' function
    /// Converts a string to a datetime using a format string.
    /// ```sql
    /// SELECT STRPTIME(column_1, '%d-%m-%Y %H:%M') FROM df;
    /// ```
    Strptime,
    /// SQL 'time' function.
    /// Converts a formatted string time to an actual Time type; ISO-8601 format is
    /// assumed unless a strftime-compatible formatting string is provided as the second
    /// parameter.
    /// ```sql
    /// SELECT TIME('10:30:45') FROM df;
    /// SELECT TIME('20.30', '%H.%M') FROM df;
    /// ```
    Time,
    /// SQL 'upper' function
    /// Returns an uppercased column.
    /// ```sql
    /// SELECT UPPER(column_1) FROM df;
    /// ```
    Upper,

    // ----
    // Conditional functions
    // ----
    /// SQL 'coalesce' function
    /// Returns the first non-null value in the provided values/columns.
    /// ```sql
    /// SELECT COALESCE(column_1, ...) FROM df;
    /// ```
    Coalesce,
    /// SQL 'greatest' function
    /// Returns the greatest value in the list of expressions.
    /// ```sql
    /// SELECT GREATEST(column_1, column_2, ...) FROM df;
    /// ```
    Greatest,
    /// SQL 'if' function
    /// Returns expr1 if the boolean condition provided as the first
    /// parameter evaluates to true, and expr2 otherwise.
    /// ```sql
    /// SELECT IF(column < 0, expr1, expr2) FROM df;
    /// ```
    If,
    /// SQL 'ifnull' function
    /// If an expression value is NULL, return an alternative value.
    /// ```sql
    /// SELECT IFNULL(string_col, 'n/a') FROM df;
    /// ```
    IfNull,
    /// SQL 'least' function
    /// Returns the smallest value in the list of expressions.
    /// ```sql
    /// SELECT LEAST(column_1, column_2, ...) FROM df;
    /// ```
    Least,
    /// SQL 'nullif' function
    /// Returns NULL if two expressions are equal, otherwise returns the first.
    /// ```sql
    /// SELECT NULLIF(column_1, column_2) FROM df;
    /// ```
    NullIf,

    // ----
    // Aggregate functions
    // ----
    /// SQL 'avg' function
    /// Returns the average (mean) of all the elements in the grouping.
    /// ```sql
    /// SELECT AVG(column_1) FROM df;
    /// ```
    Avg,
    /// SQL 'count' function
    /// Returns the amount of elements in the grouping.
    /// ```sql
    /// SELECT COUNT(column_1) FROM df;
    /// SELECT COUNT(*) FROM df;
    /// SELECT COUNT(DISTINCT column_1) FROM df;
    /// SELECT COUNT(DISTINCT *) FROM df;
    /// ```
    Count,
    /// SQL 'first' function
    /// Returns the first element of the grouping.
    /// ```sql
    /// SELECT FIRST(column_1) FROM df;
    /// ```
    First,
    /// SQL 'last' function
    /// Returns the last element of the grouping.
    /// ```sql
    /// SELECT LAST(column_1) FROM df;
    /// ```
    Last,
    /// SQL 'max' function
    /// Returns the greatest (maximum) of all the elements in the grouping.
    /// ```sql
    /// SELECT MAX(column_1) FROM df;
    /// ```
    Max,
    /// SQL 'median' function
    /// Returns the median element from the grouping.
    /// ```sql
    /// SELECT MEDIAN(column_1) FROM df;
    /// ```
    Median,
    /// SQL 'min' function
    /// Returns the smallest (minimum) of all the elements in the grouping.
    /// ```sql
    /// SELECT MIN(column_1) FROM df;
    /// ```
    Min,
    /// SQL 'stddev' function
    /// Returns the standard deviation of all the elements in the grouping.
    /// ```sql
    /// SELECT STDDEV(column_1) FROM df;
    /// ```
    StdDev,
    /// SQL 'sum' function
    /// Returns the sum of all the elements in the grouping.
    /// ```sql
    /// SELECT SUM(column_1) FROM df;
    /// ```
    Sum,
    /// SQL 'variance' function
    /// Returns the variance of all the elements in the grouping.
    /// ```sql
    /// SELECT VARIANCE(column_1) FROM df;
    /// ```
    Variance,

    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function
    /// Returns the length of the array.
    /// ```sql
    /// SELECT ARRAY_LENGTH(column_1) FROM df;
    /// ```
    ArrayLength,
    /// SQL 'array_lower' function
    /// Returns the minimum value in an array; equivalent to `array_min`.
    /// ```sql
    /// SELECT ARRAY_LOWER(column_1) FROM df;
    /// ```
    ArrayMin,
    /// SQL 'array_upper' function
    /// Returns the maximum value in an array; equivalent to `array_max`.
    /// ```sql
    /// SELECT ARRAY_UPPER(column_1) FROM df;
    /// ```
    ArrayMax,
    /// SQL 'array_sum' function
    /// Returns the sum of all values in an array.
    /// ```sql
    /// SELECT ARRAY_SUM(column_1) FROM df;
    /// ```
    ArraySum,
    /// SQL 'array_mean' function
    /// Returns the mean of all values in an array.
    /// ```sql
    /// SELECT ARRAY_MEAN(column_1) FROM df;
    /// ```
    ArrayMean,
    /// SQL 'array_reverse' function
    /// Returns the array with the elements in reverse order.
    /// ```sql
    /// SELECT ARRAY_REVERSE(column_1) FROM df;
    /// ```
    ArrayReverse,
    /// SQL 'array_unique' function
    /// Returns the array with the unique elements.
    /// ```sql
    /// SELECT ARRAY_UNIQUE(column_1) FROM df;
    /// ```
    ArrayUnique,
    /// SQL 'unnest' function
    /// Unnest/explodes an array column into multiple rows.
    /// ```sql
    /// SELECT unnest(column_1) FROM df;
    /// ```
    Explode,
    /// SQL 'array_agg' function
    /// Concatenates the input expressions, including nulls, into an array.
    /// ```sql
    /// SELECT ARRAY_AGG(column_1, column_2, ...) FROM df;
    /// ```
    ArrayAgg,
    /// SQL 'array_to_string' function
    /// Takes all elements of the array and joins them into one string.
    /// ```sql
    /// SELECT ARRAY_TO_STRING(column_1, ',') FROM df;
    /// SELECT ARRAY_TO_STRING(column_1, ',', 'n/a') FROM df;
    /// ```
    ArrayToString,
    /// SQL 'array_get' function
    /// Returns the value at the given index in the array.
    /// ```sql
    /// SELECT ARRAY_GET(column_1, 1) FROM df;
    /// ```
    ArrayGet,
    /// SQL 'array_contains' function
    /// Returns true if the array contains the value.
    /// ```sql
    /// SELECT ARRAY_CONTAINS(column_1, 'foo') FROM df;
    /// ```
    ArrayContains,

    // ----
    // Column selection
    // ----
    Columns,

    // ----
    // User-defined
    // ----
    Udf(String),
}

impl PolarsSQLFunctions {
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
            "array_to_string",
            "array_unique",
            "array_upper",
            "asin",
            "asind",
            "atan",
            "atan2",
            "atan2d",
            "atand",
            "avg",
            "bit_length",
            "cbrt",
            "ceil",
            "ceiling",
            "char_length",
            "character_length",
            "coalesce",
            "columns",
            "concat",
            "concat_ws",
            "cos",
            "cosd",
            "cot",
            "cotd",
            "count",
            "date",
            "date_part",
            "degrees",
            "ends_with",
            "exp",
            "first",
            "floor",
            "greatest",
            "if",
            "ifnull",
            "initcap",
            "last",
            "least",
            "left",
            "length",
            "ln",
            "log",
            "log10",
            "log1p",
            "log2",
            "lower",
            "ltrim",
            "max",
            "median",
            "min",
            "mod",
            "nullif",
            "octet_length",
            "pi",
            "pow",
            "power",
            "radians",
            "regexp_like",
            "replace",
            "reverse",
            "right",
            "round",
            "rtrim",
            "sign",
            "sin",
            "sind",
            "sqrt",
            "starts_with",
            "stddev",
            "stddev_samp",
            "stdev",
            "stdev_samp",
            "strftime",
            "strpos",
            "strptime",
            "substr",
            "sum",
            "tan",
            "tand",
            "unnest",
            "upper",
            "var",
            "var_samp",
            "variance",
        ]
    }
}

impl PolarsSQLFunctions {
    fn try_from_sql(function: &'_ SQLFunction, ctx: &'_ SQLContext) -> PolarsResult<Self> {
        let function_name = function.name.0[0].value.to_lowercase();
        Ok(match function_name.as_str() {
            // ----
            // Math functions
            // ----
            "abs" => Self::Abs,
            "cbrt" => Self::Cbrt,
            "ceil" | "ceiling" => Self::Ceil,
            "div" => Self::Div,
            "exp" => Self::Exp,
            "floor" => Self::Floor,
            "ln" => Self::Ln,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "log1p" => Self::Log1p,
            "log2" => Self::Log2,
            "mod" => Self::Mod,
            "pi" => Self::Pi,
            "pow" | "power" => Self::Pow,
            "round" => Self::Round,
            "sign" => Self::Sign,
            "sqrt" => Self::Sqrt,

            // ----
            // Trig functions
            // ----
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

            // ----
            // Conditional functions
            // ----
            "coalesce" => Self::Coalesce,
            "greatest" => Self::Greatest,
            "if" => Self::If,
            "ifnull" => Self::IfNull,
            "least" => Self::Least,
            "nullif" => Self::NullIf,

            // ----
            // Date functions
            // ----
            "date_part" => Self::DatePart,
            "strftime" => Self::Strftime,

            // ----
            // String functions
            // ----
            "bit_length" => Self::BitLength,
            "concat" => Self::Concat,
            "concat_ws" => Self::ConcatWS,
            "date" => Self::Date,
            "timestamp" | "datetime" => Self::Timestamp,
            "ends_with" => Self::EndsWith,
            #[cfg(feature = "nightly")]
            "initcap" => Self::InitCap,
            "length" | "char_length" | "character_length" => Self::Length,
            "left" => Self::Left,
            "lower" => Self::Lower,
            "ltrim" => Self::LTrim,
            "octet_length" => Self::OctetLength,
            "strpos" => Self::StrPos,
            "regexp_like" => Self::RegexpLike,
            "replace" => Self::Replace,
            "reverse" => Self::Reverse,
            "right" => Self::Right,
            "rtrim" => Self::RTrim,
            "starts_with" => Self::StartsWith,
            "strptime" => Self::Strptime,
            "substr" => Self::Substring,
            "time" => Self::Time,
            "upper" => Self::Upper,

            // ----
            // Aggregate functions
            // ----
            "avg" => Self::Avg,
            "count" => Self::Count,
            "first" => Self::First,
            "last" => Self::Last,
            "max" => Self::Max,
            "median" => Self::Median,
            "min" => Self::Min,
            "stdev" | "stddev" | "stdev_samp" | "stddev_samp" => Self::StdDev,
            "sum" => Self::Sum,
            "var" | "variance" | "var_samp" => Self::Variance,

            // ----
            // Array functions
            // ----
            "array_agg" => Self::ArrayAgg,
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

            // ----
            // Column selection
            // ----
            "columns" => Self::Columns,

            other => {
                if ctx.function_registry.contains(other) {
                    Self::Udf(other.to_string())
                } else {
                    polars_bail!(SQLInterface: "unsupported function '{}'", other);
                }
            },
        })
    }
}

impl SQLFunctionVisitor<'_> {
    pub(crate) fn visit_function(&mut self) -> PolarsResult<Expr> {
        use PolarsSQLFunctions::*;
        let function_name = PolarsSQLFunctions::try_from_sql(self.func, self.ctx)?;
        let function = self.func;

        // TODO: implement the following functions where possible
        if !function.within_group.is_empty() {
            polars_bail!(SQLInterface: "'WITHIN GROUP' is not currently supported")
        }
        if function.filter.is_some() {
            polars_bail!(SQLInterface: "'FILTER' is not currently supported")
        }
        if function.null_treatment.is_some() {
            polars_bail!(SQLInterface: "'IGNORE|RESPECT NULLS' is not currently supported")
        }

        match function_name {
            // ----
            // Math functions
            // ----
            Abs => self.visit_unary(Expr::abs),
            Cbrt => self.visit_unary(Expr::cbrt),
            Ceil => self.visit_unary(Expr::ceil),
            Div => self.visit_binary(|e, d| e.floor_div(d).cast(DataType::Int64)),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Ln => self.visit_unary(|e| e.log(std::f64::consts::E)),
            Log => self.visit_binary(Expr::log),
            Log10 => self.visit_unary(|e| e.log(10.0)),
            Log1p => self.visit_unary(Expr::log1p),
            Log2 => self.visit_unary(|e| e.log(2.0)),
            Pi => self.visit_nullary(Expr::pi),
            Mod => self.visit_binary(|e1, e2| e1 % e2),
            Pow => self.visit_binary::<Expr>(Expr::pow),
            Round => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.round(0)),
                    2 => self.try_visit_binary(|e, decimals| {
                        Ok(e.round(match decimals {
                            Expr::Literal(LiteralValue::Int(n)) => {
                                if n >= 0 { n as u32 } else {
                                    polars_bail!(SQLInterface: "ROUND does not currently support negative decimals value ({})", args[1])
                                }
                            },
                            _ => polars_bail!(SQLSyntax: "invalid value for ROUND decimals ({})", args[1]),
                        }))
                    }),
                    _ => polars_bail!(SQLSyntax: "ROUND expects 1-2 arguments (found {})", args.len()),
                }
            },
            Sign => self.visit_unary(Expr::sign),
            Sqrt => self.visit_unary(Expr::sqrt),

            // ----
            // Trig functions
            // ----
            Acos => self.visit_unary(Expr::arccos),
            AcosD => self.visit_unary(|e| e.arccos().degrees()),
            Asin => self.visit_unary(Expr::arcsin),
            AsinD => self.visit_unary(|e| e.arcsin().degrees()),
            Atan => self.visit_unary(Expr::arctan),
            Atan2 => self.visit_binary(Expr::arctan2),
            Atan2D => self.visit_binary(|e, s| e.arctan2(s).degrees()),
            AtanD => self.visit_unary(|e| e.arctan().degrees()),
            Cos => self.visit_unary(Expr::cos),
            CosD => self.visit_unary(|e| e.radians().cos()),
            Cot => self.visit_unary(Expr::cot),
            CotD => self.visit_unary(|e| e.radians().cot()),
            Degrees => self.visit_unary(Expr::degrees),
            Radians => self.visit_unary(Expr::radians),
            Sin => self.visit_unary(Expr::sin),
            SinD => self.visit_unary(|e| e.radians().sin()),
            Tan => self.visit_unary(Expr::tan),
            TanD => self.visit_unary(|e| e.radians().tan()),

            // ----
            // Conditional functions
            // ----
            Coalesce => self.visit_variadic(coalesce),
            Greatest => self.visit_variadic(|exprs: &[Expr]| max_horizontal(exprs).unwrap()),
            If => {
                let args = extract_args(function)?;
                match args.len() {
                    3 => self.try_visit_ternary(|cond: Expr, expr1: Expr, expr2: Expr| {
                        Ok(when(cond).then(expr1).otherwise(expr2))
                    }),
                    _ => {
                        polars_bail!(SQLSyntax: "IF expects 3 arguments (found {})", args.len()
                        )
                    },
                }
            },
            IfNull => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_variadic(coalesce),
                    _ => {
                        polars_bail!(SQLSyntax: "IFNULL expects 2 arguments (found {})", args.len())
                    },
                }
            },
            Least => self.visit_variadic(|exprs: &[Expr]| min_horizontal(exprs).unwrap()),
            NullIf => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|l: Expr, r: Expr| {
                        when(l.clone().eq(r))
                            .then(lit(LiteralValue::Null))
                            .otherwise(l)
                    }),
                    _ => {
                        polars_bail!(SQLSyntax: "NULLIF expects 2 arguments (found {})", args.len())
                    },
                }
            },

            // ----
            // Date functions
            // ----
            DatePart => self.try_visit_binary(|part, e| {
                match part {
                    Expr::Literal(LiteralValue::String(p)) => {
                        // note: 'DATE_PART' and 'EXTRACT' are minor syntactic
                        // variations on otherwise identical functionality
                        parse_extract_date_part(
                            e,
                            &DateTimeField::Custom(Ident {
                                value: p,
                                quote_style: None,
                            }),
                        )
                    },
                    _ => {
                        polars_bail!(SQLSyntax: "invalid 'part' for EXTRACT/DATE_PART ({})", part);
                    },
                }
            }),
            Strftime => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, fmt: String| e.dt().strftime(fmt.as_str())),
                    _ => {
                        polars_bail!(SQLSyntax: "STRFTIME expects 2 arguments (found {})", args.len())
                    },
                }
            },

            // ----
            // String functions
            // ----
            BitLength => self.visit_unary(|e| e.str().len_bytes() * lit(8)),
            Concat => {
                let args = extract_args(function)?;
                if args.is_empty() {
                    polars_bail!(SQLSyntax: "CONCAT expects at least 1 argument (found 0)");
                } else {
                    self.visit_variadic(|exprs: &[Expr]| concat_str(exprs, "", true))
                }
            },
            ConcatWS => {
                let args = extract_args(function)?;
                if args.len() < 2 {
                    polars_bail!(SQLSyntax: "CONCAT_WS expects at least 2 arguments (found {})", args.len());
                } else {
                    self.try_visit_variadic(|exprs: &[Expr]| {
                        match &exprs[0] {
                            Expr::Literal(LiteralValue::String(s)) => Ok(concat_str(&exprs[1..], s, true)),
                            _ => polars_bail!(SQLSyntax: "CONCAT_WS 'separator' must be a literal string (found {:?})", exprs[0]),
                        }
                    })
                }
            },
            Date => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.str().to_date(StrptimeOptions::default())),
                    2 => self.visit_binary(|e, fmt| e.str().to_date(fmt)),
                    _ => {
                        polars_bail!(SQLSyntax: "DATE expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            EndsWith => self.visit_binary(|e, s| e.str().ends_with(s)),
            #[cfg(feature = "nightly")]
            InitCap => self.visit_unary(|e| e.str().to_titlecase()),
            Left => self.try_visit_binary(|e, length| {
                Ok(match length {
                    Expr::Literal(Null) => lit(Null),
                    Expr::Literal(LiteralValue::Int(0)) => lit(""),
                    Expr::Literal(LiteralValue::Int(n)) => {
                        let len = if n > 0 {
                            lit(n)
                        } else {
                            (e.clone().str().len_chars() + lit(n)).clip_min(lit(0))
                        };
                        e.str().slice(lit(0), len)
                    },
                    Expr::Literal(v) => {
                        polars_bail!(SQLSyntax: "invalid 'n_chars' for LEFT ({:?})", v)
                    },
                    _ => when(length.clone().gt_eq(lit(0)))
                        .then(e.clone().str().slice(lit(0), length.clone().abs()))
                        .otherwise(e.clone().str().slice(
                            lit(0),
                            (e.clone().str().len_chars() + length.clone()).clip_min(lit(0)),
                        )),
                })
            }),
            Length => self.visit_unary(|e| e.str().len_chars()),
            Lower => self.visit_unary(|e| e.str().to_lowercase()),
            LTrim => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.str().strip_chars_start(lit(Null))),
                    2 => self.visit_binary(|e, s| e.str().strip_chars_start(s)),
                    _ => {
                        polars_bail!(SQLSyntax: "LTRIM expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            OctetLength => self.visit_unary(|e| e.str().len_bytes()),
            StrPos => {
                // // note: SQL is 1-indexed; returns zero if no match found
                self.visit_binary(|expr, substring| {
                    (expr.str().find(substring, true) + typed_lit(1u32)).fill_null(typed_lit(0u32))
                })
            },
            RegexpLike => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, s| e.str().contains(s, true)),
                    3 => self.try_visit_ternary(|e, pat, flags| {
                        Ok(e.str().contains(
                            match (pat, flags) {
                                (Expr::Literal(LiteralValue::String(s)), Expr::Literal(LiteralValue::String(f))) => {
                                    if f.is_empty() {
                                        polars_bail!(SQLSyntax: "invalid/empty 'flags' for REGEXP_LIKE ({})", args[2]);
                                    };
                                    lit(format!("(?{}){}", f, s))
                                },
                                _ => {
                                    polars_bail!(SQLSyntax: "invalid arguments for REGEXP_LIKE ({}, {})", args[1], args[2]);
                                },
                            },
                            true))
                    }),
                    _ => polars_bail!(SQLSyntax: "REGEXP_LIKE expects 2-3 arguments (found {})",args.len()),
                }
            },
            Replace => {
                let args = extract_args(function)?;
                match args.len() {
                    3 => self
                        .try_visit_ternary(|e, old, new| Ok(e.str().replace_all(old, new, true))),
                    _ => {
                        polars_bail!(SQLSyntax: "REPLACE expects 3 arguments (found {})", args.len())
                    },
                }
            },
            Reverse => self.visit_unary(|e| e.str().reverse()),
            Right => self.try_visit_binary(|e, length| {
                Ok(match length {
                    Expr::Literal(Null) => lit(Null),
                    Expr::Literal(LiteralValue::Int(0)) => typed_lit(""),
                    Expr::Literal(LiteralValue::Int(n)) => {
                        let n: i64 = n.try_into().unwrap();
                        let offset = if n < 0 {
                            lit(n.abs())
                        } else {
                            e.clone().str().len_chars().cast(DataType::Int32) - lit(n)
                        };
                        e.str().slice(offset, lit(Null))
                    },
                    Expr::Literal(v) => {
                        polars_bail!(SQLSyntax: "invalid 'n_chars' for RIGHT ({:?})", v)
                    },
                    _ => when(length.clone().lt(lit(0)))
                        .then(e.clone().str().slice(length.clone().abs(), lit(Null)))
                        .otherwise(e.clone().str().slice(
                            e.clone().str().len_chars().cast(DataType::Int32) - length.clone(),
                            lit(Null),
                        )),
                })
            }),
            RTrim => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.str().strip_chars_end(lit(Null))),
                    2 => self.visit_binary(|e, s| e.str().strip_chars_end(s)),
                    _ => {
                        polars_bail!(SQLSyntax: "RTRIM expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            StartsWith => self.visit_binary(|e, s| e.str().starts_with(s)),
            Strptime => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, fmt| {
                        e.str().strptime(
                            DataType::Datetime(TimeUnit::Microseconds, None),
                            StrptimeOptions {
                                format: Some(fmt),
                                ..Default::default()
                            },
                            lit("latest"),
                        )
                    }),
                    _ => {
                        polars_bail!(SQLSyntax: "STRPTIME expects 2 arguments (found {})", args.len())
                    },
                }
            },
            Time => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.str().to_time(StrptimeOptions::default())),
                    2 => self.visit_binary(|e, fmt| e.str().to_time(fmt)),
                    _ => {
                        polars_bail!(SQLSyntax: "TIME expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            Timestamp => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| {
                        e.str()
                            .to_datetime(None, None, StrptimeOptions::default(), lit("latest"))
                    }),
                    2 => self
                        .visit_binary(|e, fmt| e.str().to_datetime(None, None, fmt, lit("latest"))),
                    _ => {
                        polars_bail!(SQLSyntax: "DATETIME expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            Substring => {
                let args = extract_args(function)?;
                match args.len() {
                    // note: SQL is 1-indexed, hence the need for adjustments
                    2 => self.try_visit_binary(|e, start| {
                        Ok(match start {
                            Expr::Literal(Null) => lit(Null),
                            Expr::Literal(LiteralValue::Int(n)) if n <= 0 => e,
                            Expr::Literal(LiteralValue::Int(n)) => e.str().slice(lit(n - 1), lit(Null)),
                            Expr::Literal(_) => polars_bail!(SQLSyntax: "invalid 'start' for SUBSTR ({})", args[1]),
                            _ => start.clone() + lit(1),
                        })
                    }),
                    3 => self.try_visit_ternary(|e: Expr, start: Expr, length: Expr| {
                        Ok(match (start.clone(), length.clone()) {
                            (Expr::Literal(Null), _) | (_, Expr::Literal(Null)) => lit(Null),
                            (_, Expr::Literal(LiteralValue::Int(n))) if n < 0 => {
                                polars_bail!(SQLSyntax: "SUBSTR does not support negative length ({})", args[2])
                            },
                            (Expr::Literal(LiteralValue::Int(n)), _) if n > 0 => e.str().slice(lit(n - 1), length.clone()),
                            (Expr::Literal(LiteralValue::Int(n)), _) => {
                                e.str().slice(lit(0), (length.clone() + lit(n - 1)).clip_min(lit(0)))
                            },
                            (Expr::Literal(_), _) => polars_bail!(SQLSyntax: "invalid 'start' for SUBSTR ({})", args[1]),
                            (_, Expr::Literal(LiteralValue::Float(_))) => {
                                polars_bail!(SQLSyntax: "invalid 'length' for SUBSTR ({})", args[1])
                            },
                            _ => {
                                let adjusted_start = start.clone() - lit(1);
                                when(adjusted_start.clone().lt(lit(0)))
                                    .then(e.clone().str().slice(lit(0), (length.clone() + adjusted_start.clone()).clip_min(lit(0))))
                                    .otherwise(e.clone().str().slice(adjusted_start.clone(), length.clone()))
                            }
                        })
                    }),
                    _ => polars_bail!(SQLSyntax: "SUBSTR expects 2-3 arguments (found {})", args.len()),
                }
            },
            Upper => self.visit_unary(|e| e.str().to_uppercase()),

            // ----
            // Aggregate functions
            // ----
            Avg => self.visit_unary(Expr::mean),
            Count => self.visit_count(),
            First => self.visit_unary(Expr::first),
            Last => self.visit_unary(Expr::last),
            Max => self.visit_unary_with_opt_cumulative(Expr::max, Expr::cum_max),
            Median => self.visit_unary(Expr::median),
            Min => self.visit_unary_with_opt_cumulative(Expr::min, Expr::cum_min),
            StdDev => self.visit_unary(|e| e.std(1)),
            Sum => self.visit_unary_with_opt_cumulative(Expr::sum, Expr::cum_sum),
            Variance => self.visit_unary(|e| e.var(1)),

            // ----
            // Array functions
            // ----
            ArrayAgg => self.visit_arr_agg(),
            ArrayContains => self.visit_binary::<Expr>(|e, s| e.list().contains(s)),
            ArrayGet => {
                // note: SQL is 1-indexed, not 0-indexed
                self.visit_binary(|e, idx: Expr| {
                    let idx = adjust_one_indexed_param(idx, true);
                    e.list().get(idx, true)
                })
            },
            ArrayLength => self.visit_unary(|e| e.list().len()),
            ArrayMax => self.visit_unary(|e| e.list().max()),
            ArrayMean => self.visit_unary(|e| e.list().mean()),
            ArrayMin => self.visit_unary(|e| e.list().min()),
            ArrayReverse => self.visit_unary(|e| e.list().reverse()),
            ArraySum => self.visit_unary(|e| e.list().sum()),
            ArrayToString => self.visit_arr_to_string(),
            ArrayUnique => self.visit_unary(|e| e.list().unique()),
            Explode => self.visit_unary(|e| e.explode()),

            // ----
            // Column selection
            // ----
            Columns => {
                let active_schema = self.active_schema;
                self.try_visit_unary(|e: Expr| {
                    match e {
                        Expr::Literal(LiteralValue::String(pat)) => {
                            if "*" == pat {
                                polars_bail!(SQLSyntax: "COLUMNS('*') is not a valid regex; did you mean COLUMNS(*)?")
                            };
                            let pat = match pat.as_str() {
                                _ if pat.starts_with('^') && pat.ends_with('$') => pat.to_string(),
                                _ if pat.starts_with('^') => format!("{}.*$", pat),
                                _ if pat.ends_with('$') => format!("^.*{}", pat),
                                _ => format!("^.*{}.*$", pat),
                            };
                            if let Some(active_schema) = &active_schema {
                                let rx = regex::Regex::new(&pat).unwrap();
                                let col_names = active_schema
                                    .iter_names()
                                    .filter(|name| rx.is_match(name))
                                    .collect::<Vec<_>>();

                                Ok(if col_names.len() == 1 {
                                    col(col_names[0])
                                } else {
                                    cols(col_names)
                                })
                            } else {
                                Ok(col(&pat))
                            }
                        },
                        Expr::Wildcard => Ok(col("*")),
                        _ => polars_bail!(SQLSyntax: "COLUMNS expects a regex; found {:?}", e),
                    }
                })
            },

            // ----
            // User-defined
            // ----
            Udf(func_name) => self.visit_udf(&func_name),
        }
    }

    fn visit_udf(&mut self, func_name: &str) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?
            .into_iter()
            .map(|arg| {
                if let FunctionArgExpr::Expr(e) = arg {
                    parse_sql_expr(e, self.ctx, self.active_schema)
                } else {
                    polars_bail!(SQLInterface: "only expressions are supported in UDFs")
                }
            })
            .collect::<PolarsResult<Vec<_>>>()?;

        self.ctx
            .function_registry
            .get_udf(func_name)?
            .ok_or_else(|| polars_err!(SQLInterface: "UDF {} not found", func_name))?
            .call(args)
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
                    let expr = parse_sql_expr(&o.expr, self.ctx, self.active_schema)?;
                    Ok(match o.asc {
                        Some(b) => (expr, !b),
                        None => (expr, false),
                    })
                })
                .collect::<PolarsResult<Vec<_>>>()?
                .into_iter()
                .unzip();
            self.visit_unary_no_window(|e| {
                cumulative_f(
                    e.sort_by(
                        &order_by,
                        SortMultipleOptions::default().with_order_descending_multi(desc.clone()),
                    ),
                    false,
                )
            })
        } else {
            self.visit_unary(f)
        }
    }

    fn visit_unary(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        self.try_visit_unary(|e| Ok(f(e)))
    }

    fn try_visit_unary(&mut self, f: impl Fn(Expr) -> PolarsResult<Expr>) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                f(parse_sql_expr(sql_expr, self.ctx, self.active_schema)?)
            },
            [FunctionArgExpr::Wildcard] => f(parse_sql_expr(
                &SQLExpr::Wildcard,
                self.ctx,
                self.active_schema,
            )?),
            _ => self.not_supported_error(),
        }
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
                SQLInterface: "Named windows are not currently supported; found {:?}",
                named_window
            ),
            _ => self.visit_unary(f),
        }
    }

    fn visit_unary_no_window(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                // apply the function on the inner expr -- e.g. SUM(a) -> SUM
                Ok(f(expr))
            },
            _ => self.not_supported_error(),
        }
    }

    fn visit_binary<Arg: FromSQLExpr>(
        &mut self,
        f: impl Fn(Expr, Arg) -> Expr,
    ) -> PolarsResult<Expr> {
        self.try_visit_binary(|e, a| Ok(f(e, a)))
    }

    fn try_visit_binary<Arg: FromSQLExpr>(
        &mut self,
        f: impl Fn(Expr, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr1), FunctionArgExpr::Expr(sql_expr2)] => {
                let expr1 = parse_sql_expr(sql_expr1, self.ctx, self.active_schema)?;
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
        let args = extract_args(self.func)?;
        let mut expr_args = vec![];
        for arg in args {
            if let FunctionArgExpr::Expr(sql_expr) = arg {
                expr_args.push(parse_sql_expr(sql_expr, self.ctx, self.active_schema)?);
            } else {
                return self.not_supported_error();
            };
        }
        f(&expr_args)
    }

    fn try_visit_ternary<Arg: FromSQLExpr>(
        &mut self,
        f: impl Fn(Expr, Arg, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr1), FunctionArgExpr::Expr(sql_expr2), FunctionArgExpr::Expr(sql_expr3)] =>
            {
                let expr1 = parse_sql_expr(sql_expr1, self.ctx, self.active_schema)?;
                let expr2 = Arg::from_sql_expr(sql_expr2, self.ctx)?;
                let expr3 = Arg::from_sql_expr(sql_expr3, self.ctx)?;
                f(expr1, expr2, expr3)
            },
            _ => self.not_supported_error(),
        }
    }

    fn visit_nullary(&self, f: impl Fn() -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        if !args.is_empty() {
            return self.not_supported_error();
        }
        Ok(f())
    }

    fn visit_arr_agg(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct, clauses) = extract_args_and_clauses(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let mut base = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                if is_distinct {
                    base = base.unique_stable();
                }
                for clause in clauses {
                    match clause {
                        FunctionArgumentClause::OrderBy(order_exprs) => {
                            base = self.apply_order_by(base, order_exprs.as_slice())?;
                        },
                        FunctionArgumentClause::Limit(limit_expr) => {
                            let limit = parse_sql_expr(&limit_expr, self.ctx, self.active_schema)?;
                            match limit {
                                Expr::Literal(LiteralValue::Int(n)) if n >= 0 => {
                                    base = base.head(Some(n as usize))
                                },
                                _ => {
                                    polars_bail!(SQLSyntax: "LIMIT in ARRAY_AGG must be a positive integer")
                                },
                            };
                        },
                        _ => {},
                    }
                }
                Ok(base.implode())
            },
            _ => {
                polars_bail!(SQLSyntax: "ARRAY_AGG must have exactly one argument; found {}", args.len())
            },
        }
    }

    fn visit_arr_to_string(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.len() {
            2 => self.try_visit_binary(|e, sep| {
                Ok(e.cast(DataType::List(Box::from(DataType::String)))
                    .list()
                    .join(sep, true))
            }),
            #[cfg(feature = "list_eval")]
            3 => self.try_visit_ternary(|e, sep, null_value| match null_value {
                Expr::Literal(LiteralValue::String(v)) => Ok(if v.is_empty() {
                    e.cast(DataType::List(Box::from(DataType::String)))
                        .list()
                        .join(sep, true)
                } else {
                    e.cast(DataType::List(Box::from(DataType::String)))
                        .list()
                        .eval(col("").fill_null(lit(v)), false)
                        .list()
                        .join(sep, false)
                }),
                _ => {
                    polars_bail!(SQLSyntax: "invalid null value for ARRAY_TO_STRING ({})", args[2])
                },
            }),
            _ => {
                polars_bail!(SQLSyntax: "ARRAY_TO_STRING expects 2-3 arguments (found {})", args.len())
            },
        }
    }

    fn visit_count(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;
        match (is_distinct, args.as_slice()) {
            // count(*), count()
            (false, [FunctionArgExpr::Wildcard] | []) => Ok(len()),
            // count(column_name)
            (false, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                let expr = self.apply_window_spec(expr, &self.func.over)?;
                Ok(expr.count())
            },
            // count(distinct column_name)
            (true, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                let expr = self.apply_window_spec(expr, &self.func.over)?;
                Ok(expr.clone().n_unique().sub(expr.null_count().gt(lit(0))))
            },
            _ => self.not_supported_error(),
        }
    }

    fn apply_order_by(&mut self, expr: Expr, order_by: &[OrderByExpr]) -> PolarsResult<Expr> {
        let mut by = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        let mut nulls_last = Vec::with_capacity(order_by.len());

        for ob in order_by {
            // note: if not specified 'NULLS FIRST' is default for DESC, 'NULLS LAST' otherwise
            // https://www.postgresql.org/docs/current/queries-order.html
            let desc_order = !ob.asc.unwrap_or(true);
            by.push(parse_sql_expr(&ob.expr, self.ctx, self.active_schema)?);
            nulls_last.push(!ob.nulls_first.unwrap_or(desc_order));
            descending.push(desc_order);
        }
        Ok(expr.sort_by(
            by,
            SortMultipleOptions::default()
                .with_order_descending_multi(descending)
                .with_nulls_last_multi(nulls_last)
                .with_maintain_order(true),
        ))
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
                            let e = parse_sql_expr(&o.expr, self.ctx, self.active_schema)?;
                            Ok(o.asc.map_or(e.clone(), |b| {
                                e.sort(SortOptions::default().with_order_descending(!b))
                            }))
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;
                    expr.over(exprs)
                } else {
                    // Process for simple window specification, partition by first
                    let partition_by = window_spec
                        .partition_by
                        .iter()
                        .map(|p| parse_sql_expr(p, self.ctx, self.active_schema))
                        .collect::<PolarsResult<Vec<_>>>()?;
                    expr.over(partition_by)
                }
            },
            Some(WindowType::NamedWindow(named_window)) => polars_bail!(
                SQLInterface: "Named windows are not currently supported; found {:?}",
                named_window
            ),
            None => expr,
        })
    }

    fn not_supported_error(&self) -> PolarsResult<Expr> {
        polars_bail!(
            SQLInterface:
            "no function matches the given name and arguments: `{}`",
            self.func.to_string()
        );
    }
}

fn extract_args(func: &SQLFunction) -> PolarsResult<Vec<&FunctionArgExpr>> {
    let (args, _, _) = _extract_func_args(func, false, false)?;
    Ok(args)
}

fn extract_args_distinct(func: &SQLFunction) -> PolarsResult<(Vec<&FunctionArgExpr>, bool)> {
    let (args, is_distinct, _) = _extract_func_args(func, true, false)?;
    Ok((args, is_distinct))
}

fn extract_args_and_clauses(
    func: &SQLFunction,
) -> PolarsResult<(Vec<&FunctionArgExpr>, bool, Vec<FunctionArgumentClause>)> {
    _extract_func_args(func, true, true)
}

fn _extract_func_args(
    func: &SQLFunction,
    get_distinct: bool,
    get_clauses: bool,
) -> PolarsResult<(Vec<&FunctionArgExpr>, bool, Vec<FunctionArgumentClause>)> {
    match &func.args {
        FunctionArguments::List(FunctionArgumentList {
            args,
            duplicate_treatment,
            clauses,
        }) => {
            let is_distinct = matches!(duplicate_treatment, Some(DuplicateTreatment::Distinct));
            if !(get_clauses || get_distinct) && is_distinct {
                polars_bail!(SQLSyntax: "unexpected use of DISTINCT found in '{}'", func.name)
            } else if !get_clauses && !clauses.is_empty() {
                polars_bail!(SQLSyntax: "unexpected clause found in '{}' ({})", func.name, clauses[0])
            } else {
                let unpacked_args = args
                    .iter()
                    .map(|arg| match arg {
                        FunctionArg::Named { arg, .. } => arg,
                        FunctionArg::Unnamed(arg) => arg,
                    })
                    .collect();
                Ok((unpacked_args, is_distinct, clauses.clone()))
            }
        },
        FunctionArguments::Subquery { .. } => {
            Err(polars_err!(SQLInterface: "subquery not expected in {}", func.name))
        },
        FunctionArguments::None => Ok((vec![], false, vec![])),
    }
}

pub(crate) trait FromSQLExpr {
    fn from_sql_expr(expr: &SQLExpr, ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized;
}

impl FromSQLExpr for f64 {
    fn from_sql_expr(expr: &SQLExpr, _ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(v) => match v {
                SQLValue::Number(s, _) => s
                    .parse()
                    .map_err(|_| polars_err!(SQLInterface: "cannot parse literal {:?}", s)),
                _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", v),
            },
            _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", expr),
        }
    }
}

impl FromSQLExpr for bool {
    fn from_sql_expr(expr: &SQLExpr, _ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(v) => match v {
                SQLValue::Boolean(v) => Ok(*v),
                _ => polars_bail!(SQLInterface: "cannot parse boolean {:?}", v),
            },
            _ => polars_bail!(SQLInterface: "cannot parse boolean {:?}", expr),
        }
    }
}

impl FromSQLExpr for String {
    fn from_sql_expr(expr: &SQLExpr, _: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(v) => match v {
                SQLValue::SingleQuotedString(s) => Ok(s.clone()),
                _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", v),
            },
            _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", expr),
        }
    }
}

impl FromSQLExpr for StrptimeOptions {
    fn from_sql_expr(expr: &SQLExpr, _: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(v) => match v {
                SQLValue::SingleQuotedString(s) => Ok(StrptimeOptions {
                    format: Some(s.clone()),
                    ..StrptimeOptions::default()
                }),
                _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", v),
            },
            _ => polars_bail!(SQLInterface: "cannot parse literal {:?}", expr),
        }
    }
}

impl FromSQLExpr for Expr {
    fn from_sql_expr(expr: &SQLExpr, ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        parse_sql_expr(expr, ctx, None)
    }
}
