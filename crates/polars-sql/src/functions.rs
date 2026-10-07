use std::ops::Sub;

use polars_core::chunked_array::ops::{SortMultipleOptions, SortOptions};
use polars_core::prelude::{
    DataType, ExplodeOptions, PolarsResult, QuantileMethod, Scalar, Schema, TimeUnit, polars_bail,
    polars_err,
};
use polars_defs::expr::UnicodeForm;
#[cfg(feature = "rank")]
use polars_defs::expr::{RankMethod, RankOptions};
use polars_lazy::dsl::Expr;
#[cfg(feature = "approx_quantile")]
use polars_lazy::prelude::ApproxQuantileMethod;
use polars_plan::dsl::functions::{
    coalesce, col, cols, concat_str, element, int_range, len, lit, max_horizontal, min_horizontal,
    when,
};
use polars_plan::dsl::{FunctionExpr, SqlBinaryOp, SqlFunction};
use polars_plan::plans::{DynLiteralValue, LiteralValue, RowEncodingVariant, typed_lit};
use polars_plan::prelude::StrptimeOptions;
use polars_utils::pl_str::PlSmallStr;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    DateTimeField, DuplicateTreatment, Expr as SQLExpr, Function as SQLFunction, FunctionArg,
    FunctionArgExpr, FunctionArgumentClause, FunctionArgumentList, FunctionArguments, Ident,
    OrderByExpr, Value as SQLValue, ValueWithSpan, WindowFrame, WindowFrameBound, WindowFrameUnits,
    WindowSpec,
};
use sqlparser::tokenizer::Span;

use crate::SQLContext;
use crate::grouping_sets::MAX_GROUPING_ARGS;
use crate::sql_expr::{
    adjust_one_indexed_param, approximate_literal, decimal_literal, decimal_literal_to_f64,
    order_by_sort_options, parse_extract_date_part, parse_sql_array, parse_sql_expr, sql_binary,
};
use crate::sql_visitors::grouping_call_args;
use crate::window_frames::{MAX_ROW_OFFSET, is_frame_aggregate};

pub(crate) struct SQLFunctionVisitor<'a> {
    pub(crate) func: &'a SQLFunction,
    pub(crate) ctx: &'a mut SQLContext,
    pub(crate) active_schema: Option<&'a Schema>,
    pub(crate) filter: Option<Expr>,
    /// The `OVER` clause, with named windows resolved.
    pub(crate) window: Option<WindowSpec>,
    /// Whether the value arguments are read once per row, as by an aggregate without `OVER`
    /// (see `parse_sql_arg`).
    pub(crate) reads_rows: bool,
}

/// A parsed value argument. The constant argument of an aggregate without OVER is kept apart:
/// the aggregate follows from the constant and the number of rows read, while reading the
/// constant once per row would materialize it.
enum ValueArg {
    /// Read once per row, and subject to FILTER.
    Rows(Expr),
    Constant(Expr),
}

/// SQL functions that are supported by Polars
pub(crate) enum PolarsSQLFunctions {
    // ----
    // Bitwise functions
    // ----
    /// SQL 'bit_and' function.
    /// Returns the bitwise AND of the input expressions.
    /// ```sql
    /// SELECT BIT_AND(col1, col2) FROM df;
    /// ```
    BitAnd,
    /// SQL 'bit_count' function.
    /// Returns the number of set bits in the input expression.
    /// ```sql
    /// SELECT BIT_COUNT(col1) FROM df;
    /// ```
    #[cfg(feature = "bitwise")]
    BitCount,
    /// SQL 'bit_or' function.
    /// Returns the bitwise OR of the input expressions.
    /// ```sql
    /// SELECT BIT_OR(col1, col2) FROM df;
    /// ```
    BitNot,
    /// SQL 'bit_not' function.
    /// Returns the bitwise Not of the input expression.
    /// ```sql
    /// SELECT BIT_Not(col1) FROM df;
    /// ```
    BitOr,
    /// SQL 'bit_xor' function.
    /// Returns the bitwise XOR of the input expressions.
    /// ```sql
    /// SELECT BIT_XOR(col1, col2) FROM df;
    /// ```
    BitXor,

    // ----
    // Math functions
    // ----
    /// SQL 'abs' function.
    /// Returns the absolute value of the input expression.
    /// ```sql
    /// SELECT ABS(col1) FROM df;
    /// ```
    Abs,
    /// SQL 'ceil' function.
    /// Returns the nearest integer closest from zero.
    /// ```sql
    /// SELECT CEIL(col1) FROM df;
    /// ```
    Ceil,
    /// SQL 'div' function.
    /// Returns the integer quotient of the division.
    /// ```sql
    /// SELECT DIV(col1, 2) FROM df;
    /// ```
    Div,
    /// SQL 'erf' function.
    /// Computes the error function of the given value.
    /// ```sql
    /// SELECT ERF(col1) FROM df;
    /// ```
    Erf,
    /// SQL 'erfc' function.
    /// Computes the complementary error function of the given value.
    /// ```sql
    /// SELECT ERFC(col1) FROM df;
    /// ```
    Erfc,
    /// SQL 'exp' function.
    /// Computes the exponential of the given value.
    /// ```sql
    /// SELECT EXP(col1) FROM df;
    /// ```
    Exp,
    /// SQL 'floor' function.
    /// Returns the nearest integer away from zero.
    ///   0.5 will be rounded
    /// ```sql
    /// SELECT FLOOR(col1) FROM df;
    /// ```
    Floor,
    /// SQL 'pi' function.
    /// Returns a (very good) approximation of 𝜋.
    /// ```sql
    /// SELECT PI() FROM df;
    /// ```
    Pi,
    /// SQL 'ln' function.
    /// Computes the natural logarithm of the given value.
    /// ```sql
    /// SELECT LN(col1) FROM df;
    /// ```
    Ln,
    /// SQL 'log2' function.
    /// Computes the logarithm of the given value in base 2.
    /// ```sql
    /// SELECT LOG2(col1) FROM df;
    /// ```
    Log2,
    /// SQL 'log10' function.
    /// Computes the logarithm of the given value in base 10.
    /// ```sql
    /// SELECT LOG10(col1) FROM df;
    /// ```
    Log10,
    /// SQL 'log' function.
    /// Computes the `base` logarithm of the given value.
    /// ```sql
    /// SELECT LOG(col1, 10) FROM df;
    /// ```
    Log,
    /// SQL 'log1p' function.
    /// Computes the natural logarithm of "given value plus one".
    /// ```sql
    /// SELECT LOG1P(col1) FROM df;
    /// ```
    Log1p,
    /// SQL 'pow' function.
    /// Returns the value to the power of the given exponent.
    /// ```sql
    /// SELECT POW(col1, 2) FROM df;
    /// ```
    Pow,
    /// SQL 'mod' function.
    /// Returns the remainder of a numeric expression divided by another numeric expression.
    /// ```sql
    /// SELECT MOD(col1, 2) FROM df;
    /// ```
    Mod,
    /// SQL 'sqrt' function.
    /// Returns the square root (√) of a number.
    /// ```sql
    /// SELECT SQRT(col1) FROM df;
    /// ```
    Sqrt,
    /// SQL 'cbrt' function.
    /// Returns the cube root (∛) of a number.
    /// ```sql
    /// SELECT CBRT(col1) FROM df;
    /// ```
    Cbrt,
    /// SQL 'round' function.
    /// Round a number to `n` decimals (default: 0) away from zero.
    ///   .5 is rounded away from zero.
    /// ```sql
    /// SELECT ROUND(col1, 3) FROM df;
    /// ```
    Round,
    /// SQL 'truncate' function.
    /// Truncate a number toward zero to `n` decimals (default: 0).
    /// ```sql
    /// SELECT TRUNCATE(col1, 2) FROM df;
    /// ```
    Truncate,
    /// SQL 'sign' function.
    /// Returns the sign of the argument as -1, 0, or +1.
    /// ```sql
    /// SELECT SIGN(col1) FROM df;
    /// ```
    Sign,

    // ----
    // Trig functions
    // ----
    /// SQL 'cos' function.
    /// Compute the cosine sine of the input expression (in radians).
    /// ```sql
    /// SELECT COS(col1) FROM df;
    /// ```
    Cos,
    /// SQL 'cot' function.
    /// Compute the cotangent of the input expression (in radians).
    /// ```sql
    /// SELECT COT(col1) FROM df;
    /// ```
    Cot,
    /// SQL 'sin' function.
    /// Compute the sine of the input expression (in radians).
    /// ```sql
    /// SELECT SIN(col1) FROM df;
    /// ```
    Sin,
    /// SQL 'tan' function.
    /// Compute the tangent of the input expression (in radians).
    /// ```sql
    /// SELECT TAN(col1) FROM df;
    /// ```
    Tan,
    /// SQL 'cosd' function.
    /// Compute the cosine sine of the input expression (in degrees).
    /// ```sql
    /// SELECT COSD(col1) FROM df;
    /// ```
    CosD,
    /// SQL 'cotd' function.
    /// Compute cotangent of the input expression (in degrees).
    /// ```sql
    /// SELECT COTD(col1) FROM df;
    /// ```
    CotD,
    /// SQL 'sind' function.
    /// Compute the sine of the input expression (in degrees).
    /// ```sql
    /// SELECT SIND(col1) FROM df;
    /// ```
    SinD,
    /// SQL 'tand' function.
    /// Compute the tangent of the input expression (in degrees).
    /// ```sql
    /// SELECT TAND(col1) FROM df;
    /// ```
    TanD,
    /// SQL 'acos' function.
    /// Compute inverse cosine of the input expression (in radians).
    /// ```sql
    /// SELECT ACOS(col1) FROM df;
    /// ```
    Acos,
    /// SQL 'asin' function.
    /// Compute inverse sine of the input expression (in radians).
    /// ```sql
    /// SELECT ASIN(col1) FROM df;
    /// ```
    Asin,
    /// SQL 'atan' function.
    /// Compute inverse tangent of the input expression (in radians).
    /// ```sql
    /// SELECT ATAN(col1) FROM df;
    /// ```
    Atan,
    /// SQL 'atan2' function.
    /// Compute the inverse tangent of col1/col2 (in radians).
    /// ```sql
    /// SELECT ATAN2(col1, col2) FROM df;
    /// ```
    Atan2,
    /// SQL 'acosd' function.
    /// Compute inverse cosine of the input expression (in degrees).
    /// ```sql
    /// SELECT ACOSD(col1) FROM df;
    /// ```
    AcosD,
    /// SQL 'asind' function.
    /// Compute inverse sine of the input expression (in degrees).
    /// ```sql
    /// SELECT ASIND(col1) FROM df;
    /// ```
    AsinD,
    /// SQL 'atand' function.
    /// Compute inverse tangent of the input expression (in degrees).
    /// ```sql
    /// SELECT ATAND(col1) FROM df;
    /// ```
    AtanD,
    /// SQL 'atan2d' function.
    /// Compute the inverse tangent of col1/col2 (in degrees).
    /// ```sql
    /// SELECT ATAN2D(col1) FROM df;
    /// ```
    Atan2D,
    /// SQL 'degrees' function.
    /// Convert between radians and degrees.
    /// ```sql
    /// SELECT DEGREES(col1) FROM df;
    /// ```
    ///
    ///
    Degrees,
    /// SQL 'RADIANS' function.
    /// Convert between degrees and radians.
    /// ```sql
    /// SELECT RADIANS(col1) FROM df;
    /// ```
    Radians,

    // ----
    // Temporal functions
    // ----
    /// SQL 'date_part' function.
    /// Extracts a part of a date (or datetime) such as 'year', 'month', etc.
    /// ```sql
    /// SELECT DATE_PART('year', col1) FROM df;
    /// SELECT DATE_PART('day', col1) FROM df;
    DatePart,
    /// SQL date part accessor functions ('YEAR', 'MONTH', 'DAY', 'HOUR', etc).
    /// Shorthand for DATE_PART with a fixed part.
    /// ```sql
    /// SELECT YEAR(col1), MONTH(col1), DAYOFWEEK(col1) FROM df;
    /// ```
    DatePartOf(DateTimeField),
    /// SQL 'strftime' function.
    /// Converts a datetime to a string using a format string.
    /// ```sql
    /// SELECT STRFTIME(col1, '%d-%m-%Y %H:%M') FROM df;
    /// ```
    Strftime,

    // ----
    // String functions
    // ----
    /// SQL 'bit_length' function (bytes).
    /// ```sql
    /// SELECT BIT_LENGTH(col1) FROM df;
    /// ```
    BitLength,
    /// SQL 'concat' function.
    /// Returns all input expressions concatenated together as a string.
    /// ```sql
    /// SELECT CONCAT(col1, col2) FROM df;
    /// ```
    Concat,
    /// SQL 'concat_ws' function.
    /// Returns all input expressions concatenated together
    /// (and interleaved with a separator) as a string.
    /// ```sql
    /// SELECT CONCAT_WS(':', col1, col2, col3) FROM df;
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
    /// SQL 'ends_with' function.
    /// Returns True if the value ends with the second argument.
    /// ```sql
    /// SELECT ENDS_WITH(col1, 'a') FROM df;
    /// SELECT col2 from df WHERE ENDS_WITH(col1, 'a');
    /// ```
    EndsWith,
    /// SQL 'initcap' function.
    /// Returns the value with the first letter capitalized.
    /// ```sql
    /// SELECT INITCAP(col1) FROM df;
    /// ```
    #[cfg(feature = "nightly")]
    InitCap,
    /// SQL 'left' function.
    /// Returns the first (leftmost) `n` characters.
    /// ```sql
    /// SELECT LEFT(col1, 3) FROM df;
    /// ```
    Left,
    /// SQL 'lpad' function.
    /// Pads a string on the left to a specified length, using an optional fill character.
    /// ```sql
    /// SELECT LPAD(col1, 10, 'x') FROM df;
    /// ```
    LeftPad,
    /// SQL 'ltrim' function.
    /// Strip whitespaces from the left.
    /// ```sql
    /// SELECT LTRIM(col1) FROM df;
    /// ```
    LeftTrim,
    /// SQL 'length' function (characters.
    /// Returns the character length of the string.
    /// ```sql
    /// SELECT LENGTH(col1) FROM df;
    /// ```
    Length,
    /// SQL 'lower' function.
    /// Returns an lowercased column.
    /// ```sql
    /// SELECT LOWER(col1) FROM df;
    /// ```
    Lower,
    /// SQL 'normalize' function.
    /// Convert string to Unicode normalization form
    /// (one of NFC, NFKC, NFD, or NFKD - unquoted).
    /// ```sql
    /// SELECT NORMALIZE(col1, NFC) FROM df;
    /// ```
    Normalize,
    /// SQL 'octet_length' function.
    /// Returns the length of a given string in bytes.
    /// ```sql
    /// SELECT OCTET_LENGTH(col1) FROM df;
    /// ```
    OctetLength,
    /// SQL 'regexp_like' function.
    /// True if `pattern` matches the value (optional: `flags`).
    /// ```sql
    /// SELECT REGEXP_LIKE(col1, 'xyz', 'i') FROM df;
    /// ```
    RegexpLike,
    /// SQL 'replace' function.
    /// Replace a given substring with another string.
    /// ```sql
    /// SELECT REPLACE(col1, 'old', 'new') FROM df;
    /// ```
    Replace,
    /// SQL 'reverse' function.
    /// Return the reversed string.
    /// ```sql
    /// SELECT REVERSE(col1) FROM df;
    /// ```
    Reverse,
    /// SQL 'right' function.
    /// Returns the last (rightmost) `n` characters.
    /// ```sql
    /// SELECT RIGHT(col1, 3) FROM df;
    /// ```
    Right,
    /// SQL 'rpad' function.
    /// Pads a string on the right to a specified length, using an optional fill character.
    /// ```sql
    /// SELECT RPAD(col1, 10, 'x') FROM df;
    /// ```
    RightPad,
    /// SQL 'rtrim' function.
    /// Strip whitespaces from the right.
    /// ```sql
    /// SELECT RTRIM(col1) FROM df;
    /// ```
    RightTrim,
    /// SQL 'split_part' function.
    /// Splits a string into an array of strings using the given delimiter
    /// and returns the `n`-th part (1-indexed).
    /// ```sql
    /// SELECT SPLIT_PART(col1, ',', 2) FROM df;
    /// ```
    SplitPart,
    /// SQL 'starts_with' function.
    /// Returns True if the value starts with the second argument.
    /// ```sql
    /// SELECT STARTS_WITH(col1, 'a') FROM df;
    /// SELECT col2 from df WHERE STARTS_WITH(col1, 'a');
    /// ```
    StartsWith,
    /// SQL 'strpos' function.
    /// Returns the index of the given substring in the target string.
    /// ```sql
    /// SELECT STRPOS(col1,'xyz') FROM df;
    /// ```
    StrPos,
    /// SQL 'substr' function.
    /// Returns a portion of the data (first character = 1) in the range.
    ///   \[start, start + length]
    /// ```sql
    /// SELECT SUBSTR(col1, 3, 5) FROM df;
    /// ```
    Substring,
    /// SQL 'string_to_array' function.
    /// Splits a string into an array of strings using the given delimiter.
    /// ```sql
    /// SELECT STRING_TO_ARRAY(col1, ',') FROM df;
    /// ```
    StringToArray,
    /// SQL 'strptime' function.
    /// Converts a string to a datetime using a format string.
    /// ```sql
    /// SELECT STRPTIME(col1, '%d-%m-%Y %H:%M') FROM df;
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
    /// SQL 'timestamp' function.
    /// Converts a formatted string datetime to an actual Datetime type; ISO-8601 format is
    /// assumed unless a strftime-compatible formatting string is provided as the second
    /// parameter.
    /// ```sql
    /// SELECT TIMESTAMP('2021-03-15 10:30:45') FROM df;
    /// SELECT TIMESTAMP('2021-15-03T00:01:02.333', '%Y-d%-%m %H:%M:%S') FROM df;
    /// ```
    Timestamp,
    /// SQL 'upper' function.
    /// Returns an uppercased column.
    /// ```sql
    /// SELECT UPPER(col1) FROM df;
    /// ```
    Upper,

    // ----
    // Conditional functions
    // ----
    /// SQL 'coalesce' function.
    /// Returns the first non-null value in the provided values/columns.
    /// ```sql
    /// SELECT COALESCE(col1, ...) FROM df;
    /// ```
    Coalesce,
    /// SQL 'greatest' function.
    /// Returns the greatest value in the list of expressions.
    /// ```sql
    /// SELECT GREATEST(col1, col2, ...) FROM df;
    /// ```
    Greatest,
    /// SQL 'if' function.
    /// Returns expr1 if the boolean condition provided as the first
    /// parameter evaluates to true, and expr2 otherwise.
    /// ```sql
    /// SELECT IF(column < 0, expr1, expr2) FROM df;
    /// ```
    If,
    /// SQL 'ifnull' function.
    /// If an expression value is NULL, return an alternative value.
    /// ```sql
    /// SELECT IFNULL(string_col, 'n/a') FROM df;
    /// ```
    IfNull,
    /// SQL 'least' function.
    /// Returns the smallest value in the list of expressions.
    /// ```sql
    /// SELECT LEAST(col1, col2, ...) FROM df;
    /// ```
    Least,
    /// SQL 'nullif' function.
    /// Returns NULL if two expressions are equal, otherwise returns the first.
    /// ```sql
    /// SELECT NULLIF(col1, col2) FROM df;
    /// ```
    NullIf,

    // ----
    // Aggregate functions
    // ----
    /// SQL 'approx_quantile' function.
    /// Returns an approximation of the given quantile of the grouping, with an optional
    /// allowed rank error and sketch method.
    /// ```sql
    /// SELECT APPROX_QUANTILE(col1, 0.5) FROM df;
    /// SELECT APPROX_QUANTILE(col1, 0.5, 0.01) FROM df;
    /// SELECT APPROX_QUANTILE(col1, 0.5, 0.01, 'kll') FROM df;
    /// ```
    #[cfg(feature = "approx_quantile")]
    ApproxQuantile,
    /// SQL 'avg' function.
    /// Returns the average (mean) of all the elements in the grouping.
    /// ```sql
    /// SELECT AVG(col1) FROM df;
    /// ```
    Avg,
    /// SQL 'corr' function.
    /// Returns the Pearson correlation coefficient between two columns.
    /// ```sql
    /// SELECT CORR(col1, col2) FROM df;
    /// ```
    Corr,
    /// SQL 'count' function.
    /// Returns the amount of elements in the grouping.
    /// ```sql
    /// SELECT COUNT(col1) FROM df;
    /// SELECT COUNT(*) FROM df;
    /// SELECT COUNT(DISTINCT col1) FROM df;
    /// SELECT COUNT(DISTINCT *) FROM df;
    /// ```
    Count,
    /// SQL 'covar_pop' function.
    /// Returns the population covariance between two columns.
    /// ```sql
    /// SELECT COVAR_POP(col1, col2) FROM df;
    /// ```
    CovarPop,
    /// SQL 'covar_samp' function.
    /// Returns the sample covariance between two columns.
    /// ```sql
    /// SELECT COVAR_SAMP(col1, col2) FROM df;
    /// ```
    CovarSamp,
    /// SQL 'first' function.
    /// Returns the first element of the grouping.
    /// ```sql
    /// SELECT FIRST(col1) FROM df;
    /// ```
    First,
    /// SQL 'grouping' function.
    /// Returns, for each argument, whether the current row's grouping set omits
    /// that key, as bits with the last argument in the least significant position.
    /// ```sql
    /// SELECT col1, GROUPING(col1) FROM df GROUP BY ROLLUP(col1);
    /// ```
    Grouping,
    /// SQL 'grouping_id' function; an alias for `GROUPING`.
    GroupingId,
    /// SQL 'last' function.
    /// Returns the last element of the grouping.
    /// ```sql
    /// SELECT LAST(col1) FROM df;
    /// ```
    Last,
    /// SQL 'max' function.
    /// Returns the greatest (maximum) of all the elements in the grouping.
    /// ```sql
    /// SELECT MAX(col1) FROM df;
    /// ```
    Max,
    /// SQL 'median' function.
    /// Returns the median element from the grouping.
    /// ```sql
    /// SELECT MEDIAN(col1) FROM df;
    /// ```
    Median,
    /// SQL 'quantile_cont' function.
    /// Returns the continuous quantile element from the grouping
    /// (interpolated value between two closest values).
    /// ```sql
    /// SELECT QUANTILE_CONT(col1) FROM df;
    /// ```
    QuantileCont,
    /// SQL 'quantile_disc' function.
    /// Divides the [0, 1] interval into equal-length subintervals, each corresponding to a value,
    /// and returns the value associated with the subinterval where the quantile value falls.
    /// ```sql
    /// SELECT QUANTILE_DISC(col1) FROM df;
    /// ```
    QuantileDisc,
    /// SQL 'min' function.
    /// Returns the smallest (minimum) of all the elements in the grouping.
    /// ```sql
    /// SELECT MIN(col1) FROM df;
    /// ```
    Min,
    /// SQL 'stddev' function.
    /// Returns the standard deviation of all the elements in the grouping.
    /// ```sql
    /// SELECT STDDEV(col1) FROM df;
    /// ```
    StdDev,
    /// SQL 'string_agg' function (also known as `GROUP_CONCAT`).
    /// Concatenates the input string values into a single string,
    /// separated by the given delimiter (`,` if unspecified).
    /// ```sql
    /// SELECT STRING_AGG(col1) FROM df;
    /// SELECT STRING_AGG(col1, ',' ORDER BY col2 DESC) FROM df;
    /// SELECT STRING_AGG(DISTINCT col1, ',' ORDER BY col1) FROM df;
    /// ```
    StringAgg,
    /// SQL 'sum' function.
    /// Returns the sum of all the elements in the grouping.
    /// ```sql
    /// SELECT SUM(col1) FROM df;
    /// ```
    Sum,
    /// SQL 'total' function.
    /// Returns the sum of all the elements in the grouping; unlike `SUM`,
    /// empty or all-null input returns zero rather than `NULL`.
    /// ```sql
    /// SELECT TOTAL(col1) FROM df;
    /// ```
    Total,
    /// SQL 'variance' function.
    /// Returns the variance of all the elements in the grouping.
    /// ```sql
    /// SELECT VARIANCE(col1) FROM df;
    /// ```
    Variance,

    // ----
    // Array functions
    // ----
    /// SQL 'array_length' function.
    /// Returns the length of the array.
    /// ```sql
    /// SELECT ARRAY_LENGTH(col1) FROM df;
    /// ```
    ArrayLength,
    /// SQL 'array_lower' function.
    /// Returns the minimum value in an array; equivalent to `array_min`.
    /// ```sql
    /// SELECT ARRAY_LOWER(col1) FROM df;
    /// ```
    ArrayMin,
    /// SQL 'array_upper' function.
    /// Returns the maximum value in an array; equivalent to `array_max`.
    /// ```sql
    /// SELECT ARRAY_UPPER(col1) FROM df;
    /// ```
    ArrayMax,
    /// SQL 'array_sum' function.
    /// Returns the sum of all values in an array.
    /// ```sql
    /// SELECT ARRAY_SUM(col1) FROM df;
    /// ```
    ArraySum,
    /// SQL 'array_mean' function.
    /// Returns the mean of all values in an array.
    /// ```sql
    /// SELECT ARRAY_MEAN(col1) FROM df;
    /// ```
    ArrayMean,
    /// SQL 'array_reverse' function.
    /// Returns the array with the elements in reverse order.
    /// ```sql
    /// SELECT ARRAY_REVERSE(col1) FROM df;
    /// ```
    ArrayReverse,
    /// SQL 'array_unique' function.
    /// Returns the array with the unique elements.
    /// ```sql
    /// SELECT ARRAY_UNIQUE(col1) FROM df;
    /// ```
    ArrayUnique,
    /// SQL 'array_agg' function.
    /// Concatenates the input expressions, including nulls, into an array.
    /// ```sql
    /// SELECT ARRAY_AGG(col1, col2, ...) FROM df;
    /// ```
    ArrayAgg,
    /// SQL 'array_to_string' function.
    /// Takes all elements of the array and joins them into one string.
    /// ```sql
    /// SELECT ARRAY_TO_STRING(col1, ',') FROM df;
    /// SELECT ARRAY_TO_STRING(col1, ',', 'n/a') FROM df;
    /// ```
    ArrayToString,
    /// SQL 'array_get' function.
    /// Returns the value at the given index in the array.
    /// ```sql
    /// SELECT ARRAY_GET(col1, 1) FROM df;
    /// ```
    ArrayGet,
    /// SQL 'array_contains' function.
    /// Returns true if the array contains the value.
    /// ```sql
    /// SELECT ARRAY_CONTAINS(col1, 'foo') FROM df;
    /// ```
    ArrayContains,
    /// SQL 'array_inner_product' function (also known as `array_dot_product`).
    /// Returns the inner product of two fixed-size arrays.
    /// ```sql
    /// SELECT ARRAY_INNER_PRODUCT(col1, col2) FROM df;
    /// ```
    ArrayInnerProduct,
    /// SQL 'unnest' function.
    /// Unnest/explodes an array column into multiple rows.
    /// ```sql
    /// SELECT UNNEST(col1) FROM df;
    /// ```
    Explode,

    // ----
    // Window functions
    // ----
    /// SQL 'first_value' window function.
    /// Returns the first value in an ordered set of values (respecting window frame).
    /// ```sql
    /// SELECT FIRST_VALUE(col1) OVER (PARTITION BY category ORDER BY id) FROM df;
    /// ```
    FirstValue,
    /// SQL 'last_value' window function.
    /// Returns the last value in an ordered set of values (respecting window frame).
    /// With ORDER BY and no frame, the frame ends at the last row tied with the current row.
    /// ```sql
    /// SELECT LAST_VALUE(col1) OVER (PARTITION BY category ORDER BY id) FROM df;
    /// ```
    LastValue,
    /// SQL 'nth_value' window function.
    /// Returns the value at row `n` of the window frame (from 1), or NULL if the frame has fewer
    /// rows. `n` must be a positive integer literal.
    /// ```sql
    /// SELECT NTH_VALUE(col1, 2) OVER (PARTITION BY category ORDER BY id) FROM df;
    /// ```
    NthValue,
    /// SQL 'lag' function.
    /// Returns the value of the expression evaluated at the row n rows before the current row.
    /// ```sql
    /// SELECT lag(column_1, 1) OVER (PARTITION BY column_2 ORDER BY column_3) FROM df;
    /// ```
    Lag,
    /// SQL 'lead' function.
    /// Returns the value of the expression evaluated at the row n rows after the current row.
    /// ```sql
    /// SELECT lead(column_1, 1) OVER (PARTITION BY column_2 ORDER BY column_3) FROM df;
    /// ```
    Lead,
    /// SQL 'row_number' function.
    /// Returns the sequential row number within a window partition, starting from 1.
    /// ```sql
    /// SELECT ROW_NUMBER() OVER (ORDER BY col1) FROM df;
    /// SELECT ROW_NUMBER() OVER (PARTITION BY col1 ORDER BY col2) FROM df;
    /// ```
    RowNumber,
    /// SQL 'rank' function.
    /// Returns the rank of each row within a window partition, with gaps for ties.
    /// Rows with equal values receive the same rank, and the next rank skips numbers.
    /// ```sql
    /// SELECT RANK() OVER (ORDER BY col1) FROM df;
    /// SELECT RANK() OVER (PARTITION BY col1 ORDER BY col2 DESC) FROM df;
    /// ```
    Rank,
    /// SQL 'dense_rank' function.
    /// Returns the rank of each row within a window partition, without gaps for ties.
    /// Rows with equal values receive the same rank, and the next rank is consecutive.
    /// ```sql
    /// SELECT DENSE_RANK() OVER (ORDER BY col1) FROM df;
    /// SELECT DENSE_RANK() OVER (PARTITION BY col1 ORDER BY col2 DESC) FROM df;
    /// ```
    DenseRank,
    /// SQL 'percent_rank' function.
    /// Returns `(rank - 1) / (rows in partition - 1)`, or 0 for a partition of one row.
    /// ```sql
    /// SELECT PERCENT_RANK() OVER (ORDER BY col1) FROM df;
    /// ```
    PercentRank,
    /// SQL 'cume_dist' function.
    /// Returns the fraction of rows in the partition that come before the current row or are
    /// equal to it.
    /// ```sql
    /// SELECT CUME_DIST() OVER (ORDER BY col1) FROM df;
    /// ```
    CumeDist,
    /// SQL 'ntile' function.
    /// Splits the partition into `n` buckets of near-equal size and returns the bucket number,
    /// starting from 1. The first buckets get one more row when the rows don't split evenly.
    /// `n` must be a positive integer literal.
    /// ```sql
    /// SELECT NTILE(4) OVER (ORDER BY col1) FROM df;
    /// ```
    Ntile,

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
            "approx_quantile",
            "array_contains",
            "array_dot_product",
            "array_get",
            "array_inner_product",
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
            "bit_and",
            "bit_count",
            "bit_length",
            "bit_or",
            "bit_xor",
            "cbrt",
            "ceil",
            "ceiling",
            "char_length",
            "character_length",
            "coalesce",
            "columns",
            "concat",
            "concat_ws",
            "corr",
            "cos",
            "cosd",
            "cot",
            "cotd",
            "count",
            "covar",
            "covar_pop",
            "covar_samp",
            "cume_dist",
            "date",
            "date_part",
            "day",
            "dayofmonth",
            "dayofweek",
            "dayofyear",
            "degrees",
            "dense_rank",
            "ends_with",
            "erf",
            "erfc",
            "exp",
            "first",
            "first_value",
            "floor",
            "greatest",
            "hour",
            "if",
            "ifnull",
            "initcap",
            "lag",
            "last",
            "last_value",
            "lead",
            "least",
            "left",
            "length",
            "ln",
            "log",
            "log10",
            "log1p",
            "log2",
            "lower",
            "lpad",
            "ltrim",
            "max",
            "median",
            "min",
            "minute",
            "mod",
            "month",
            "nth_value",
            "ntile",
            "nullif",
            "octet_length",
            "percent_rank",
            "pi",
            "pow",
            "power",
            "quantile_cont",
            "quantile_disc",
            "quarter",
            "radians",
            "rank",
            "regexp_like",
            "replace",
            "reverse",
            "right",
            "round",
            "row_number",
            "rpad",
            "rtrim",
            "second",
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
            "total",
            "unnest",
            "upper",
            "var",
            "var_samp",
            "variance",
            "week",
            "year",
        ]
    }
}

impl PolarsSQLFunctions {
    fn try_from_sql(function: &'_ SQLFunction, ctx: &'_ SQLContext) -> PolarsResult<Self> {
        let function_name = function.name.0[0].as_ident().unwrap().value.to_lowercase();
        Ok(match function_name.as_str() {
            // ----
            // Bitwise functions
            // ----
            "bit_and" | "bitand" => Self::BitAnd,
            #[cfg(feature = "bitwise")]
            "bit_count" | "bitcount" => Self::BitCount,
            "bit_not" | "bitnot" => Self::BitNot,
            "bit_or" | "bitor" => Self::BitOr,
            "bit_xor" | "bitxor" | "xor" => Self::BitXor,

            // ----
            // Math functions
            // ----
            "abs" => Self::Abs,
            "cbrt" => Self::Cbrt,
            "ceil" | "ceiling" => Self::Ceil,
            "div" => Self::Div,
            "erf" => Self::Erf,
            "erfc" => Self::Erfc,
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
            "trunc" | "truncate" => Self::Truncate,
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
            // Temporal functions
            // ----
            "date" => Self::Date,
            "date_part" => Self::DatePart,
            "year" => Self::DatePartOf(DateTimeField::Year),
            "quarter" => Self::DatePartOf(DateTimeField::Quarter),
            "month" => Self::DatePartOf(DateTimeField::Month),
            "week" => Self::DatePartOf(DateTimeField::IsoWeek),
            "day" | "dayofmonth" => Self::DatePartOf(DateTimeField::Day),
            "dayofweek" => Self::DatePartOf(DateTimeField::DayOfWeek),
            "dayofyear" => Self::DatePartOf(DateTimeField::DayOfYear),
            "hour" => Self::DatePartOf(DateTimeField::Hour),
            "minute" => Self::DatePartOf(DateTimeField::Minute),
            "second" => Self::DatePartOf(DateTimeField::Second),
            "strftime" => Self::Strftime,
            "timestamp" | "datetime" => Self::Timestamp,

            // ----
            // String functions
            // ----
            "bit_length" => Self::BitLength,
            "concat" => Self::Concat,
            "concat_ws" => Self::ConcatWS,
            "ends_with" => Self::EndsWith,
            #[cfg(feature = "nightly")]
            "initcap" => Self::InitCap,
            "left" => Self::Left,
            "length" | "char_length" | "character_length" => Self::Length,
            "lower" => Self::Lower,
            "lpad" => Self::LeftPad,
            "ltrim" => Self::LeftTrim,
            "normalize" => Self::Normalize,
            "octet_length" => Self::OctetLength,
            "regexp_like" => Self::RegexpLike,
            "replace" => Self::Replace,
            "reverse" => Self::Reverse,
            "right" => Self::Right,
            "rpad" => Self::RightPad,
            "rtrim" => Self::RightTrim,
            "split_part" => Self::SplitPart,
            "starts_with" => Self::StartsWith,
            "string_to_array" => Self::StringToArray,
            "strpos" => Self::StrPos,
            "strptime" => Self::Strptime,
            "substr" => Self::Substring,
            "time" => Self::Time,
            "upper" => Self::Upper,

            // ----
            // Aggregate functions
            // ----
            #[cfg(feature = "approx_quantile")]
            "approx_quantile" => Self::ApproxQuantile,
            "avg" => Self::Avg,
            "corr" => Self::Corr,
            "count" => Self::Count,
            "covar_pop" => Self::CovarPop,
            "covar_samp" | "covar" => Self::CovarSamp,
            "first" => Self::First,
            "grouping" => Self::Grouping,
            "grouping_id" => Self::GroupingId,
            "last" => Self::Last,
            "max" => Self::Max,
            "median" => Self::Median,
            "min" => Self::Min,
            "quantile_cont" => Self::QuantileCont,
            "quantile_disc" => Self::QuantileDisc,
            "stdev" | "stddev" | "stdev_samp" | "stddev_samp" => Self::StdDev,
            "string_agg" | "listagg" | "group_concat" => Self::StringAgg,
            "sum" => Self::Sum,
            "total" => Self::Total,
            "var" | "variance" | "var_samp" => Self::Variance,

            // ----
            // Array functions
            // ----
            "array_agg" => Self::ArrayAgg,
            "array_contains" => Self::ArrayContains,
            "array_dot_product" | "array_inner_product" => Self::ArrayInnerProduct,
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
            // Window functions
            // ----
            "cume_dist" => Self::CumeDist,
            "dense_rank" => Self::DenseRank,
            "first_value" => Self::FirstValue,
            "last_value" => Self::LastValue,
            "lag" => Self::Lag,
            "lead" => Self::Lead,
            "nth_value" => Self::NthValue,
            "ntile" => Self::Ntile,
            "percent_rank" => Self::PercentRank,
            "rank" => Self::Rank,
            "row_number" => Self::RowNumber,

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

    /// Whether `call`, the parsed call of `function` without OVER, aggregates the rows of a
    /// group: a SQL aggregate, or a user-defined function that returns one value.
    pub(crate) fn is_aggregate_call(
        function: &SQLFunction,
        ctx: &SQLContext,
        call: &Expr,
    ) -> PolarsResult<bool> {
        Ok(match Self::try_from_sql(function, ctx)? {
            Self::Udf(_) => {
                matches!(call, Expr::AnonymousFunction { options, .. } if options.returns_scalar())
            },
            function => function.is_builtin_aggregate(),
        })
    }

    /// Whether this is a SQL aggregate function when called without OVER.
    fn is_builtin_aggregate(&self) -> bool {
        use PolarsSQLFunctions::*;
        match self {
            #[cfg(feature = "approx_quantile")]
            ApproxQuantile => true,
            ArrayAgg | Avg | Corr | Count | CovarPop | CovarSamp | First | Last | Max | Median
            | Min | QuantileCont | QuantileDisc | StdDev | StringAgg | Sum | Total | Variance => {
                true
            },
            // Without OVER, FIRST_VALUE is FIRST.
            FirstValue => true,
            _ => false,
        }
    }
}

impl SQLFunctionVisitor<'_> {
    pub(crate) fn visit_function(&mut self) -> PolarsResult<Expr> {
        use PolarsSQLFunctions::*;
        use polars_lazy::prelude::Literal;

        let function_name = PolarsSQLFunctions::try_from_sql(self.func, self.ctx)?;
        let function = self.func;

        // TODO: implement the following modifiers where possible
        if !function.within_group.is_empty() {
            polars_bail!(SQLInterface: "'WITHIN GROUP' is not currently supported")
        }
        if function.null_treatment.is_some() {
            polars_bail!(SQLInterface: "'IGNORE|RESPECT NULLS' is not currently supported")
        }
        if let Some(filter_expr) = &function.filter {
            self.filter = Some(parse_sql_expr(filter_expr, self.ctx, self.active_schema)?);
        }
        self.reads_rows = self.window.is_none() && function_name.is_builtin_aggregate();
        self.check_window_shape(&function_name)?;
        if self.window.is_some() && is_frame_aggregate(&function_name, self.is_distinct()) {
            return self.visit_window_aggregate(&function_name);
        }
        if self.window.is_some() && matches!(function_name, FirstValue | LastValue | NthValue) {
            return self.visit_window_value(&function_name);
        }

        let log_with_base =
            |e: Expr, base: f64| e.log(LiteralValue::Dyn(DynLiteralValue::Float(base)).lit());

        match function_name {
            // ----
            // Bitwise functions
            // ----
            BitAnd => self.visit_binary::<Expr>(Expr::and),
            #[cfg(feature = "bitwise")]
            BitCount => self.visit_unary(Expr::bitwise_count_ones),
            BitNot => self.visit_unary(Expr::not),
            BitOr => self.visit_binary::<Expr>(Expr::or),
            BitXor => self.visit_binary::<Expr>(Expr::xor),

            // ----
            // Math functions
            // ----
            Abs => self.visit_unary(Expr::abs),
            Cbrt => self.visit_unary(Expr::cbrt),
            Ceil => self.visit_unary(Expr::ceil),
            Div => self.visit_binary(|e, d| sql_binary(e, SqlBinaryOp::IntDiv, d)),
            Erf => self.visit_unary(Expr::erf),
            Erfc => self.visit_unary(Expr::erfc),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Ln => self.visit_unary(|e| log_with_base(e, std::f64::consts::E)),
            Log => self.visit_binary(Expr::log),
            Log10 => self.visit_unary(|e| log_with_base(e, 10.0)),
            Log1p => self.visit_unary(Expr::log1p),
            Log2 => self.visit_unary(|e| log_with_base(e, 2.0)),
            Pi => self.visit_nullary(Expr::pi),
            Mod => self.visit_binary(|e1, e2| sql_binary(e1, SqlBinaryOp::Rem, e2)),
            Pow => self.visit_binary(|e: Expr, p: Expr| sql_to_float(e).pow(sql_to_float(p))),
            Round => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| sql_round(e, 0)),
                    2 => self.try_visit_binary(|e, decimals| {
                        Ok(sql_round(e, match decimals {
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                                if n >= 0 { n as u32 } else {
                                    polars_bail!(SQLInterface: "ROUND does not support negative decimals value ({})", args[1])
                                }
                            },
                            _ => polars_bail!(SQLSyntax: "invalid value for ROUND decimals ({})", args[1]),
                        }))
                    }),
                    _ => polars_bail!(SQLSyntax: "ROUND expects 1-2 arguments (found {})", args.len()),
                }
            },
            Truncate => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.truncate(0)),
                    2 => self.try_visit_binary(|e, decimals| {
                        Ok(e.truncate(match decimals {
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                                if n >= 0 { n as u32 } else {
                                    polars_bail!(SQLInterface: "TRUNCATE does not support negative decimals value ({})", args[1])
                                }
                            },
                            _ => polars_bail!(SQLSyntax: "invalid value for TRUNCATE decimals ({})", args[1]),
                        }))
                    }),
                    _ => polars_bail!(SQLSyntax: "TRUNCATE expects 1-2 arguments (found {})", args.len()),
                }
            },
            Sign => self.visit_unary(Expr::sign),
            Sqrt => self.visit_unary(Expr::sqrt),

            // ----
            // Trig functions
            // ----
            Acos => self.visit_unary(|e| sql_to_float(e).arccos()),
            AcosD => self.visit_unary(|e| sql_to_float(e).arccos().degrees()),
            Asin => self.visit_unary(|e| sql_to_float(e).arcsin()),
            AsinD => self.visit_unary(|e| sql_to_float(e).arcsin().degrees()),
            Atan => self.visit_unary(|e| sql_to_float(e).arctan()),
            Atan2 => self.visit_binary(|e: Expr, s: Expr| sql_to_float(e).arctan2(sql_to_float(s))),
            Atan2D => self.visit_binary(|e: Expr, s: Expr| {
                sql_to_float(e).arctan2(sql_to_float(s)).degrees()
            }),
            AtanD => self.visit_unary(|e| sql_to_float(e).arctan().degrees()),
            Cos => self.visit_unary(|e| sql_to_float(e).cos()),
            CosD => self.visit_unary(|e| sql_to_float(e).radians().cos()),
            Cot => self.visit_unary(|e| sql_to_float(e).cot()),
            CotD => self.visit_unary(|e| sql_to_float(e).radians().cot()),
            Degrees => self.visit_unary(|e| sql_to_float(e).degrees()),
            Radians => self.visit_unary(|e| sql_to_float(e).radians()),
            Sin => self.visit_unary(|e| sql_to_float(e).sin()),
            SinD => self.visit_unary(|e| sql_to_float(e).radians().sin()),
            Tan => self.visit_unary(|e| sql_to_float(e).tan()),
            TanD => self.visit_unary(|e| sql_to_float(e).radians().tan()),

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
                            .then(lit(LiteralValue::untyped_null()))
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
                    Expr::Literal(p) if p.extract_str().is_some() => {
                        let p = p.extract_str().unwrap();
                        // note: 'DATE_PART' and 'EXTRACT' are minor syntactic
                        // variations on otherwise identical functionality
                        parse_extract_date_part(
                            e,
                            &DateTimeField::Custom(Ident {
                                value: p.to_string(),
                                quote_style: None,
                                span: Span::empty(),
                            }),
                        )
                    },
                    _ => {
                        polars_bail!(SQLSyntax: "invalid 'part' for EXTRACT/DATE_PART ({})", part);
                    },
                }
            }),
            DatePartOf(field) => self.try_visit_unary(|e| parse_extract_date_part(e, &field)),
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
                            Expr::Literal(lv) if lv.extract_str().is_some() => Ok(concat_str(&exprs[1..], lv.extract_str().unwrap(), true)),
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
                    Expr::Literal(lv) if lv.is_null() => lit(lv),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0))) => lit(""),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
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
                            (e.str().len_chars() + length.clone()).clip_min(lit(0)),
                        )),
                })
            }),
            LeftPad | RightPad => {
                let is_lpad = matches!(function_name, LeftPad);
                let fname = if is_lpad { "LPAD" } else { "RPAD" };
                let args = extract_args(function)?;
                let pad = |e: Expr, length: Expr, fill_char: char| {
                    let padded = if is_lpad {
                        e.str().pad_start(length.clone(), fill_char)
                    } else {
                        e.str().pad_end(length.clone(), fill_char)
                    };
                    Ok(padded.str().slice(lit(0), length))
                };
                match args.len() {
                    2 => self.try_visit_binary(|e, length| pad(e, length, ' ')),
                    3 => self.try_visit_ternary(|e: Expr, length: Expr, fill: Expr| match fill {
                        Expr::Literal(lv) if lv.extract_str().is_some() => {
                            let s = lv.extract_str().unwrap();
                            let mut chars = s.chars();
                            match (chars.next(), chars.next()) {
                                (Some(c), None) => pad(e, length, c),
                                _ => polars_bail!(SQLSyntax: "{} fill value must be a single character (found '{}')", fname, s),
                            }
                        },
                        _ => polars_bail!(SQLSyntax: "{} fill value must be a string literal", fname),
                    }),
                    _ => polars_bail!(SQLSyntax: "{} expects 2-3 arguments (found {})", fname, args.len()),
                }
            },
            LeftTrim | RightTrim => {
                let is_ltrim = matches!(function_name, LeftTrim);
                let fname = if is_ltrim { "LTRIM" } else { "RTRIM" };
                let strip: fn(Expr, Expr) -> Expr = if is_ltrim {
                    |e, s| e.str().strip_chars_start(s)
                } else {
                    |e, s| e.str().strip_chars_end(s)
                };
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| strip(e, lit(LiteralValue::untyped_null()))),
                    2 => self.visit_binary(strip),
                    _ => {
                        polars_bail!(SQLSyntax: "{} expects 1-2 arguments (found {})", fname, args.len())
                    },
                }
            },
            Length => self.visit_unary(|e| e.str().len_chars()),
            Lower => self.visit_unary(|e| e.str().to_lowercase()),
            Normalize => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.str().normalize(UnicodeForm::NFC)),
                    2 => {
                        let form = if let FunctionArgExpr::Expr(SQLExpr::Identifier(Ident {
                            value: s,
                            quote_style: None,
                            span: _,
                        })) = args[1]
                        {
                            match s.to_uppercase().as_str() {
                                "NFC" => UnicodeForm::NFC,
                                "NFD" => UnicodeForm::NFD,
                                "NFKC" => UnicodeForm::NFKC,
                                "NFKD" => UnicodeForm::NFKD,
                                _ => {
                                    polars_bail!(SQLSyntax: "invalid 'form' for NORMALIZE (found {})", s)
                                },
                            }
                        } else {
                            polars_bail!(SQLSyntax: "invalid 'form' for NORMALIZE (found {})", args[1])
                        };
                        self.try_visit_binary(|e, _form: Expr| Ok(e.str().normalize(form.clone())))
                    },
                    _ => {
                        polars_bail!(SQLSyntax: "NORMALIZE expects 1-2 arguments (found {})", args.len())
                    },
                }
            },
            OctetLength => self.visit_unary(|e| e.str().len_bytes()),
            StrPos => {
                // note: SQL is 1-indexed; returns zero if no match found
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
                                (Expr::Literal(s_lv), Expr::Literal(f_lv)) if s_lv.extract_str().is_some() && f_lv.extract_str().is_some() => {
                                    let s = s_lv.extract_str().unwrap();
                                    let f = f_lv.extract_str().unwrap();
                                    if f.is_empty() {
                                        polars_bail!(SQLSyntax: "invalid/empty 'flags' for REGEXP_LIKE ({})", args[2]);
                                    };
                                    lit(format!("(?{f}){s}"))
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
                    Expr::Literal(lv) if lv.is_null() => lit(lv),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0))) => typed_lit(""),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                        let n: i64 = n.try_into().unwrap();
                        let offset = if n < 0 {
                            lit(n.abs())
                        } else {
                            e.clone().str().len_chars().cast(DataType::Int32) - lit(n)
                        };
                        e.str().slice(offset, lit(LiteralValue::untyped_null()))
                    },
                    Expr::Literal(v) => {
                        polars_bail!(SQLSyntax: "invalid 'n_chars' for RIGHT ({:?})", v)
                    },
                    _ => when(length.clone().lt(lit(0)))
                        .then(
                            e.clone()
                                .str()
                                .slice(length.clone().abs(), lit(LiteralValue::untyped_null())),
                        )
                        .otherwise(e.clone().str().slice(
                            e.str().len_chars().cast(DataType::Int32) - length.clone(),
                            lit(LiteralValue::untyped_null()),
                        )),
                })
            }),
            SplitPart => {
                let args = extract_args(function)?;
                match args.len() {
                    3 => self.try_visit_ternary(|e, sep, idx| {
                        let idx = adjust_one_indexed_param(idx, true);
                        Ok(when(e.clone().is_not_null())
                            .then(
                                e.clone()
                                    .str()
                                    .split(sep)
                                    .list()
                                    .get(idx, true)
                                    .fill_null(lit("")),
                            )
                            .otherwise(e))
                    }),
                    _ => {
                        polars_bail!(SQLSyntax: "SPLIT_PART expects 3 arguments (found {})", args.len())
                    },
                }
            },
            StartsWith => self.visit_binary(|e, s| e.str().starts_with(s)),
            StringToArray => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, sep| e.str().split(sep)),
                    _ => {
                        polars_bail!(SQLSyntax: "STRING_TO_ARRAY expects 2 arguments (found {})", args.len())
                    },
                }
            },
            Strptime => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, fmt: String| {
                        e.str().strptime(
                            DataType::Datetime(TimeUnit::Microseconds, None),
                            StrptimeOptions {
                                format: Some(fmt.into()),
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
                            Expr::Literal(lv) if lv.is_null() => lit(lv),
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n <= 0 => e,
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => e.str().slice(lit(n - 1), lit(LiteralValue::untyped_null())),
                            Expr::Literal(_) => polars_bail!(SQLSyntax: "invalid 'start' for SUBSTR ({})", args[1]),
                            _ => start.clone() + lit(1),
                        })
                    }),
                    3 => self.try_visit_ternary(|e: Expr, start: Expr, length: Expr| {
                        let length = approximate_literal(length);
                        Ok(match (start.clone(), length.clone()) {
                            (Expr::Literal(lv), _) | (_, Expr::Literal(lv)) if lv.is_null() => lit(lv),
                            (_, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n)))) if n < 0 => {
                                polars_bail!(SQLSyntax: "SUBSTR does not support negative length ({})", args[2])
                            },
                            (Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))), _) if n > 0 => e.str().slice(lit(n - 1), length),
                            (Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))), _) => {
                                e.str().slice(lit(0), (length + lit(n - 1)).clip_min(lit(0)))
                            },
                            (Expr::Literal(_), _) => polars_bail!(SQLSyntax: "invalid 'start' for SUBSTR ({})", args[1]),
                            (_, Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(_)))) => {
                                polars_bail!(SQLSyntax: "invalid 'length' for SUBSTR ({})", args[1])
                            },
                            _ => {
                                let adjusted_start = start - lit(1);
                                when(adjusted_start.clone().lt(lit(0)))
                                    .then(e.clone().str().slice(lit(0), (length.clone() + adjusted_start.clone()).clip_min(lit(0))))
                                    .otherwise(e.str().slice(adjusted_start, length))
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
            #[cfg(feature = "approx_quantile")]
            ApproxQuantile => self.visit_approx_quantile(),
            Avg => self.visit_avg(),
            Corr => self.visit_binary(sql_corr),
            Count => self.visit_count(),
            CovarPop => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 0)),
            CovarSamp => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 1)),
            First => self.visit_unary_aggregate(Expr::first),
            Grouping | GroupingId => self.visit_grouping(),
            Last => self.visit_unary_aggregate(Expr::last),
            Max => self.visit_min_max(Expr::max),
            Median => self.visit_unary_aggregate(Expr::median),
            QuantileCont | QuantileDisc => {
                let (fname, method) = if matches!(function_name, QuantileCont) {
                    ("QUANTILE_CONT", QuantileMethod::Linear)
                } else {
                    ("QUANTILE_DISC", QuantileMethod::Equiprobable)
                };
                let args = extract_args(function)?;
                match args.as_slice() {
                    [
                        FunctionArgExpr::Expr(value),
                        FunctionArgExpr::Expr(quantile),
                    ] => {
                        // Parameters are not subject to an active FILTER clause; only the values are.
                        let quantile = parse_sql_expr(quantile, self.ctx, self.active_schema)?;
                        let quantile = parse_quantile_literal(quantile, fname, args[1])?;
                        let e =
                            self.aggregate_value(value, |e| e.quantile(quantile.clone(), method))?;
                        self.apply_window_spec(e)
                    },
                    _ => {
                        polars_bail!(SQLSyntax: "{} expects 2 arguments (found {})", fname, args.len())
                    },
                }
            },
            Min => self.visit_min_max(Expr::min),
            StdDev => self.visit_spread_aggregate(|e| e.std(1)),
            StringAgg => self.visit_string_agg(),
            Sum => self.visit_sum(),
            Total => self.visit_total(),
            Variance => self.visit_spread_aggregate(|e| e.var(1)),

            // ----
            // Array functions
            // ----
            ArrayAgg => self.visit_arr_agg(),
            ArrayContains => self.visit_binary::<Expr>(|e, s| e.list().contains(s, true)),
            ArrayInnerProduct => self.visit_array_inner_product(),
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
            ArrayReverse => self.visit_unary(|e| e.list().eval(element().reverse())),
            ArraySum => self.visit_unary(|e| e.list().sum()),
            ArrayToString => self.visit_arr_to_string(),
            ArrayUnique => self.visit_unary(|e| e.list().eval(element().unique_stable())),
            Explode => self.visit_unary(|e| {
                e.explode(ExplodeOptions {
                    empty_as_null: true,
                    keep_nulls: true,
                })
            }),

            // ----
            // Column selection
            // ----
            Columns => {
                let active_schema = self.active_schema;
                self.try_visit_unary(|e: Expr| match e {
                    Expr::Literal(lv) if lv.extract_str().is_some() => {
                        let pat = lv.extract_str().unwrap();
                        if pat == "*" {
                            polars_bail!(
                                SQLSyntax: "COLUMNS('*') is not a valid regex; \
                                did you mean COLUMNS(*)?"
                            )
                        };
                        let pat = match pat {
                            _ if pat.starts_with('^') && pat.ends_with('$') => pat.to_string(),
                            _ if pat.starts_with('^') => format!("{pat}.*$"),
                            _ if pat.ends_with('$') => format!("^.*{pat}"),
                            _ => format!("^.*{pat}.*$"),
                        };
                        if let Some(active_schema) = &active_schema {
                            let rx = polars_utils::regex_cache::compile_regex(&pat).unwrap();
                            let col_names = active_schema
                                .iter_names()
                                .filter(|name| rx.is_match(name))
                                .cloned()
                                .collect::<Vec<_>>();

                            Ok(if col_names.len() == 1 {
                                col(col_names.into_iter().next().unwrap())
                            } else {
                                cols(col_names).as_expr()
                            })
                        } else {
                            Ok(col(pat.as_str()))
                        }
                    },
                    Expr::Selector(s) => Ok(s.as_expr()),
                    _ => polars_bail!(SQLSyntax: "COLUMNS expects a regex; found {:?}", e),
                })
            },

            // ----
            // Window functions
            // ----
            // With an OVER clause, see `visit_window_value`.
            FirstValue => self.visit_unary_aggregate(Expr::first),
            LastValue => self.visit_unary(|e| e),
            NthValue => polars_bail!(SQLSyntax: "NTH_VALUE requires an OVER clause"),
            Lag => self.visit_window_offset_function(1),
            Lead => self.visit_window_offset_function(-1),
            Rank | DenseRank | PercentRank | CumeDist => self.visit_rank_function(&function_name),
            Ntile => self.visit_ntile(),
            RowNumber => {
                let args = extract_args(function)?;
                if !args.is_empty() {
                    polars_bail!(SQLSyntax: "ROW_NUMBER expects 0 arguments (found {})", args.len());
                }
                let (row_index, _) = window_row_index();
                self.apply_window_spec(row_index + lit(1i64))
            },

            // ----
            // User-defined
            // ----
            Udf(func_name) => self.visit_udf(&func_name),
        }
    }

    /// RANK, DENSE_RANK, PERCENT_RANK and CUME_DIST. The window sorts each partition, so rows
    /// with equal ORDER BY values (peers) are next to each other and get the same result.
    /// Without ORDER BY, all rows of the partition are peers.
    fn visit_rank_function(&mut self, function: &PolarsSQLFunctions) -> PolarsResult<Expr> {
        use PolarsSQLFunctions::*;
        let args = extract_args(self.func)?;
        if !args.is_empty() {
            polars_bail!(SQLSyntax: "{} expects 0 arguments (found {})", self.func.name, args.len());
        }
        if self.window.is_none() {
            polars_bail!(SQLSyntax: "{} requires an OVER clause", self.func.name);
        }
        let order_keys = self.parse_window_order_keys()?;
        let (row_index, n) = window_row_index();

        // With one ORDER BY key, the key is ranked directly, without sorting the window.
        #[cfg(feature = "rank")]
        if let ([(key, options)], Rank | DenseRank | PercentRank) =
            (order_keys.as_slice(), function)
        {
            let rank = rank_of_key(key.clone(), *options, matches!(function, DenseRank));
            let expr = match function {
                PercentRank => percent_rank(rank, n),
                _ => rank,
            };
            return self.apply_window_spec_with_order_keys(expr, Vec::new());
        }

        // The same keys sort the window and find the peers.
        let keys: Vec<Expr> = order_keys.iter().map(|(key, _)| key.clone()).collect();
        let row_number = row_index.clone() + lit(1i64);
        // The row number of the first peer.
        let rank = || {
            when(is_first_peer(&keys, &row_index))
                .then(row_number.clone())
                .otherwise(lit(0i64))
                .cum_max(false)
        };

        let expr = match function {
            Rank => rank(),
            DenseRank => is_first_peer(&keys, &row_index)
                .cast(DataType::Int64)
                .cum_sum(false),
            PercentRank => percent_rank(rank(), n),
            CumeDist => {
                // The row number of the last peer.
                let last_peer = when(is_last_peer(&keys, &row_index, &n))
                    .then(row_number.clone())
                    .otherwise(n.clone())
                    .cum_min(true);
                last_peer.cast(DataType::Float64) / n
            },
            _ => unreachable!(),
        };
        self.apply_window_spec_with_order_keys(expr, order_keys)
    }

    /// NTILE(k), with `k` a positive integer literal: with `q = n / k` and `rem = n % k` for `n`
    /// rows, the first `rem` buckets get `q + 1` rows and the others get `q` rows.
    fn visit_ntile(&mut self) -> PolarsResult<Expr> {
        if self.window.is_none() {
            polars_bail!(SQLSyntax: "NTILE requires an OVER clause");
        }
        let args = extract_args(self.func)?;
        let buckets = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                match parse_sql_expr(sql_expr, self.ctx, self.active_schema)? {
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                        i64::try_from(n).ok().filter(|n| *n > 0)
                    },
                    _ => None,
                }
            },
            _ => polars_bail!(SQLSyntax: "NTILE expects 1 argument (found {})", args.len()),
        };
        let Some(buckets) = buckets else {
            polars_bail!(SQLSyntax: "NTILE expects a positive integer literal as the number of buckets");
        };

        let (row_index, n) = window_row_index();
        let buckets = lit(buckets);
        let q = n.clone().floor_div(buckets.clone());
        let rem = n % buckets;
        let rows_in_larger_buckets = rem.clone() * (q.clone() + lit(1i64));
        let bucket = when(row_index.clone().lt(rows_in_larger_buckets))
            .then(row_index.clone().floor_div(q.clone() + lit(1i64)))
            .otherwise((row_index - rem).floor_div(q));
        self.apply_window_spec(bucket + lit(1i64))
    }

    /// LAG (`direction` 1) or LEAD (`direction` -1): the value `offset` rows before or after the
    /// current row, or the default (NULL if not given) where that row is outside the partition.
    fn visit_window_offset_function(&mut self, direction: i64) -> PolarsResult<Expr> {
        let Some(window_spec) = &self.window else {
            polars_bail!(SQLSyntax: "{} requires an OVER clause", self.func.name);
        };
        if window_spec.order_by.is_empty() {
            polars_bail!(SQLSyntax: "{} requires an ORDER BY in the OVER clause", self.func.name);
        }

        let args = extract_args(self.func)?;
        let (value, offset, default) = match args.as_slice() {
            [FunctionArgExpr::Expr(value)] => (value, None, None),
            [FunctionArgExpr::Expr(value), FunctionArgExpr::Expr(offset)] => {
                (value, Some(offset), None)
            },
            [
                FunctionArgExpr::Expr(value),
                FunctionArgExpr::Expr(offset),
                FunctionArgExpr::Expr(default),
            ] => (value, Some(offset), Some(default)),
            _ => polars_bail!(
                SQLSyntax: "{} expects 1 to 3 arguments (found {})",
                self.func.name, args.len()
            ),
        };
        let value = parse_sql_expr(value, self.ctx, self.active_schema)?;
        let offset = match offset {
            None => 1,
            Some(offset) => match parse_sql_expr(offset, self.ctx, self.active_schema)? {
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => n,
                offset => polars_bail!(SQLSyntax: "offset must be an integer (found {:?})", offset),
            },
        };
        // Each row reads the row `shift` rows before it.
        let max_offset = i128::from(MAX_ROW_OFFSET);
        let shift = (i128::from(direction) * offset).clamp(-max_offset, max_offset) as i64;
        // A scalar, as in `LAG(1)`, is the value at every row.
        let is_scalar = is_constant_key(&value);
        let expr = match default {
            None if !is_scalar => value.shift(lit(shift)),
            default => {
                let default = match default {
                    Some(default) => parse_sql_expr(default, self.ctx, self.active_schema)?,
                    None => Expr::Literal(LiteralValue::untyped_null()),
                };
                let shifted = if is_scalar {
                    value
                } else {
                    value.shift(lit(shift))
                };
                let (row_index, n) = window_row_index();
                let in_partition = row_index
                    .clone()
                    .gt_eq(lit(shift))
                    .and(row_index.lt(n + lit(shift)));
                when(in_partition).then(shifted).otherwise(default)
            },
        };
        self.apply_window_spec(expr)
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

        let expr = self
            .ctx
            .function_registry
            .get_udf(func_name)?
            .ok_or_else(|| polars_err!(SQLInterface: "UDF {} not found", func_name))?
            .call(args);

        self.apply_window_spec(expr)
    }

    fn is_distinct(&self) -> bool {
        matches!(
            &self.func.args,
            FunctionArguments::List(list)
                if list.duplicate_treatment == Some(DuplicateTreatment::Distinct)
        )
    }

    /// Raise for window shapes that would otherwise give a wrong result.
    fn check_window_shape(&self, function: &PolarsSQLFunctions) -> PolarsResult<()> {
        use PolarsSQLFunctions::*;
        let Some(spec) = &self.window else {
            return Ok(());
        };
        let is_distinct = self.is_distinct();
        let has_order_by = !spec.order_by.is_empty();
        let has_frame = spec.window_frame.is_some();

        if self.func.filter.is_some() && !is_frame_aggregate(function, is_distinct) {
            polars_bail!(
                SQLInterface: "'FILTER' combined with 'OVER' is not supported for {}",
                self.func.name
            );
        }
        // Only computed over the whole partition.
        let whole_partition = match function {
            #[cfg(feature = "approx_quantile")]
            ApproxQuantile => true,
            ArrayAgg | Corr | CovarPop | CovarSamp | Last | Median | QuantileCont
            | QuantileDisc | StdDev | StringAgg | Variance => true,
            Count => is_distinct,
            _ => false,
        };
        if whole_partition && has_order_by {
            polars_bail!(
                SQLInterface: "{} with ORDER BY in OVER is not supported yet; without ORDER BY it uses the whole partition",
                self.func.name
            );
        }
        if whole_partition && has_frame {
            polars_bail!(
                SQLInterface: "{} with a window frame is not supported yet",
                self.func.name
            );
        }
        Ok(())
    }

    /// Validate the window frame of functions other than the aggregates in
    /// `visit_window_aggregate`. Only `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is
    /// supported.
    fn validate_window_frame(&self, window_frame: &Option<WindowFrame>) -> PolarsResult<()> {
        if let Some(frame) = window_frame {
            match frame.units {
                WindowFrameUnits::Range => {
                    polars_bail!(
                        SQLInterface:
                        "RANGE-based window frames are not supported"
                    );
                },
                WindowFrameUnits::Groups => {
                    polars_bail!(
                        SQLInterface:
                        "GROUPS-based window frames are not supported"
                    );
                },
                WindowFrameUnits::Rows => {
                    if !matches!(
                        (&frame.start_bound, &frame.end_bound),
                        (
                            WindowFrameBound::Preceding(None),         // UNBOUNDED PRECEDING
                            None | Some(WindowFrameBound::CurrentRow)  // CURRENT ROW
                        )
                    ) {
                        polars_bail!(
                            SQLInterface:
                            "only 'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW' is currently supported; found 'ROWS BETWEEN {} AND {}'",
                            frame.start_bound,
                            frame.end_bound.as_ref().map_or("CURRENT ROW", |b| {
                                match b {
                                    WindowFrameBound::CurrentRow => "CURRENT ROW",
                                    WindowFrameBound::Preceding(_) => "N PRECEDING",
                                    WindowFrameBound::Following(_) => "N FOLLOWING",
                                }
                            })
                        );
                    }
                },
            }
        }
        Ok(())
    }

    /// Parse a value argument of the function currently being visited, as opposed to a
    /// parameter.
    ///
    /// Behaves like [`parse_sql_expr`] but also accounts for any
    /// active `FILTER (WHERE …)` clause from the surrounding call.
    fn parse_sql_arg(&mut self, expr: &SQLExpr) -> PolarsResult<Expr> {
        Ok(match self.parse_value_arg(expr)? {
            ValueArg::Rows(e) => e,
            // A function that reads its values once per row reads a constant once per row too.
            ValueArg::Constant(c) => self.apply_filter(polars_lazy::dsl::repeat(c, len())),
        })
    }

    /// Parse a value argument, keeping a constant that is read once per row apart (see
    /// [`ValueArg`]).
    fn parse_value_arg(&mut self, expr: &SQLExpr) -> PolarsResult<ValueArg> {
        let parsed = parse_sql_expr(expr, self.ctx, self.active_schema)?;
        Ok(if self.reads_rows && is_constant_key(&parsed) {
            ValueArg::Constant(parsed)
        } else {
            ValueArg::Rows(self.apply_filter(parsed))
        })
    }

    /// The number of rows an aggregate reads: all rows, or those that pass FILTER.
    fn rows_read(&self) -> Expr {
        match &self.filter {
            Some(pred) => pred.clone().sum(),
            None => len(),
        }
    }

    /// An aggregate whose value over a constant is that constant, if any row is read, as MIN
    /// or AVG.
    fn aggregate_value(
        &mut self,
        sql_expr: &SQLExpr,
        f: impl Fn(Expr) -> Expr,
    ) -> PolarsResult<Expr> {
        Ok(match self.parse_value_arg(sql_expr)? {
            ValueArg::Rows(e) => f(e),
            ValueArg::Constant(c) => when(self.rows_read().gt(lit(0)))
                .then(f(c))
                .otherwise(Expr::Literal(LiteralValue::untyped_null())),
        })
    }

    fn apply_filter(&self, expr: Expr) -> Expr {
        match &self.filter {
            Some(pred) => expr.filter(pred.clone()),
            None => expr,
        }
    }

    fn parse_array_inner_product_arg(&mut self, expr: &SQLExpr) -> PolarsResult<Expr> {
        // Keep ordinary SQL arrays List-backed. Only direct literals in this
        // function become scalar Arrays so native arr.dot can broadcast them.
        let array_expr = match expr {
            SQLExpr::Array(_) => expr,
            SQLExpr::Nested(inner) => return self.parse_array_inner_product_arg(inner),
            _ => return self.parse_sql_arg(expr),
        };
        let mut values = parse_sql_array(array_expr, self.ctx)?;
        // `arr.dot` has no decimal kernel; decimal literals take part approximately.
        if values.dtype().is_decimal() {
            values = values.cast(&DataType::Float64)?;
        }
        let width = values.len();
        Ok(self.apply_filter(lit(Scalar::new_array(values, width))))
    }

    fn visit_array_inner_product(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(lhs), FunctionArgExpr::Expr(rhs)] => Ok(self
                .parse_array_inner_product_arg(lhs)?
                .arr()
                .dot(self.parse_array_inner_product_arg(rhs)?)),
            _ => self.not_supported_error(),
        }
        .and_then(|e| self.apply_window_spec(e))
    }

    fn visit_unary(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        self.try_visit_unary(|e| Ok(f(e)))
    }

    fn try_visit_unary(&mut self, f: impl Fn(Expr) -> PolarsResult<Expr>) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => f(self.parse_sql_arg(sql_expr)?),
            [FunctionArgExpr::Wildcard] => {
                f(self.parse_sql_arg(&SQLExpr::Wildcard(AttachedToken::empty()))?)
            },
            _ => self.not_supported_error(),
        }
        .and_then(|e| self.apply_window_spec(e))
    }

    /// Like `visit_unary`, for an aggregate whose value over a constant is that constant.
    fn visit_unary_aggregate(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        let e = self.aggregate_value(sql_expr, f)?;
        self.apply_window_spec(e)
    }

    /// Like `visit_unary`, for STDDEV or VARIANCE: the sample spread of a constant is 0 when
    /// two or more rows are read.
    fn visit_spread_aggregate(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        let e = match self.parse_value_arg(sql_expr)? {
            ValueArg::Rows(e) => f(e),
            ValueArg::Constant(c) => when(self.rows_read().gt(lit(1)).and(c.clone().is_not_null()))
                .then(f(c).fill_null(lit(0.0)))
                .otherwise(Expr::Literal(LiteralValue::untyped_null())),
        };
        self.apply_window_spec(e)
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
            [
                FunctionArgExpr::Expr(sql_expr1),
                FunctionArgExpr::Expr(sql_expr2),
            ] => {
                let expr1 = self.parse_sql_arg(sql_expr1)?;
                let expr2 = Arg::from_sql_arg(sql_expr2, self)?;
                f(expr1, expr2)
            },
            _ => self.not_supported_error(),
        }
        .and_then(|e| self.apply_window_spec(e))
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
                expr_args.push(self.parse_sql_arg(sql_expr)?);
            } else {
                return self.not_supported_error();
            };
        }
        f(&expr_args).and_then(|e| self.apply_window_spec(e))
    }

    fn try_visit_ternary<Arg: FromSQLExpr>(
        &mut self,
        f: impl Fn(Expr, Arg, Arg) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        match args.as_slice() {
            [
                FunctionArgExpr::Expr(sql_expr1),
                FunctionArgExpr::Expr(sql_expr2),
                FunctionArgExpr::Expr(sql_expr3),
            ] => {
                let expr1 = self.parse_sql_arg(sql_expr1)?;
                let expr2 = Arg::from_sql_arg(sql_expr2, self)?;
                let expr3 = Arg::from_sql_arg(sql_expr3, self)?;
                f(expr1, expr2, expr3)
            },
            _ => self.not_supported_error(),
        }
        .and_then(|e| self.apply_window_spec(e))
    }

    fn visit_nullary(&self, f: impl Fn() -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        if !args.is_empty() {
            return self.not_supported_error();
        }
        Ok(f())
    }

    /// Apply in-arg "aggregate modifiers" inside an aggregate's argument
    /// list, eg: `ARRAY_AGG(DISTINCT x ORDER BY y LIMIT 5)`. Composes
    /// with the visitor-level `FILTER (WHERE …)` clause.
    fn apply_aggregate_clauses(
        &mut self,
        mut base: Expr,
        is_distinct: bool,
        clauses: &[FunctionArgumentClause],
        base_sql_expr: &SQLExpr,
        func_name: &str,
    ) -> PolarsResult<Expr> {
        let mut order_by_clause = None;
        let mut limit_clause = None;
        for clause in clauses {
            match clause {
                FunctionArgumentClause::OrderBy(order_exprs) => {
                    order_by_clause = Some(order_exprs.as_slice());
                },
                FunctionArgumentClause::Limit(limit_expr) => {
                    limit_clause = Some(limit_expr);
                },
                _ => {},
            }
        }
        if is_distinct {
            // DISTINCT: apply unique first, then sort the deduplicated result.
            base = base.unique_stable();
            if let Some(order_by) = order_by_clause {
                base = self.apply_order_by_to_distinct_array(base, order_by, base_sql_expr)?;
            }
        } else if let Some(order_by) = order_by_clause {
            base = self.apply_order_by(base, order_by)?;
        }
        if let Some(limit_expr) = limit_clause {
            let limit = parse_sql_expr(limit_expr, self.ctx, self.active_schema)?;
            match limit {
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n >= 0 => {
                    base = base.head(Some(n as usize))
                },
                _ => {
                    polars_bail!(SQLSyntax: "LIMIT in {} must be a positive integer", func_name)
                },
            };
        }
        Ok(base)
    }

    fn visit_arr_agg(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct, clauses) = extract_args_and_clauses(self.func)?;
        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let base = self.parse_sql_arg(sql_expr)?;
                let base = self.apply_aggregate_clauses(
                    base,
                    is_distinct,
                    &clauses,
                    sql_expr,
                    "ARRAY_AGG",
                )?;
                self.apply_window_spec(base.implode(true))
            },
            _ => {
                polars_bail!(SQLSyntax: "ARRAY_AGG must have exactly one argument; found {}", args.len())
            },
        }
    }

    #[cfg(feature = "approx_quantile")]
    fn visit_approx_quantile(&mut self) -> PolarsResult<Expr> {
        /// Matches the default of `Expr.approx_quantile` in Python.
        const DEFAULT_ERROR: f64 = 0.001;

        let args = extract_args(self.func)?;
        let (value_arg, quantile_arg, error_arg, method_arg) = match args.as_slice() {
            [FunctionArgExpr::Expr(v), FunctionArgExpr::Expr(q)] => (v, q, None, None),
            [
                FunctionArgExpr::Expr(v),
                FunctionArgExpr::Expr(q),
                FunctionArgExpr::Expr(e),
            ] => (v, q, Some(e), None),
            [
                FunctionArgExpr::Expr(v),
                FunctionArgExpr::Expr(q),
                FunctionArgExpr::Expr(e),
                FunctionArgExpr::Expr(m),
            ] => (v, q, Some(e), Some(m)),
            _ => polars_bail!(
                SQLSyntax: "APPROX_QUANTILE expects 2-4 arguments (found {})",
                args.len()
            ),
        };

        // Parameters are not subject to an active FILTER clause; only the values are.
        let quantile = parse_sql_expr(quantile_arg, self.ctx, self.active_schema)?;
        let quantile = parse_quantile_literal(quantile, "APPROX_QUANTILE", args[1])?;

        let error = match error_arg {
            Some(e) => match parse_sql_expr(e, self.ctx, self.active_schema)? {
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(f))) => f,
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => n as f64,
                e if let Some(f) = decimal_literal_to_f64(&e) => f,
                _ => {
                    polars_bail!(SQLSyntax: "invalid error value for APPROX_QUANTILE ({})", args[2])
                },
            },
            None => DEFAULT_ERROR,
        };
        let method = match method_arg {
            Some(m) => String::from_sql_arg(m, self)?.parse()?,
            None => ApproxQuantileMethod::Auto,
        };

        let expr = self.aggregate_value(value_arg, |e| {
            e.approx_quantile(quantile.clone(), error, false, method.clone())
        })?;
        self.apply_window_spec(expr)
    }

    fn visit_string_agg(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct, clauses) = extract_args_and_clauses(self.func)?;
        let (sql_expr, separator) = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => (sql_expr, lit(",")),
            [
                FunctionArgExpr::Expr(sql_expr),
                FunctionArgExpr::Expr(sep_sql_expr),
            ] => {
                // `GROUP_CONCAT` (SQLite) disallows DISTINCT together with a separator
                // argument; the standard `STRING_AGG`/`LISTAGG` forms allow it.
                let is_group_concat = self
                    .func
                    .name
                    .0
                    .first()
                    .and_then(|part| part.as_ident())
                    .is_some_and(|ident| ident.value.eq_ignore_ascii_case("group_concat"));
                if is_distinct && is_group_concat {
                    polars_bail!(SQLSyntax: "DISTINCT is only supported with a single argument in '{}'", self.func.name)
                }
                (
                    sql_expr,
                    parse_sql_expr(sep_sql_expr, self.ctx, self.active_schema)?,
                )
            },
            _ => polars_bail!(
                SQLSyntax: "STRING_AGG expects 1-2 arguments (found {})",
                args.len()
            ),
        };
        let base = self.parse_sql_arg(sql_expr)?;
        let base =
            self.apply_aggregate_clauses(base, is_distinct, &clauses, sql_expr, "STRING_AGG")?;
        let joined = base
            .clone()
            .cast(DataType::String)
            .implode(true)
            .list()
            .join(separator, true);

        self.apply_window_spec(
            when(base.clone().null_count().lt(base.len()))
                .then(joined)
                .otherwise(lit(LiteralValue::untyped_null())),
        )
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
                Expr::Literal(lv) if lv.extract_str().is_some() => {
                    Ok(if lv.extract_str().unwrap().is_empty() {
                        e.cast(DataType::List(Box::from(DataType::String)))
                            .list()
                            .join(sep, true)
                    } else {
                        e.cast(DataType::List(Box::from(DataType::String)))
                            .list()
                            .eval(element().fill_null(lit(lv.extract_str().unwrap())))
                            .list()
                            .join(sep, false)
                    })
                },
                _ => {
                    polars_bail!(SQLSyntax: "invalid null value for ARRAY_TO_STRING ({})", args[2])
                },
            }),
            _ => {
                polars_bail!(SQLSyntax: "ARRAY_TO_STRING expects 2-3 arguments (found {})", args.len())
            },
        }
    }

    /// `GROUPING(k1, ..., kn)` stands for a value that depends on the grouping set a
    /// row came from, so it is registered with the query and bound to its keys when
    /// the `GROUP BY` clause is processed.
    fn visit_grouping(&mut self) -> PolarsResult<Expr> {
        if self.func.over.is_some() {
            polars_bail!(SQLSyntax: "GROUPING() cannot be used as a window function");
        }
        if self.func.filter.is_some() {
            polars_bail!(SQLSyntax: "GROUPING() does not support a FILTER clause");
        }
        let n_args = extract_args(self.func)?.len();
        if n_args == 0 || n_args > MAX_GROUPING_ARGS {
            polars_bail!(
                SQLSyntax: "GROUPING() expects between 1 and {} arguments; found {}", MAX_GROUPING_ARGS, n_args
            );
        }
        let Some(args) = grouping_call_args(self.func) else {
            polars_bail!(SQLSyntax: "GROUPING() expects column expressions; found {}", self.func)
        };
        let name = PlSmallStr::from_string(self.func.to_string());
        Ok(col(self.ctx.register_grouping_call(args)).alias(name))
    }

    fn visit_avg(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        self.aggregate_value(sql_expr, |arg| {
            if is_distinct { arg.unique() } else { arg }.mean()
        })
    }

    /// Like `visit_unary`, but also accepts a DISTINCT modifier, which is a no-op for MIN/MAX.
    fn visit_min_max(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let (args, _) = extract_args_distinct(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        self.aggregate_value(sql_expr, f)
    }

    fn visit_count(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;
        let distinct_count = |e: Expr| e.clone().n_unique().sub(e.null_count().gt(lit(0)));
        let count_expr = match (is_distinct, args.as_slice()) {
            // COUNT(*), COUNT()
            (false, [FunctionArgExpr::Wildcard] | []) => self.rows_read(),
            // COUNT(<non-null literal>) is equivalent to COUNT(*)
            (false, [FunctionArgExpr::Expr(sql_expr)]) if is_non_null_literal(sql_expr) => {
                self.rows_read()
            },
            // COUNT(col)
            (false, [FunctionArgExpr::Expr(sql_expr)]) => match self.parse_value_arg(sql_expr)? {
                ValueArg::Rows(e) => e.count(),
                // A constant is counted once per row read, unless it is NULL.
                ValueArg::Constant(c) => c.count() * self.rows_read(),
            },
            // COUNT(DISTINCT col)
            (true, [FunctionArgExpr::Expr(sql_expr)]) => match self.parse_value_arg(sql_expr)? {
                ValueArg::Rows(e) => distinct_count(e),
                ValueArg::Constant(c) => when(self.rows_read().gt(lit(0)))
                    .then(distinct_count(c))
                    .otherwise(lit(0)),
            },
            _ => self.not_supported_error()?,
        };
        self.apply_window_spec(count_expr.cast(DataType::Int64))
    }

    fn visit_sum(&mut self) -> PolarsResult<Expr> {
        let (arg, is_distinct) = self.parse_sum_arg()?;
        Ok(match arg {
            ValueArg::Rows(arg) => sql_sum(if is_distinct { arg.unique() } else { arg }),
            ValueArg::Constant(c) => self.sum_of_constant(c, is_distinct),
        })
    }

    fn visit_total(&mut self) -> PolarsResult<Expr> {
        let (arg, is_distinct) = self.parse_sum_arg()?;
        let total = match arg {
            ValueArg::Rows(arg) => {
                let arg = if is_distinct { arg.unique() } else { arg };
                literal_sum(&arg).unwrap_or_else(|| arg.sum())
            },
            // TOTAL is 0 when no value is read.
            ValueArg::Constant(c) => self.sum_of_constant(c, is_distinct).fill_null(lit(0)),
        };
        Ok(total.cast(DataType::Float64))
    }

    fn parse_sum_arg(&mut self) -> PolarsResult<(ValueArg, bool)> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        Ok((self.parse_value_arg(sql_expr)?, is_distinct))
    }

    /// SUM of a constant: the constant times the number of rows read, or the constant itself
    /// with DISTINCT. NULL when no row is read.
    fn sum_of_constant(&self, c: Expr, is_distinct: bool) -> Expr {
        // Integer literals are summed as Int64, as in `literal_sum`.
        let c = match c {
            e @ Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(_))) => {
                e.cast(DataType::Int64)
            },
            e => e,
        };
        let rows = self.rows_read();
        let total = if is_distinct {
            c
        } else {
            c * rows.clone().cast(DataType::Int64)
        };
        when(rows.gt(lit(0)))
            .then(total)
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
    }

    fn apply_order_by(&mut self, expr: Expr, order_by: &[OrderByExpr]) -> PolarsResult<Expr> {
        let mut by = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        let mut nulls_last = Vec::with_capacity(order_by.len());

        for ob in order_by {
            // Note: ORDER BY exprs share their length with the (possibly filtered) base,
            // so they have to go through `parse_sql_arg` to apply any active FILTER.
            by.push(self.parse_sql_arg(&ob.expr)?);

            let options = order_by_sort_options(&ob.options);
            nulls_last.push(options.nulls_last);
            descending.push(options.descending);
        }
        Ok(expr.sort_by(
            by,
            SortMultipleOptions::default()
                .with_order_descending_multi(descending)
                .with_nulls_last_multi(nulls_last),
        ))
    }

    fn apply_order_by_to_distinct_array(
        &mut self,
        expr: Expr,
        order_by: &[OrderByExpr],
        base_sql_expr: &SQLExpr,
    ) -> PolarsResult<Expr> {
        // If ORDER BY references the base expression, use .sort() directly
        if order_by.len() == 1 && order_by[0].expr == *base_sql_expr {
            return Ok(expr.sort(order_by_sort_options(&order_by[0].options)));
        }
        // Otherwise, fall back to `sort_by` (may need to handle further edge-cases later)
        self.apply_order_by(expr, order_by)
    }

    /// The keys of the window ORDER BY with their sort options. Input-independent scalar keys,
    /// as `1`, are left out, as they don't change the order.
    pub(crate) fn parse_window_order_keys(&mut self) -> PolarsResult<Vec<(Expr, SortOptions)>> {
        let Some(spec) = &self.window else {
            return Ok(Vec::new());
        };
        let order_by = spec.order_by.clone();
        let mut keys = Vec::with_capacity(order_by.len());
        for o in &order_by {
            let key = parse_sql_expr(&o.expr, self.ctx, self.active_schema)?;
            if !is_constant_key(&key) {
                keys.push((key, order_by_sort_options(&o.options)));
            }
        }
        Ok(keys)
    }

    fn apply_window_spec(&mut self, expr: Expr) -> PolarsResult<Expr> {
        let order_keys = self.parse_window_order_keys()?;
        self.apply_window_spec_with_order_keys(expr, order_keys)
    }

    /// `apply_window_spec` with the ORDER BY keys from `parse_window_order_keys`.
    fn apply_window_spec_with_order_keys(
        &mut self,
        expr: Expr,
        order_keys: Vec<(Expr, SortOptions)>,
    ) -> PolarsResult<Expr> {
        let Some(window_spec) = &self.window else {
            return Ok(expr);
        };
        self.validate_window_frame(&window_spec.window_frame)?;
        self.apply_over(expr, Vec::new(), order_keys)
    }

    /// Evaluate `expr` per window partition, split further by `extra_partition_by` and sorted
    /// by `order_keys`.
    pub(crate) fn apply_over(
        &mut self,
        expr: Expr,
        extra_partition_by: Vec<Expr>,
        order_keys: Vec<(Expr, SortOptions)>,
    ) -> PolarsResult<Expr> {
        let Some(window_spec) = self.window.clone() else {
            return Ok(expr);
        };
        // A constant key, as in `PARTITION BY 1`, does not split the frame.
        let mut partition_by = Vec::with_capacity(window_spec.partition_by.len());
        for p in &window_spec.partition_by {
            let key = parse_sql_expr(p, self.ctx, self.active_schema)?;
            if !is_constant_key(&key) {
                partition_by.push(key);
            }
        }
        partition_by.extend(extra_partition_by);
        let partition_by = (!partition_by.is_empty()).then_some(partition_by);
        let order_by = window_sort_key(order_keys).map(|(key, options)| (vec![key], options));

        // Apply window spec; an empty window still has to be told apart from an
        // aggregate (see `GroupScope::mark_whole_frame_windows`).
        Ok(match (partition_by, order_by) {
            (None, None) if self.ctx.group_scope.mark_whole_frame_windows => {
                expr.over([col(self.ctx.whole_frame_partition())])?
            },
            (None, None) => expr,
            (Some(part), None) => expr.over(part)?,
            (part, Some(order)) => expr.over_with_options(part, Some(order), Default::default())?,
        })
    }

    pub(crate) fn not_supported_error<T>(&self) -> PolarsResult<T> {
        polars_bail!(
            SQLInterface:
            "no function matches the given name and arguments: `{}`",
            self.func.to_string()
        );
    }
}

/// Whether a window key is a scalar that doesn't depend on the input, as `1` or `LOWER('A')`.
pub(crate) fn is_constant_key(key: &Expr) -> bool {
    matches!(key.clone().meta().is_input_independent_scalar(), Ok(true))
}

/// The sort key of a window ORDER BY. Several keys are row-encoded into one, so that each key
/// keeps its own direction and NULL order.
fn window_sort_key(mut keys: Vec<(Expr, SortOptions)>) -> Option<(Expr, SortOptions)> {
    // Rows with equal keys (peers) may be processed in any order.
    match keys.len() {
        0 => None,
        1 => {
            let (key, options) = keys.pop().unwrap();
            Some((key, options.with_maintain_order(false)))
        },
        _ => {
            let (descending, nulls_last) = keys
                .iter()
                .map(|(_, options)| (options.descending, options.nulls_last))
                .unzip();
            let key = Expr::n_ary(
                FunctionExpr::RowEncode(RowEncodingVariant::Ordered {
                    descending: Some(descending),
                    nulls_last: Some(nulls_last),
                    broadcast_nulls: None,
                }),
                keys.into_iter().map(|(key, _)| key).collect(),
            );
            Some((key, SortOptions::default().with_maintain_order(false)))
        },
    }
}

/// RANK, or DENSE_RANK if `dense`, of a single ORDER BY key. NULL keys are peers, and come
/// first or last as `options` says.
#[cfg(feature = "rank")]
fn rank_of_key(key: Expr, options: SortOptions, dense: bool) -> Expr {
    let method = if dense {
        RankMethod::Dense
    } else {
        RankMethod::Min
    };
    let rank = key
        .clone()
        .rank(
            RankOptions {
                method,
                descending: options.descending,
            },
            None,
        )
        .cast(DataType::Int64);
    // The engine gives NULL keys a NULL rank.
    if options.nulls_last {
        let null_rank = if dense {
            rank.clone().max().fill_null(lit(0i64)) + lit(1i64)
        } else {
            key.count().cast(DataType::Int64) + lit(1i64)
        };
        rank.fill_null(null_rank)
    } else {
        let null_count = key.null_count().cast(DataType::Int64);
        let offset = if dense {
            null_count.gt(lit(0i64)).cast(DataType::Int64)
        } else {
            null_count
        };
        (rank + offset).fill_null(lit(1i64))
    }
}

/// PERCENT_RANK from the rank and the number of rows `n` in the window.
fn percent_rank(rank: Expr, n: Expr) -> Expr {
    when(n.clone().gt(lit(1i64)))
        .then((rank - lit(1i64)).cast(DataType::Float64) / (n - lit(1i64)))
        .otherwise(lit(0.0))
}

/// Whether each row is the first of its peers in the sorted window: the first row, or a row
/// where a key differs from the row before.
pub(crate) fn is_first_peer(keys: &[Expr], row_index: &Expr) -> Expr {
    keys.iter()
        .fold(row_index.clone().eq(lit(0i64)), |acc, key| {
            acc.or(key.clone().neq_missing(key.clone().shift(lit(1))))
        })
}

/// Whether each row is the last of its peers in the sorted window of `n` rows.
pub(crate) fn is_last_peer(keys: &[Expr], row_index: &Expr, n: &Expr) -> Expr {
    keys.iter()
        .fold(row_index.clone().eq(n.clone() - lit(1i64)), |acc, key| {
            acc.or(key.clone().neq_missing(key.clone().shift(lit(-1))))
        })
}

/// The row index (from 0) in the window partition, and the number of rows in it.
pub(crate) fn window_row_index() -> (Expr, Expr) {
    let n = len().cast(DataType::Int64);
    (int_range(lit(0i64), n.clone(), 1, DataType::Int64), n)
}

/// SQL semantics require `NULL` when there are no complete (eg: both non-null)
/// pairs to correlate, whereas Polars' native `pearson_corr` returns `NaN`.
fn sql_corr(a: Expr, b: Expr) -> Expr {
    let has_corr_pairs = a
        .clone()
        .is_not_null()
        .and(b.clone().is_not_null())
        .any(true);

    when(has_corr_pairs)
        .then(polars_lazy::dsl::pearson_corr(a, b))
        .otherwise(lit(LiteralValue::untyped_null()))
}

/// Returns true if the SQL expression is a non-null literal value (e.g. `1`, `'hello'`, `TRUE`).
pub(crate) fn is_non_null_literal(expr: &SQLExpr) -> bool {
    matches!(
        expr,
        SQLExpr::Value(ValueWithSpan {
            value: v,
            ..
        }) if !matches!(v, SQLValue::Null)
    )
}

/// A decimal as `Float64` when planned, for functions only defined on floats.
fn sql_to_float(e: Expr) -> Expr {
    e.map_unary(FunctionExpr::Sql(SqlFunction::ToFloat))
}

/// SQL `ROUND`: decimals round half away from zero (as in Postgres), other types keep the
/// default rounding.
fn sql_round(e: Expr, decimals: u32) -> Expr {
    e.map_unary(FunctionExpr::Sql(SqlFunction::Round { decimals }))
}

/// SQL `SUM`: NULL, not 0, when there are no non-NULL values.
pub(crate) fn sql_sum(arg: Expr) -> Expr {
    let (total, non_empty) = match literal_sum(&arg) {
        Some(total) => (total, len().gt(lit(0))),
        None => (arg.clone().sum(), arg.count().gt(lit(0))),
    };
    when(non_empty)
        .then(total)
        .otherwise(Expr::Literal(LiteralValue::untyped_null()))
}

/// SUM of a numeric literal counts it once per row.
fn literal_sum(arg: &Expr) -> Option<Expr> {
    match arg {
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(_))) => {
            Some(arg.clone() * len().cast(DataType::Int64))
        },
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(_))) => Some(arg.clone() * len()),
        _ if decimal_literal(arg).is_some() => Some(arg.clone() * len()),
        _ => None,
    }
}

/// Parse a literal quantile argument, validating that it lies in [0, 1].
fn parse_quantile_literal(
    quantile: Expr,
    fname: &str,
    arg: &FunctionArgExpr,
) -> PolarsResult<Expr> {
    match quantile {
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(f))) if (0.0..=1.0).contains(&f) => {
            Ok(Expr::from(f))
        },
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if (0..=1).contains(&n) => {
            Ok(Expr::from(n as f64))
        },
        ref e if let Some(f) = decimal_literal_to_f64(e) => {
            if !(0.0..=1.0).contains(&f) {
                polars_bail!(SQLSyntax: "{} value must be between 0 and 1 ({})", fname, arg)
            }
            Ok(Expr::from(f))
        },
        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(_) | DynLiteralValue::Int(_))) => {
            polars_bail!(SQLSyntax: "{} value must be between 0 and 1 ({})", fname, arg)
        },
        _ => polars_bail!(SQLSyntax: "invalid value for {} ({})", fname, arg),
    }
}

pub(crate) fn extract_args(func: &SQLFunction) -> PolarsResult<Vec<&FunctionArgExpr>> {
    let (args, _, _) = _extract_func_args(func, false, false)?;
    Ok(args)
}

pub(crate) fn extract_args_distinct(
    func: &SQLFunction,
) -> PolarsResult<(Vec<&FunctionArgExpr>, bool)> {
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
                        FunctionArg::ExprNamed { arg, .. } => arg,
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

    /// Parse SQL expression as an argument of the function being visited, taking
    /// the surrounding visitor into account. Allows active `FILTER (WHERE …)`
    /// clauses to be applied to all args without each call knowing about FILTER.
    fn from_sql_arg(expr: &SQLExpr, visitor: &mut SQLFunctionVisitor<'_>) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        Self::from_sql_expr(expr, visitor.ctx)
    }
}

impl FromSQLExpr for f64 {
    fn from_sql_expr(expr: &SQLExpr, _ctx: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(ValueWithSpan { value: v, .. }) => match v {
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
            SQLExpr::Value(ValueWithSpan { value: v, .. }) => match v {
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
            SQLExpr::Value(ValueWithSpan { value: v, .. }) => match v {
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
            SQLExpr::Value(ValueWithSpan { value: v, .. }) => match v {
                SQLValue::SingleQuotedString(s) => Ok(StrptimeOptions {
                    format: Some(PlSmallStr::from_str(s)),
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

    fn from_sql_arg(expr: &SQLExpr, visitor: &mut SQLFunctionVisitor<'_>) -> PolarsResult<Self> {
        visitor.parse_sql_arg(expr)
    }
}
