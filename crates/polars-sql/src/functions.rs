use std::ops::{Add, Sub};

use polars_core::chunked_array::ops::{FillNullStrategy, SortMultipleOptions, SortOptions};
use polars_core::prelude::{
    DataType, ExplodeOptions, PolarsResult, QuantileMethod, Schema, TimeUnit, polars_bail,
    polars_err,
};
use polars_lazy::dsl::Expr;
#[cfg(feature = "rank")]
use polars_lazy::prelude::{RankMethod, RankOptions};
use polars_ops::chunked_array::UnicodeForm;
use polars_ops::series::RoundMode;
use polars_plan::dsl::functions::{
    as_struct, coalesce, col, cols, concat_str, element, int_range, len, lit, max_horizontal,
    min_horizontal, when,
};
use polars_plan::plans::{DynLiteralValue, LiteralValue, typed_lit};
use polars_plan::prelude::StrptimeOptions;
use polars_utils::pl_str::PlSmallStr;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    DateTimeField, DuplicateTreatment, Expr as SQLExpr, Function as SQLFunction, FunctionArg,
    FunctionArgExpr, FunctionArgumentClause, FunctionArgumentList, FunctionArguments, Ident,
    OrderByExpr, Value as SQLValue, ValueWithSpan, WindowFrame, WindowFrameBound, WindowFrameUnits,
    WindowSpec, WindowType,
};
use sqlparser::tokenizer::Span;

use crate::SQLContext;
use crate::sql_expr::{adjust_one_indexed_param, parse_extract_date_part, parse_sql_expr};

pub(crate) struct SQLFunctionVisitor<'a> {
    pub(crate) func: &'a SQLFunction,
    pub(crate) ctx: &'a mut SQLContext,
    pub(crate) active_schema: Option<&'a Schema>,
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
    /// Round a number to `x` decimals (default: 0) away from zero.
    ///   .5 is rounded away from zero.
    /// ```sql
    /// SELECT ROUND(col1, 3) FROM df;
    /// ```
    Round,
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
    /// SQL 'sum' function.
    /// Returns the sum of all the elements in the grouping.
    /// ```sql
    /// SELECT SUM(col1) FROM df;
    /// ```
    Sum,
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
    /// SQL 'unnest' function.
    /// Unnest/explodes an array column into multiple rows.
    /// ```sql
    /// SELECT unnest(col1) FROM df;
    /// ```
    Explode,
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
    /// With default frame, returns the current row's value.
    /// ```sql
    /// SELECT LAST_VALUE(col1) OVER (PARTITION BY category ORDER BY id) FROM df;
    /// ```
    LastValue,
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
    #[cfg(feature = "rank")]
    Rank,
    /// SQL 'dense_rank' function.
    /// Returns the rank of each row within a window partition, without gaps for ties.
    /// Rows with equal values receive the same rank, and the next rank is consecutive.
    /// ```sql
    /// SELECT DENSE_RANK() OVER (ORDER BY col1) FROM df;
    /// SELECT DENSE_RANK() OVER (PARTITION BY col1 ORDER BY col2 DESC) FROM df;
    /// ```
    #[cfg(feature = "rank")]
    DenseRank,

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
            "date",
            "date_part",
            "degrees",
            "dense_rank",
            "ends_with",
            "exp",
            "first",
            "first_value",
            "floor",
            "greatest",
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
            "quantile_disc",
            "min",
            "mod",
            "nullif",
            "octet_length",
            "pi",
            "pow",
            "power",
            "quantile_cont",
            "quantile_disc",
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
            // Temporal functions
            // ----
            "date" => Self::Date,
            "date_part" => Self::DatePart,
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
            "avg" => Self::Avg,
            "corr" => Self::Corr,
            "count" => Self::Count,
            "covar_pop" => Self::CovarPop,
            "covar_samp" | "covar" => Self::CovarSamp,
            "first" => Self::First,
            "last" => Self::Last,
            "max" => Self::Max,
            "median" => Self::Median,
            "min" => Self::Min,
            "quantile_cont" => Self::QuantileCont,
            "quantile_disc" => Self::QuantileDisc,
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
            // Window functions
            // ----
            #[cfg(feature = "rank")]
            "dense_rank" => Self::DenseRank,
            "first_value" => Self::FirstValue,
            "last_value" => Self::LastValue,
            "lag" => Self::Lag,
            "lead" => Self::Lead,
            #[cfg(feature = "rank")]
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
}

impl SQLFunctionVisitor<'_> {
    pub(crate) fn visit_function(&mut self) -> PolarsResult<Expr> {
        use PolarsSQLFunctions::*;
        use polars_lazy::prelude::Literal;

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
            Div => self.visit_binary(|e, d| e.floor_div(d).cast(DataType::Int64)),
            Exp => self.visit_unary(Expr::exp),
            Floor => self.visit_unary(Expr::floor),
            Ln => self.visit_unary(|e| log_with_base(e, std::f64::consts::E)),
            Log => self.visit_binary(Expr::log),
            Log10 => self.visit_unary(|e| log_with_base(e, 10.0)),
            Log1p => self.visit_unary(Expr::log1p),
            Log2 => self.visit_unary(|e| log_with_base(e, 2.0)),
            Pi => self.visit_nullary(Expr::pi),
            Mod => self.visit_binary(|e1, e2| e1 % e2),
            Pow => self.visit_binary::<Expr>(Expr::pow),
            Round => {
                let args = extract_args(function)?;
                match args.len() {
                    1 => self.visit_unary(|e| e.round(0, RoundMode::default())),
                    2 => self.try_visit_binary(|e, decimals| {
                        Ok(e.round(match decimals {
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                                if n >= 0 { n as u32 } else {
                                    polars_bail!(SQLInterface: "ROUND does not currently support negative decimals value ({})", args[1])
                                }
                            },
                            _ => polars_bail!(SQLSyntax: "invalid value for ROUND decimals ({})", args[1]),
                        }, RoundMode::default()))
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
            Avg => self.visit_unary(Expr::mean),
            Corr => self.visit_binary(polars_lazy::dsl::pearson_corr),
            Count => self.visit_count(),
            CovarPop => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 0)),
            CovarSamp => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 1)),
            First => self.visit_unary(Expr::first),
            Last => self.visit_unary(Expr::last),
            Max => self.visit_unary_with_opt_cumulative(Expr::max, Expr::cum_max),
            Median => self.visit_unary(Expr::median),
            QuantileCont | QuantileDisc => {
                let (fname, method) = if matches!(function_name, QuantileCont) {
                    ("QUANTILE_CONT", QuantileMethod::Linear)
                } else {
                    ("QUANTILE_DISC", QuantileMethod::Equiprobable)
                };
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.try_visit_binary(|e, q| {
                        let value = match q {
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Float(f))) => {
                                if (0.0..=1.0).contains(&f) {
                                    Expr::from(f)
                                } else {
                                    polars_bail!(SQLSyntax: "{} value must be between 0 and 1 ({})", fname, args[1])
                                }
                            },
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                                if (0..=1).contains(&n) {
                                    Expr::from(n as f64)
                                } else {
                                    polars_bail!(SQLSyntax: "{} value must be between 0 and 1 ({})", fname, args[1])
                                }
                            },
                            _ => polars_bail!(SQLSyntax: "invalid value for {} ({})", fname, args[1])
                        };
                        Ok(e.quantile(value, method))
                    }),
                    _ => polars_bail!(SQLSyntax: "{} expects 2 arguments (found {})", fname, args.len()),
                }
            },
            Min => self.visit_unary_with_opt_cumulative(Expr::min, Expr::cum_min),
            StdDev => self.visit_unary(|e| e.std(1)),
            Sum => self.visit_unary_with_opt_cumulative(Expr::sum, Expr::cum_sum),
            Variance => self.visit_unary(|e| e.var(1)),

            // ----
            // Array functions
            // ----
            ArrayAgg => self.visit_arr_agg(),
            ArrayContains => self.visit_binary::<Expr>(|e, s| e.list().contains(s, true)),
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
            ArrayUnique => self.visit_unary(|e| e.list().unique_stable()),
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
            FirstValue => self.visit_unary(Expr::first),
            LastValue => {
                // With the default window frame (ROWS UNBOUNDED PRECEDING TO CURRENT ROW),
                // LAST_VALUE returns the last value from the start of the partition up
                // to the current row - which is simply the current row's value.
                let args = extract_args(function)?;
                match args.as_slice() {
                    [FunctionArgExpr::Expr(sql_expr)] => {
                        parse_sql_expr(sql_expr, self.ctx, self.active_schema)
                    },
                    _ => polars_bail!(
                        SQLSyntax: "LAST_VALUE expects exactly 1 argument (found {})",
                        args.len()
                    ),
                }
            },
            Lag => self.visit_window_offset_function(1),
            Lead => self.visit_window_offset_function(-1),
            #[cfg(feature = "rank")]
            Rank | DenseRank => {
                let (func_name, rank_method) = match function_name {
                    Rank => ("RANK", RankMethod::Min),
                    DenseRank => ("DENSE_RANK", RankMethod::Dense),
                    _ => unreachable!(),
                };
                let args = extract_args(function)?;
                if !args.is_empty() {
                    polars_bail!(SQLSyntax: "{} expects 0 arguments (found {})", func_name, args.len());
                }
                let window_spec = match &self.func.over {
                    Some(WindowType::WindowSpec(spec)) if !spec.order_by.is_empty() => spec,
                    _ => {
                        polars_bail!(SQLSyntax: "{} requires an OVER clause with ORDER BY", func_name)
                    },
                };
                let (order_exprs, all_desc) =
                    self.parse_order_by_in_window(&window_spec.order_by)?;
                let rank_expr = if order_exprs.len() == 1 {
                    order_exprs[0].clone().rank(
                        RankOptions {
                            method: rank_method,
                            descending: all_desc,
                        },
                        None,
                    )
                } else {
                    as_struct(order_exprs).rank(
                        RankOptions {
                            method: rank_method,
                            descending: all_desc,
                        },
                        None,
                    )
                };
                self.apply_window_spec(rank_expr, &self.func.over)
            },
            RowNumber => {
                let args = extract_args(function)?;
                if !args.is_empty() {
                    polars_bail!(SQLSyntax: "ROW_NUMBER expects 0 arguments (found {})", args.len());
                }
                // note: SQL is 1-indexed
                let row_num_expr = int_range(lit(0i64), len(), 1, DataType::UInt32) + lit(1u32);
                self.apply_window_spec(row_num_expr, &self.func.over)
            },

            // ----
            // User-defined
            // ----
            Udf(func_name) => self.visit_udf(&func_name),
        }
    }

    fn visit_window_offset_function(&mut self, offset_multiplier: i64) -> PolarsResult<Expr> {
        // LAG/LEAD require an OVER clause
        if self.func.over.is_none() {
            polars_bail!(SQLSyntax: "{} requires an OVER clause", self.func.name);
        }

        // LAG/LEAD require ORDER BY in the OVER clause
        let window_type = self.func.over.as_ref().unwrap();
        let window_spec = self.resolve_window_spec(window_type)?;
        if window_spec.order_by.is_empty() {
            polars_bail!(SQLSyntax: "{} requires an ORDER BY in the OVER clause", self.func.name);
        }

        let args = extract_args(self.func)?;

        match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                Ok(expr.shift(offset_multiplier.into()))
            },
            [FunctionArgExpr::Expr(sql_expr), FunctionArgExpr::Expr(offset_expr)] => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                let offset = parse_sql_expr(offset_expr, self.ctx, self.active_schema)?;
                if let Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) = offset {
                    if n <= 0 {
                        polars_bail!(SQLSyntax: "offset must be positive (found {})", n)
                    }
                    Ok(expr.shift((offset_multiplier * n as i64).into()))
                } else {
                    polars_bail!(SQLSyntax: "offset must be an integer (found {:?})", offset)
                }
            },
            _ => polars_bail!(SQLSyntax: "{} expects 1 or 2 arguments (found {})", self.func.name, args.len()),
        }.and_then(|e| self.apply_window_spec(e, &self.func.over))
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

        Ok(self
            .ctx
            .function_registry
            .get_udf(func_name)?
            .ok_or_else(|| polars_err!(SQLInterface: "UDF {} not found", func_name))?
            .call(args))
    }

    /// Validate window frame specifications.
    ///
    /// Polars only supports ROWS frame semantics, and does
    /// not currently support customising the window.
    ///
    /// **Supported Frame Spec**
    /// - `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
    ///
    /// **Unsupported Frame Spec**
    /// - `RANGE ...` (peer group semantics not implemented)
    /// - `GROUPS ...` (peer group semantics not implemented)
    /// - `ROWS` with other bounds (e.g., `<n> PRECEDING`, `FOLLOWING`, etc)
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

    /// Window specs that map to cumulative functions.
    ///
    /// Converts SQL window functions with ORDER BY to compatible cumulative ops:
    /// - `SUM(a) OVER (ORDER BY b)` → `a.cum_sum().over(order_by=b)`
    /// - `MAX(a) OVER (ORDER BY b)` → `a.cum_max().over(order_by=b)`
    /// - `MIN(a) OVER (ORDER BY b)` → `a.cum_min().over(order_by=b)`
    ///
    /// ROWS vs RANGE Semantics (show default behaviour if no frame spec):
    ///
    /// **Polars (ROWS)**
    /// Each row gets its own cumulative value row-by-row.
    /// ```text
    /// Data: [(A,X,10), (A,X,15), (A,Y,20)]
    /// Query: SUM(value) OVER (ORDER BY category, subcategory)
    /// Result: [10, 25, 45]  ← row-by-row cumulative
    /// ```
    ///
    /// **SQL (RANGE)**
    /// Rows with identical ORDER BY values (peers) get the same result.
    /// ```text
    /// Same data, query with RANGE (eg: using a relational DB):
    /// Result: [25, 25, 45]  ← both (A,X) rows get 25
    /// ```
    fn apply_cumulative_window(
        &mut self,
        f: impl Fn(Expr) -> Expr,
        cumulative_fn: impl Fn(Expr, bool) -> Expr,
        WindowSpec {
            partition_by,
            order_by,
            window_frame,
            ..
        }: &WindowSpec,
    ) -> PolarsResult<Expr> {
        self.validate_window_frame(window_frame)?;

        if !order_by.is_empty() {
            // Extract ORDER BY exprs and sort direction
            let (order_by_exprs, all_desc) = self.parse_order_by_in_window(order_by)?;

            // Get the base expr/column
            let args = extract_args(self.func)?;
            let base_expr = match args.as_slice() {
                [FunctionArgExpr::Expr(sql_expr)] => {
                    parse_sql_expr(sql_expr, self.ctx, self.active_schema)?
                },
                _ => return self.not_supported_error(),
            };
            let partition_by_exprs = if partition_by.is_empty() {
                None
            } else {
                Some(
                    partition_by
                        .iter()
                        .map(|p| parse_sql_expr(p, self.ctx, self.active_schema))
                        .collect::<PolarsResult<Vec<_>>>()?,
                )
            };

            // Apply cumulative function; the forward-fill ensures we match SQL semantics
            let cumulative_expr = cumulative_fn(base_expr, false)
                .fill_null_with_strategy(FillNullStrategy::Forward(None));
            let sort_opts = SortOptions::default().with_order_descending(all_desc);
            cumulative_expr.over_with_options(
                partition_by_exprs,
                Some((order_by_exprs, sort_opts)),
                Default::default(),
            )
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
                &SQLExpr::Wildcard(AttachedToken::empty()),
                self.ctx,
                self.active_schema,
            )?),
            _ => self.not_supported_error(),
        }
        .and_then(|e| self.apply_window_spec(e, &self.func.over))
    }

    /// Resolve a WindowType to a concrete WindowSpec (handles named window references)
    fn resolve_window_spec(&self, window_type: &WindowType) -> PolarsResult<WindowSpec> {
        match window_type {
            WindowType::WindowSpec(spec) => Ok(spec.clone()),
            WindowType::NamedWindow(name) => self
                .ctx
                .named_windows
                .get(&name.value)
                .cloned()
                .ok_or_else(|| {
                    polars_err!(
                        SQLInterface:
                        "named window '{}' was not found",
                        name.value
                    )
                }),
        }
    }

    /// Some functions have cumulative equivalents that can be applied to window specs
    /// e.g. SUM(a) OVER (ORDER BY b DESC) -> CUMSUM(a, false)
    fn visit_unary_with_opt_cumulative(
        &mut self,
        f: impl Fn(Expr) -> Expr,
        cumulative_fn: impl Fn(Expr, bool) -> Expr,
    ) -> PolarsResult<Expr> {
        match self.func.over.as_ref() {
            Some(window_type) => {
                let spec = self.resolve_window_spec(window_type)?;
                self.apply_cumulative_window(f, cumulative_fn, &spec)
            },
            None => self.visit_unary(f),
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
            [
                FunctionArgExpr::Expr(sql_expr1),
                FunctionArgExpr::Expr(sql_expr2),
            ] => {
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
            [
                FunctionArgExpr::Expr(sql_expr1),
                FunctionArgExpr::Expr(sql_expr2),
                FunctionArgExpr::Expr(sql_expr3),
            ] => {
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
                let mut order_by_clause = None;
                let mut limit_clause = None;
                for clause in &clauses {
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
                if !is_distinct {
                    // No DISTINCT: apply ORDER BY normally
                    if let Some(order_by) = order_by_clause {
                        base = self.apply_order_by(base, order_by)?;
                    }
                } else {
                    // DISTINCT: apply unique, then sort the result
                    base = base.unique_stable();
                    if let Some(order_by) = order_by_clause {
                        base = self.apply_order_by_to_distinct_array(base, order_by, sql_expr)?;
                    }
                }
                if let Some(limit_expr) = limit_clause {
                    let limit = parse_sql_expr(limit_expr, self.ctx, self.active_schema)?;
                    match limit {
                        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n >= 0 => {
                            base = base.head(Some(n as usize))
                        },
                        _ => {
                            polars_bail!(SQLSyntax: "LIMIT in ARRAY_AGG must be a positive integer")
                        },
                    };
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

    fn visit_count(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;

        // Window function with an ORDER BY clause?
        let has_order_by = match &self.func.over {
            Some(WindowType::WindowSpec(spec)) => !spec.order_by.is_empty(),
            _ => false,
        };
        if has_order_by && !is_distinct {
            if let Some(WindowType::WindowSpec(spec)) = &self.func.over {
                self.validate_window_frame(&spec.window_frame)?;

                match args.as_slice() {
                    [FunctionArgExpr::Wildcard] | [] => {
                        // COUNT(*) with ORDER BY -> map to `int_range`
                        let (order_by_exprs, all_desc) =
                            self.parse_order_by_in_window(&spec.order_by)?;
                        let partition_by_exprs = if spec.partition_by.is_empty() {
                            None
                        } else {
                            Some(
                                spec.partition_by
                                    .iter()
                                    .map(|p| parse_sql_expr(p, self.ctx, self.active_schema))
                                    .collect::<PolarsResult<Vec<_>>>()?,
                            )
                        };
                        let sort_opts = SortOptions::default().with_order_descending(all_desc);
                        let row_number = int_range(lit(0), len(), 1, DataType::Int64).add(lit(1)); // SQL is 1-indexed

                        return row_number.over_with_options(
                            partition_by_exprs,
                            Some((order_by_exprs, sort_opts)),
                            Default::default(),
                        );
                    },
                    [FunctionArgExpr::Expr(_)] => {
                        // COUNT(column) with ORDER BY -> use cum_count
                        return self.visit_unary_with_opt_cumulative(
                            |e| e.count(),
                            |e, reverse| e.cum_count(reverse),
                        );
                    },
                    _ => {},
                }
            }
        }
        let count_expr = match (is_distinct, args.as_slice()) {
            // COUNT(*), COUNT()
            (false, [FunctionArgExpr::Wildcard] | []) => len(),
            // COUNT(col)
            (false, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                expr.count()
            },
            // COUNT(DISTINCT col)
            (true, [FunctionArgExpr::Expr(sql_expr)]) => {
                let expr = parse_sql_expr(sql_expr, self.ctx, self.active_schema)?;
                expr.clone().n_unique().sub(expr.null_count().gt(lit(0)))
            },
            _ => self.not_supported_error()?,
        };
        self.apply_window_spec(count_expr, &self.func.over)
    }

    fn apply_order_by(&mut self, expr: Expr, order_by: &[OrderByExpr]) -> PolarsResult<Expr> {
        let mut by = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        let mut nulls_last = Vec::with_capacity(order_by.len());

        for ob in order_by {
            // Note: if not specified 'NULLS FIRST' is default for DESC, 'NULLS LAST' otherwise
            // https://www.postgresql.org/docs/current/queries-order.html
            let desc_order = !ob.options.asc.unwrap_or(true);
            by.push(parse_sql_expr(&ob.expr, self.ctx, self.active_schema)?);
            nulls_last.push(!ob.options.nulls_first.unwrap_or(desc_order));
            descending.push(desc_order);
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
            let desc_order = !order_by[0].options.asc.unwrap_or(true);
            let nulls_last = !order_by[0].options.nulls_first.unwrap_or(desc_order);
            return Ok(expr.sort(
                SortOptions::default()
                    .with_order_descending(desc_order)
                    .with_nulls_last(nulls_last)
                    .with_maintain_order(true),
            ));
        }
        // Otherwise, fall back to `sort_by` (may need to handle further edge-cases later)
        self.apply_order_by(expr, order_by)
    }

    /// Parse ORDER BY (in OVER clause), validating uniform direction.
    fn parse_order_by_in_window(
        &mut self,
        order_by: &[OrderByExpr],
    ) -> PolarsResult<(Vec<Expr>, bool)> {
        if order_by.is_empty() {
            return Ok((Vec::new(), false));
        }
        // Parse expressions and validate uniform direction
        let all_ascending = order_by[0].options.asc.unwrap_or(true);
        let mut exprs = Vec::with_capacity(order_by.len());
        for o in order_by {
            if all_ascending != o.options.asc.unwrap_or(true) {
                // TODO: mixed sort directions are not currently supported; we
                //  need to enhance `over_with_options` to take SortMultipleOptions
                polars_bail!(
                    SQLSyntax:
                    "OVER does not (yet) support mixed asc/desc directions for ORDER BY"
                )
            }
            let expr = parse_sql_expr(&o.expr, self.ctx, self.active_schema)?;
            exprs.push(expr);
        }
        Ok((exprs, !all_ascending))
    }

    fn apply_window_spec(
        &mut self,
        expr: Expr,
        window_type: &Option<WindowType>,
    ) -> PolarsResult<Expr> {
        let Some(window_type) = window_type else {
            return Ok(expr);
        };
        let window_spec = self.resolve_window_spec(window_type)?;
        self.validate_window_frame(&window_spec.window_frame)?;

        let partition_by = if window_spec.partition_by.is_empty() {
            None
        } else {
            Some(
                window_spec
                    .partition_by
                    .iter()
                    .map(|p| parse_sql_expr(p, self.ctx, self.active_schema))
                    .collect::<PolarsResult<Vec<_>>>()?,
            )
        };
        let order_by = if window_spec.order_by.is_empty() {
            None
        } else {
            let (order_exprs, all_desc) = self.parse_order_by_in_window(&window_spec.order_by)?;
            let sort_opts = SortOptions::default().with_order_descending(all_desc);
            Some((order_exprs, sort_opts))
        };

        // Apply window spec
        Ok(match (partition_by, order_by) {
            (None, None) => expr,
            (Some(part), None) => expr.over(part),
            (part, Some(order)) => expr.over_with_options(part, Some(order), Default::default())?,
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
}
