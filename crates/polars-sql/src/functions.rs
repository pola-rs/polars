use chrono::Timelike;
use polars_core::chunked_array::ops::{SortMultipleOptions, SortOptions};
use polars_core::prelude::{
    AnyValue, DataType, ExplodeOptions, PolarsResult, QuantileFractionDtypes, QuantileMethod,
    Scalar, Schema, TimeUnit, TimeZone, UnknownKind, datetime_to_timestamp_us,
    extract_non_null_quantiles, polars_bail, polars_ensure, polars_err,
};
use polars_core::utils::try_get_supertype;
#[cfg(feature = "rank")]
use polars_defs::expr::{RankMethod, RankOptions};
use polars_defs::expr::{RoundMode, UnicodeForm};
use polars_defs::time::date_diff::DateDiffUnit;
use polars_defs::time::duration::Duration;
use polars_defs::time::group_by::ClosedWindow;
use polars_lazy::dsl::Expr;
#[cfg(feature = "approx_quantile")]
use polars_lazy::prelude::ApproxQuantileMethod;
use polars_lazy::prelude::DataTypeExpr;
use polars_plan::dsl::functions::{
    DatetimeArgs, by_name, coalesce, concat_str, date_diff, datetime, datetime_ranges, element,
    format_str, int_range, int_ranges, len, lit, max_horizontal, min_horizontal, when,
};
use polars_plan::dsl::{FunctionExpr, RangeFunction, Selector, SqlBinaryOp, SqlFunction};
use polars_plan::plans::{
    DynLiteralValue, LiteralValue, RowEncodingOutputType, RowEncodingVariant,
};
use polars_plan::prelude::StrptimeOptions;
use polars_utils::pl_str::PlSmallStr;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, DateTimeField, DuplicateTreatment, Expr as SQLExpr,
    Function as SQLFunction, FunctionArg, FunctionArgExpr, FunctionArgumentClause,
    FunctionArgumentList, FunctionArguments, Ident, ObjectName, OrderByExpr, OrderBySort,
    Value as SQLValue, ValueWithSpan, WindowFrame, WindowFrameBound, WindowFrameUnits, WindowSpec,
};
use sqlparser::tokenizer::Span;

use crate::context::is_using_left_key;
use crate::datetime_template::{ParseFormat, format_with_template, parse_format};
use crate::group_context::{is_marked_aggregate, mark_aggregate};
use crate::grouping_sets::MAX_GROUPING_ARGS;
use crate::lambda::{is_lambda_arg, lambda_arg, lambda_element_name};
use crate::literal_folding::fold_scalar;
use crate::sql_expr::{
    adjust_one_indexed_param, approximate_literal, clip_substring_length, decimal_literal,
    decimal_literal_to_f64, interval_to_duration, null_as_string, null_on_null_length,
    null_reading, numeric_length, order_by_sort_options, parse_extract_date_part, parse_sql_array,
    parse_sql_expr, resolve_expr_dtype, sql_binary, sql_position, substring_offset,
    temporal_string_as,
};
use crate::sql_visitors::{contains_random, grouping_call_args};
use crate::types::sql_type_name;
use crate::window_frames::{MAX_ROW_OFFSET, is_frame_aggregate};
use crate::{SQLContext, col};

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
    /// Whether the next value argument parsed is the string input of a string function.
    pub(crate) string_input: bool,
}

/// A parsed value argument. The constant argument of an aggregate is kept apart:
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
    /// SQL 'random' function.
    /// Returns a pseudo-random value in [0, 1) for each row.
    /// ```sql
    /// SELECT RANDOM() FROM df;
    /// ```
    Random,
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
    /// Compute the cosine of the input expression (in radians).
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
    /// Compute the cosine of the input expression (in degrees).
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
    /// SQL 'current_date' function.
    /// Returns the local date at the start of the statement.
    /// ```sql
    /// SELECT CURRENT_DATE;
    /// ```
    CurrentDate,
    /// SQL 'current_time' function (alias: 'localtime').
    /// Returns the local time at the start of the statement.
    /// ```sql
    /// SELECT CURRENT_TIME;
    /// ```
    CurrentTime,
    /// SQL 'current_timestamp' function (aliases: 'localtimestamp', 'now').
    /// Returns the local date and time at the start of the statement, without a time zone.
    /// ```sql
    /// SELECT CURRENT_TIMESTAMP, NOW();
    /// ```
    CurrentTimestamp,
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
    /// SQL 'date_diff' function (alias: 'datediff').
    /// Counts the boundaries of a part, such as 'day' or 'month', crossed from a start
    /// to an end value.
    /// ```sql
    /// SELECT DATE_DIFF('month', col1, col2) FROM df;
    /// ```
    DateDiff,
    /// SQL 'date_trunc' function.
    /// Truncates a date (or datetime) to the given part, such as 'hour' or 'month'.
    /// ```sql
    /// SELECT DATE_TRUNC('month', col1) FROM df;
    /// ```
    DateTrunc,
    /// SQL 'make_date' function.
    /// Builds a date from integer year, month and day parts.
    /// ```sql
    /// SELECT MAKE_DATE(2024, 2, 29) FROM df;
    /// ```
    MakeDate,
    /// SQL 'strftime' function.
    /// Converts a datetime to a string using a format string.
    /// ```sql
    /// SELECT STRFTIME(col1, '%d-%m-%Y %H:%M') FROM df;
    /// ```
    Strftime,
    /// SQL 'to_char' function.
    /// Converts a date, datetime or time to a string using a PostgreSQL template
    /// (or a strftime format string).
    /// ```sql
    /// SELECT TO_CHAR(col1, 'YYYY-MM-DD HH24:MI') FROM df;
    /// ```
    ToChar,
    /// SQL 'to_date' function.
    /// Converts a string to a date using a PostgreSQL template (or a strftime format
    /// string); ISO-8601 format is assumed without one.
    /// ```sql
    /// SELECT TO_DATE(col1, 'DD Mon YYYY') FROM df;
    /// ```
    ToDate,
    /// SQL 'to_timestamp' function.
    /// Converts a string to a datetime using a PostgreSQL template (or a strftime format
    /// string); ISO-8601 format is assumed without one.
    /// ```sql
    /// SELECT TO_TIMESTAMP(col1, 'YYYY-MM-DD HH24:MI:SS') FROM df;
    /// ```
    ToTimestamp,

    // ----
    // String functions
    // ----
    /// SQL 'bit_length' function (bytes).
    /// ```sql
    /// SELECT BIT_LENGTH(col1) FROM df;
    /// ```
    BitLength,
    /// SQL 'chr' function.
    /// Returns the character of an integer Unicode code point.
    /// ```sql
    /// SELECT CHR(65) FROM df;
    /// ```
    Chr,
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
    /// SELECT DATE('2021-15-03', '%Y-%d-%m') FROM df;
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
    /// SQL 'format' function.
    /// Fills the `{}` placeholders of a format string, which can hold a Python format
    /// specification such as `{:.2f}`.
    /// ```sql
    /// SELECT FORMAT('{} costs {:.2f}', col1, col2) FROM df;
    /// ```
    Format,
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
    /// SQL 'printf' function.
    /// Fills the `%` conversions of a format string, as in C's printf.
    /// ```sql
    /// SELECT PRINTF('%s costs %.2f', col1, col2) FROM df;
    /// ```
    Printf,
    /// SQL 'json_extract_path' function, as in Postgres.
    /// The value at a path of keys or array indexes, as `#>` reads it.
    /// ```sql
    /// SELECT JSON_EXTRACT_PATH(col1, 'user', 'name') FROM df;
    /// ```
    JsonExtractPath,
    /// SQL 'json_extract_path_text' function, as in Postgres.
    /// The value at a path of keys or array indexes as text, as `#>>` reads it.
    /// ```sql
    /// SELECT JSON_EXTRACT_PATH_TEXT(col1, 'user', 'name') FROM df;
    /// ```
    JsonExtractPathText,
    /// SQL 'regexp_extract' function.
    /// Returns the given capture group of the first match of `pattern` (optional: `group`,
    /// 0 for the whole match), or an empty string if there is no match.
    /// ```sql
    /// SELECT REGEXP_EXTRACT(col1, '(\d+)-(\d+)', 2) FROM df;
    /// ```
    RegexpExtract,
    /// SQL 'regexp_substr' function, as in Postgres.
    /// The `n`th match of `pattern`, or its capture group `subexpr`; NULL if there is none.
    /// ```sql
    /// SELECT REGEXP_SUBSTR(col1, 'user=(\w+)', 1, 1, 'i', 1) FROM df;
    /// ```
    RegexpSubstr,
    /// SQL 'regexp_like' function.
    /// True if `pattern` matches the value (optional: `flags`).
    /// ```sql
    /// SELECT REGEXP_LIKE(col1, 'xyz', 'i') FROM df;
    /// ```
    RegexpLike,
    /// SQL 'regexp_matches' function.
    /// True if `pattern` matches the value (optional: `options`).
    /// ```sql
    /// SELECT REGEXP_MATCHES(col1, 'xyz', 'i') FROM df;
    /// ```
    RegexpMatches,
    /// SQL 'regexp_replace' function.
    /// Replaces the first match of `pattern` with `replacement` (optional: `flags`;
    /// the 'g' flag replaces all matches). `\1` refers to the first capture group.
    /// ```sql
    /// SELECT REGEXP_REPLACE(col1, '(\d+)-(\d+)', '\2-\1', 'g') FROM df;
    /// ```
    RegexpReplace,
    /// SQL 'replace' function.
    /// Replace a given substring with another string.
    /// ```sql
    /// SELECT REPLACE(col1, 'old', 'new') FROM df;
    /// ```
    Replace,
    /// SQL 'repeat' function.
    /// Returns the string repeated `n` times.
    /// ```sql
    /// SELECT REPEAT(col1, 3) FROM df;
    /// ```
    Repeat,
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
    /// SQL 'try_strptime' function.
    /// Like 'strptime', but gives null for a string that does not match the format.
    /// ```sql
    /// SELECT TRY_STRPTIME(col1, '%d-%m-%Y %H:%M') FROM df;
    /// ```
    TryStrptime,
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
    /// SELECT TIMESTAMP('2021-15-03T00:01:02.333', '%Y-%d-%m %H:%M:%S') FROM df;
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
    /// SQL 'approx_count_distinct' function.
    /// Returns an approximation of the number of distinct non-null elements in the grouping.
    /// ```sql
    /// SELECT APPROX_COUNT_DISTINCT(col1) FROM df;
    /// ```
    #[cfg(feature = "approx_unique")]
    ApproxCountDistinct,
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
    /// SQL 'any_value' function.
    /// Returns a non-null element of the grouping, if any.
    /// ```sql
    /// SELECT ANY_VALUE(col1) FROM df;
    /// ```
    ArbitraryValue,
    /// SQL 'avg' function.
    /// Returns the average (mean) of all the elements in the grouping.
    /// ```sql
    /// SELECT AVG(col1) FROM df;
    /// ```
    Avg,
    /// SQL 'bool_and' function.
    /// Returns true if all non-null elements of the grouping are true.
    /// ```sql
    /// SELECT BOOL_AND(col1) FROM df;
    /// ```
    BoolAnd,
    /// SQL 'bool_or' function.
    /// Returns true if any non-null element of the grouping is true.
    /// ```sql
    /// SELECT BOOL_OR(col1) FROM df;
    /// ```
    BoolOr,
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
    /// SQL 'count_if' function.
    /// Returns the number of elements in the grouping for which the condition is true.
    /// ```sql
    /// SELECT COUNT_IF(col1 > 0) FROM df;
    /// ```
    CountIf,
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
    /// SQL 'max_by' function.
    /// Returns the value of the first expression at the row where the second is greatest.
    /// ```sql
    /// SELECT MAX_BY(col1, col2) FROM df;
    /// ```
    MaxBy,
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
    /// SQL 'percentile_cont' function.
    /// Returns the continuous percentile of the values in the WITHIN GROUP order
    /// (interpolated value between two closest values).
    /// ```sql
    /// SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY col1) FROM df;
    /// ```
    PercentileCont,
    /// SQL 'percentile_disc' function.
    /// Returns the first value in the WITHIN GROUP order whose position reaches the percentile.
    /// ```sql
    /// SELECT PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY col1) FROM df;
    /// ```
    PercentileDisc,
    /// SQL 'min' function.
    /// Returns the smallest (minimum) of all the elements in the grouping.
    /// ```sql
    /// SELECT MIN(col1) FROM df;
    /// ```
    Min,
    /// SQL 'min_by' function.
    /// Returns the value of the first expression at the row where the second is smallest.
    /// ```sql
    /// SELECT MIN_BY(col1, col2) FROM df;
    /// ```
    MinBy,
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
    /// Returns the length of the array (optional: the dimension, which must be 1).
    /// ```sql
    /// SELECT ARRAY_LENGTH(col1) FROM df;
    /// ```
    ArrayLength,
    /// SQL 'cardinality' function.
    /// Returns the number of elements in the array.
    /// ```sql
    /// SELECT CARDINALITY(col1) FROM df;
    /// ```
    Cardinality,
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
    /// SQL 'list_transform' function.
    /// Returns the list with a lambda applied to each element.
    /// ```sql
    /// SELECT LIST_TRANSFORM(col1, lambda x: x + 1) FROM df;
    /// ```
    ListTransform,
    /// SQL 'list_filter' function.
    /// Returns the list with the elements for which a lambda is true.
    /// ```sql
    /// SELECT LIST_FILTER(col1, lambda x: x > 1) FROM df;
    /// ```
    ListFilter,
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
    /// SQL 'generate_series' function.
    /// Returns a list from start to stop, stop included. Dates and timestamps take an INTERVAL
    /// step and return timestamps.
    /// ```sql
    /// SELECT GENERATE_SERIES(1, 10, 2) FROM df;
    /// SELECT UNNEST(GENERATE_SERIES(DATE '2026-01-01', DATE '2026-01-31', INTERVAL '1 day'));
    /// ```
    GenerateSeries,
    /// SQL 'range' function.
    /// Like `generate_series`, but stop is not included.
    /// ```sql
    /// SELECT RANGE(0, 10, 2) FROM df;
    /// ```
    Range,

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
    // Type functions
    // ----
    /// SQL 'typeof' function.
    /// Returns the SQL name of the data type of the argument, as `DESCRIBE` gives it.
    /// ```sql
    /// SELECT TYPEOF(col1) FROM df;
    /// ```
    Typeof,

    // ----
    // Column selection
    // ----
    Columns,

    // ----
    // User-defined
    // ----
    Udf(String),
}

/// A literal key of `JSON_EXTRACT_PATH` as an element of a Postgres text array.
fn json_path_element(key: &FunctionArgExpr) -> Option<String> {
    let FunctionArgExpr::Expr(SQLExpr::Value(ValueWithSpan { value, .. })) = key else {
        return None;
    };
    Some(match value {
        // quoted only when needed: a struct's path does not read quotes
        SQLValue::SingleQuotedString(s) if !needs_quotes(s) => s.clone(),
        SQLValue::SingleQuotedString(s) => {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        },
        SQLValue::Number(n, _) => n.to_string(),
        SQLValue::Null => "NULL".to_string(),
        _ => return None,
    })
}

/// Whether a Postgres text array element must be quoted to be read as written.
fn needs_quotes(key: &str) -> bool {
    key.is_empty()
        || key.eq_ignore_ascii_case("NULL")
        || key.contains([',', '{', '}', '"', '\\'])
        || key.contains(char::is_whitespace)
}

/// How `UNNEST` and `EXPLODE` explode a list: as in Postgres, an empty or NULL list gives no row.
pub(crate) const UNNEST_OPTIONS: ExplodeOptions = ExplodeOptions {
    empty_as_null: false,
    keep_nulls: false,
};

/// Whether `name` is a SQL function or a registered user-defined function.
pub(crate) fn is_builtin_function(name: &Ident, ctx: &SQLContext) -> bool {
    let function = SQLFunction {
        name: ObjectName::from(vec![name.clone()]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::None,
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    };
    PolarsSQLFunctions::try_from_sql(&function, ctx).is_ok()
}

impl PolarsSQLFunctions {
    /// Whether the first argument is a string, which a NULL literal is read as.
    fn takes_string_input(&self) -> bool {
        use PolarsSQLFunctions::*;
        match self {
            #[cfg(feature = "nightly")]
            InitCap => true,
            BitLength | Date | EndsWith | Left | LeftPad | LeftTrim | Length | Lower
            | Normalize | OctetLength | RegexpExtract | RegexpLike | RegexpMatches
            | RegexpSubstr | RegexpReplace | Replace | Repeat | Reverse | Right | RightPad
            | RightTrim | SplitPart | StartsWith | StrPos | Substring | StringToArray
            | Strptime | TryStrptime | Time | Timestamp | Upper => true,
            _ => false,
        }
    }

    pub(crate) fn keywords() -> &'static [&'static str] {
        &[
            "abs",
            "acos",
            "acosd",
            "any_value",
            "apply",
            "approx_count_distinct",
            "approx_quantile",
            "arg_max",
            "arg_min",
            "array_agg",
            "array_apply",
            "array_contains",
            "array_dot_product",
            "array_filter",
            "array_get",
            "array_inner_product",
            "array_length",
            "array_lower",
            "array_mean",
            "array_reverse",
            "array_sum",
            "array_to_string",
            "array_transform",
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
            "bit_not",
            "bit_or",
            "bit_xor",
            "bitand",
            "bitcount",
            "bitnot",
            "bitor",
            "bitxor",
            "bool_and",
            "bool_or",
            "cardinality",
            "cbrt",
            "ceil",
            "ceiling",
            "char_length",
            "character_length",
            "chr",
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
            "count_if",
            "countif",
            "covar",
            "covar_pop",
            "covar_samp",
            "cume_dist",
            "current_date",
            "current_time",
            "current_timestamp",
            "date",
            "date_diff",
            "date_part",
            "date_trunc",
            "datediff",
            "datetime",
            "day",
            "dayofmonth",
            "dayofweek",
            "dayofyear",
            "degrees",
            "dense_rank",
            "div",
            "ends_with",
            "erf",
            "erfc",
            "every",
            "exp",
            "filter",
            "first",
            "first_value",
            "floor",
            "format",
            "generate_series",
            "greatest",
            "group_concat",
            "grouping",
            "grouping_id",
            "hour",
            "if",
            "ifnull",
            "initcap",
            "json_extract_path",
            "json_extract_path_text",
            "lag",
            "last",
            "last_value",
            "lead",
            "least",
            "left",
            "length",
            "list_apply",
            "list_filter",
            "list_transform",
            "listagg",
            "ln",
            "localtime",
            "localtimestamp",
            "log",
            "log10",
            "log1p",
            "log2",
            "lower",
            "lpad",
            "ltrim",
            "make_date",
            "max",
            "max_by",
            "median",
            "min",
            "min_by",
            "minute",
            "mod",
            "month",
            "normalize",
            "now",
            "nth_value",
            "ntile",
            "nullif",
            "octet_length",
            "percent_rank",
            "percentile_cont",
            "percentile_disc",
            "pi",
            "pow",
            "power",
            "printf",
            "quantile_cont",
            "quantile_disc",
            "quarter",
            "radians",
            "random",
            "range",
            "rank",
            "regexp_extract",
            "regexp_like",
            "regexp_matches",
            "regexp_replace",
            "regexp_substr",
            "repeat",
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
            "split_part",
            "sqrt",
            "starts_with",
            "stddev",
            "stddev_samp",
            "stdev",
            "stdev_samp",
            "strftime",
            "string_agg",
            "string_to_array",
            "strpos",
            "strptime",
            "substr",
            "sum",
            "tan",
            "tand",
            "time",
            "timestamp",
            "to_char",
            "to_date",
            "to_timestamp",
            "total",
            "trunc",
            "truncate",
            "try_strptime",
            "typeof",
            "unnest",
            "upper",
            "var",
            "var_samp",
            "variance",
            "week",
            "xor",
            "year",
        ]
    }
}

impl PolarsSQLFunctions {
    pub(crate) fn try_from_sql(
        function: &'_ SQLFunction,
        ctx: &'_ SQLContext,
    ) -> PolarsResult<Self> {
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
            "random" => Self::Random,
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
            "current_date" => Self::CurrentDate,
            "current_time" | "localtime" => Self::CurrentTime,
            "current_timestamp" | "localtimestamp" | "now" => Self::CurrentTimestamp,
            "date" => Self::Date,
            "date_diff" | "datediff" => Self::DateDiff,
            "date_part" => Self::DatePart,
            "date_trunc" => Self::DateTrunc,
            "make_date" => Self::MakeDate,
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
            "to_char" => Self::ToChar,
            "to_date" => Self::ToDate,
            "to_timestamp" => Self::ToTimestamp,

            // ----
            // String functions
            // ----
            "bit_length" => Self::BitLength,
            "chr" => Self::Chr,
            "concat" => Self::Concat,
            "concat_ws" => Self::ConcatWS,
            "ends_with" => Self::EndsWith,
            "format" => Self::Format,
            #[cfg(feature = "nightly")]
            "initcap" => Self::InitCap,
            "left" => Self::Left,
            "length" | "char_length" | "character_length" => Self::Length,
            "lower" => Self::Lower,
            "lpad" => Self::LeftPad,
            "ltrim" => Self::LeftTrim,
            "normalize" => Self::Normalize,
            "octet_length" => Self::OctetLength,
            "printf" => Self::Printf,
            "regexp_extract" => Self::RegexpExtract,
            "regexp_like" => Self::RegexpLike,
            "regexp_matches" => Self::RegexpMatches,
            "regexp_replace" => Self::RegexpReplace,
            "regexp_substr" => Self::RegexpSubstr,
            "json_extract_path" => Self::JsonExtractPath,
            "json_extract_path_text" => Self::JsonExtractPathText,
            "repeat" => Self::Repeat,
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
            "try_strptime" => Self::TryStrptime,
            "substr" => Self::Substring,
            "time" => Self::Time,
            "upper" => Self::Upper,

            // ----
            // Aggregate functions
            // ----
            "any_value" => Self::ArbitraryValue,
            #[cfg(feature = "approx_unique")]
            "approx_count_distinct" => Self::ApproxCountDistinct,
            #[cfg(feature = "approx_quantile")]
            "approx_quantile" => Self::ApproxQuantile,
            "avg" => Self::Avg,
            "bool_and" | "every" => Self::BoolAnd,
            "bool_or" => Self::BoolOr,
            "corr" => Self::Corr,
            "count" => Self::Count,
            "count_if" | "countif" => Self::CountIf,
            "covar_pop" => Self::CovarPop,
            "covar_samp" | "covar" => Self::CovarSamp,
            "first" => Self::First,
            "grouping" => Self::Grouping,
            "grouping_id" => Self::GroupingId,
            "last" => Self::Last,
            "max" => Self::Max,
            "max_by" | "arg_max" => Self::MaxBy,
            "median" => Self::Median,
            "min" => Self::Min,
            "min_by" | "arg_min" => Self::MinBy,
            "quantile_cont" => Self::QuantileCont,
            "quantile_disc" => Self::QuantileDisc,
            "percentile_cont" => Self::PercentileCont,
            "percentile_disc" => Self::PercentileDisc,
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
            "cardinality" => Self::Cardinality,
            "array_lower" => Self::ArrayMin,
            "array_mean" => Self::ArrayMean,
            "array_reverse" => Self::ArrayReverse,
            "array_sum" => Self::ArraySum,
            "array_to_string" => Self::ArrayToString,
            "array_unique" => Self::ArrayUnique,
            "list_transform" | "array_transform" | "apply" | "list_apply" | "array_apply" => {
                Self::ListTransform
            },
            "list_filter" | "array_filter" | "filter" => Self::ListFilter,
            "array_upper" => Self::ArrayMax,
            "unnest" => Self::Explode,
            "generate_series" => Self::GenerateSeries,
            "range" => Self::Range,

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
            // Type functions
            // ----
            "typeof" => Self::Typeof,

            // ----
            // Column selection
            // ----
            "columns" => Self::Columns,

            other => {
                if ctx.function_registry.contains(other) {
                    Self::Udf(other.to_string())
                } else {
                    let error = polars_err!(SQLInterface: "unsupported function '{}'", other);
                    return Err(match spelling_of(other) {
                        Some(spelling) => error.with_hint(format!("use {spelling} instead")),
                        None => error,
                    });
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
    pub(crate) fn is_builtin_aggregate(&self) -> bool {
        use PolarsSQLFunctions::*;
        match self {
            #[cfg(feature = "approx_quantile")]
            ApproxQuantile => true,
            #[cfg(feature = "approx_unique")]
            ApproxCountDistinct => true,
            ArbitraryValue | ArrayAgg | Avg | BoolAnd | BoolOr | Corr | Count | CountIf
            | CovarPop | CovarSamp | First | Last | Max | MaxBy | Median | Min | MinBy
            | PercentileCont | PercentileDisc | QuantileCont | QuantileDisc | StdDev
            | StringAgg | Sum | Total | Variance => true,
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
        if !function.within_group.is_empty()
            && !matches!(function_name, PercentileCont | PercentileDisc)
        {
            polars_bail!(SQLInterface: "'WITHIN GROUP' is not currently supported")
        }
        if function.null_treatment.is_some() {
            polars_bail!(SQLInterface: "'IGNORE|RESPECT NULLS' is not currently supported")
        }
        if let Some(filter_expr) = &function.filter {
            let pred = parse_sql_expr(filter_expr, self.ctx, self.active_schema)?;
            // The predicate is read once per row.
            self.read_subqueries_per_row(&pred);
            let typed_pred = self.ctx.with_typed_subqueries(&pred)?;
            // As in WHERE, a condition that reads no input is accepted as any type that casts
            // to boolean.
            let pred = if reads_no_input(&typed_pred) {
                pred.cast(DataType::Boolean)
            } else {
                // The rows that pass are counted, which needs a boolean.
                let dtype = self
                    .active_schema
                    .and_then(|schema| Some(typed_pred.to_field(schema).ok()?.dtype))
                    .filter(|dtype| dtype.is_known());
                match dtype {
                    Some(dtype) => {
                        polars_ensure!(
                            dtype.is_bool(),
                            InvalidOperation: "filter predicate must be of type `Boolean`, got `{}`", dtype
                        );
                        pred
                    },
                    // Otherwise `when` checks for a boolean when it runs.
                    None => when(pred).then(lit(true)).otherwise(lit(false)),
                }
            };
            self.filter = Some(pred);
        }
        self.reads_rows = function_name.is_builtin_aggregate();
        self.string_input = function_name.takes_string_input();
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
            BitCount => self.visit_unary(|e| e.bitwise_count_ones().cast(DataType::Int64)),
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
            Random => self.visit_nullary(random),
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
            Coalesce => self.visit_temporal_variadic(|exprs: &[Expr]| Ok(coalesce(exprs))),
            Greatest => self.visit_temporal_variadic(|exprs: &[Expr]| max_horizontal(exprs)),
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
                    2 => self.visit_temporal_variadic(|exprs: &[Expr]| Ok(coalesce(exprs))),
                    _ => {
                        polars_bail!(SQLSyntax: "IFNULL expects 2 arguments (found {})", args.len())
                    },
                }
            },
            Least => self.visit_temporal_variadic(|exprs: &[Expr]| min_horizontal(exprs)),
            NullIf => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_temporal_variadic(|exprs: &[Expr]| {
                        let [l, r] = exprs else { unreachable!() };
                        Ok(when(l.clone().eq(r.clone()))
                            .then(lit(LiteralValue::untyped_null()))
                            .otherwise(l.clone()))
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
            CurrentDate => {
                let date = self.ctx.statement_time().date().to_epoch_days();
                self.visit_nullary(|| lit(Scalar::new(DataType::Date, AnyValue::Date(date))))
            },
            CurrentTime => {
                let time = self.ctx.statement_time().time();
                let time = time.num_seconds_from_midnight() as i64 * 1_000_000_000
                    + time.nanosecond() as i64;
                self.visit_nullary(|| lit(Scalar::new(DataType::Time, AnyValue::Time(time))))
            },
            CurrentTimestamp => {
                let timestamp = datetime_to_timestamp_us(self.ctx.statement_time());
                let dtype = DataType::Datetime(TimeUnit::Microseconds, None);
                let value = AnyValue::Datetime(timestamp, TimeUnit::Microseconds, None);
                self.visit_nullary(|| lit(Scalar::new(dtype.clone(), value.clone())))
            },
            DateDiff => self.visit_date_diff(),
            DateTrunc => self.visit_date_trunc(),
            MakeDate => {
                let args = extract_args(function)?;
                polars_ensure!(
                    args.len() == 3,
                    SQLSyntax: "MAKE_DATE expects 3 arguments (found {})", args.len()
                );
                self.visit_variadic(|e| {
                    let args = DatetimeArgs::new(e[0].clone(), e[1].clone(), e[2].clone());
                    datetime(args).cast(DataType::Date)
                })
            },
            Strftime => {
                let args = extract_args(function)?;
                match args.len() {
                    2 => self.visit_binary(|e, fmt: String| e.dt().strftime(fmt.as_str())),
                    _ => {
                        polars_bail!(SQLSyntax: "STRFTIME expects 2 arguments (found {})", args.len())
                    },
                }
            },
            ToChar => self.visit_to_char(),
            ToDate => {
                self.visit_parse_template("TO_DATE", false, |e, options| e.str().to_date(options))
            },
            ToTimestamp => self.visit_parse_template("TO_TIMESTAMP", true, |e, options| {
                e.str().to_datetime(
                    Some(TimeUnit::Microseconds),
                    None,
                    options,
                    lit("latest"),
                )
            }),

            // ----
            // String functions
            // ----
            BitLength => self.visit_unary(|e| e.str().len_bytes().cast(DataType::Int64) * lit(8)),
            Chr => self.visit_unary(|e| e.str().from_code_point()),
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
            Format => self.visit_format(false),
            #[cfg(feature = "nightly")]
            InitCap => self.visit_unary(|e| e.str().to_titlecase()),
            Left => self.try_visit_binary(|e, length| {
                Ok(match length {
                    Expr::Literal(lv) if lv.is_null() => null_reading(Some(DataType::String), &[e]),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0))) => {
                        when(e.is_not_null())
                            .then(lit(""))
                            .otherwise(lit(LiteralValue::untyped_null()).cast(DataType::String))
                    },
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
                    _ => {
                        let raw_length = length;
                        let length = numeric_length(raw_length.clone());
                        let left = when(length.clone().gt_eq(lit(0)))
                            .then(e.clone().str().slice(lit(0), length.clone().abs()))
                            .otherwise(e.clone().str().slice(
                                lit(0),
                                (e.str().len_chars() + length.clone()).clip_min(lit(0)),
                            ));
                        null_on_null_length(left, &raw_length)
                    },
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
            Length => self.visit_unary(|e| e.str().len_chars().cast(DataType::Int64)),
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
            OctetLength => self.visit_unary(|e| e.str().len_bytes().cast(DataType::Int64)),
            Printf => self.visit_format(true),
            StrPos => self.visit_binary(sql_position),
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
            RegexpExtract => self.visit_regexp_extract(),
            RegexpSubstr => self.visit_regexp_substr(),
            JsonExtractPath => self.visit_json_extract_path(false),
            JsonExtractPathText => self.visit_json_extract_path(true),
            RegexpMatches => self.visit_regexp_matches(),
            RegexpReplace => self.try_visit_variadic(|args| match args {
                [e, pat, replacement] => regexp_replace(e, pat, replacement, None),
                [e, pat, replacement, flags] => regexp_replace(e, pat, replacement, Some(flags)),
                _ => polars_bail!(SQLSyntax: "REGEXP_REPLACE expects 3-4 arguments (found {})", args.len()),
            }),
            Replace => {
                let args = extract_args(function)?;
                match args.len() {
                    3 => self.try_visit_ternary(|e, old: Expr, new: Expr| {
                        let is_null = |e: &Expr| matches!(e, Expr::Literal(lv) if lv.is_null());
                        Ok(if is_null(&old) || is_null(&new) {
                            // NULL, still reading every argument, so that an aggregate in
                            // one stays one.
                            let read = |e: Expr| e.cast(DataType::String);
                            when(lit(false))
                                .then(read(e))
                                .when(lit(false))
                                .then(read(old))
                                .when(lit(false))
                                .then(read(new))
                                .otherwise(lit(LiteralValue::untyped_null()).cast(DataType::String))
                        } else if matches!(&old, Expr::Literal(lv) if lv.extract_str() == Some("")) {
                            // As in Postgres; the folded branch keeps the string type check.
                            let replaced = e.clone().str().replace_all(old, new.clone(), true);
                            let kept = when(lit(true)).then(e).otherwise(replaced);
                            match new {
                                Expr::Literal(_) => kept,
                                new => when(new.is_not_null())
                                    .then(kept)
                                    .otherwise(lit(LiteralValue::untyped_null())),
                            }
                        } else if matches!((&old, &new), (Expr::Literal(_), Expr::Literal(_))) {
                            e.str().replace_all(old, new, true)
                        } else {
                            // A NULL pattern or replacement gives NULL. The string is made NULL
                            // first, and the filled-in pattern is not empty, so that it never
                            // matches the empty values under those NULLs.
                            let has_args = [&old, &new]
                                .into_iter()
                                .filter(|e| !matches!(e, Expr::Literal(_)))
                                .map(|e| e.clone().is_not_null())
                                .reduce(Expr::and)
                                .unwrap();
                            let e = when(has_args)
                                .then(e)
                                .otherwise(lit(LiteralValue::untyped_null()));
                            let pattern_is_literal = matches!(old, Expr::Literal(_));
                            let old = old.cast(DataType::String);
                            let replaced = e.clone().str().replace_all(
                                old.clone().fill_null(lit("_")),
                                new.cast(DataType::String).fill_null(lit("")),
                                true,
                            );
                            match pattern_is_literal {
                                true => replaced,
                                false => when(old.str().len_bytes().eq(lit(0)))
                                    .then(e)
                                    .otherwise(replaced),
                            }
                        })
                    }),
                    _ => {
                        polars_bail!(SQLSyntax: "REPLACE expects 3 arguments (found {})", args.len())
                    },
                }
            },
            Repeat => self.visit_binary(|e: Expr, n: Expr| {
                let e = e.cast(DataType::String);
                let n = n.strict_cast(DataType::Int64).clip_min(lit(0));
                // An empty or null string repeats to itself, so it needs no repetitions.
                let n = when(e.clone().str().len_bytes().gt(lit(0)))
                    .then(n.clone())
                    .otherwise(n.clip_max(lit(0)));
                let repeated = e.clone().repeat_by(n).list().join(lit(""), false);
                when(e.is_null())
                    .then(Expr::Literal(LiteralValue::untyped_null()))
                    .otherwise(repeated)
            }),
            Reverse => self.visit_unary(|e| e.str().reverse()),
            Right => self.try_visit_binary(|e, length| {
                Ok(match length {
                    Expr::Literal(lv) if lv.is_null() => null_reading(Some(DataType::String), &[e]),
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0))) => {
                        when(e.is_null())
                            .then(lit(LiteralValue::untyped_null()).cast(DataType::String))
                            .otherwise(lit(""))
                    },
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(_))) => {
                        e.str().tail(length)
                    },
                    Expr::Literal(v) => {
                        polars_bail!(SQLSyntax: "invalid 'n_chars' for RIGHT ({:?})", v)
                    },
                    // A negative length takes all but the first characters.
                    _ => e.str().tail(length),
                })
            }),
            SplitPart => {
                let args = extract_args(function)?;
                match args.len() {
                    3 => self.try_visit_ternary(|e, sep: Expr, idx: Expr| {
                        // A NULL argument gives NULL; an index past the parts gives ''.
                        let has_args = e
                            .clone()
                            .is_not_null()
                            .and(sep.clone().is_not_null())
                            .and(idx.clone().is_not_null());
                        let idx = adjust_one_indexed_param(idx, true);
                        Ok(when(has_args)
                            .then(
                                e.cast(DataType::String)
                                    .str()
                                    .split(sep.cast(DataType::String))
                                    .list()
                                    .get(idx, true)
                                    .fill_null(lit("")),
                            )
                            .otherwise(lit(LiteralValue::untyped_null())))
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
            Strptime => self.visit_strptime("STRPTIME", true),
            TryStrptime => self.visit_strptime("TRY_STRPTIME", false),
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
                            Expr::Literal(lv) if lv.is_null() => null_reading(Some(DataType::String), &[e]),
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n <= 0 => e,
                            Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => e.str().slice(lit(n - 1), lit(LiteralValue::untyped_null())),
                            Expr::Literal(_) => polars_bail!(SQLSyntax: "invalid 'start' for SUBSTR ({})", args[1]),
                            _ => e.str().slice(substring_offset(start).clip_min(lit(0)), lit(LiteralValue::untyped_null())),
                        })
                    }),
                    3 => self.try_visit_ternary(|e: Expr, start: Expr, length: Expr| {
                        let raw_length = approximate_literal(length);
                        let length = clip_substring_length(raw_length.clone());
                        let result = match (start.clone(), length.clone()) {
                            (Expr::Literal(lv), _) | (_, Expr::Literal(lv)) if lv.is_null() => {
                                null_reading(Some(DataType::String), &[e, start, length])
                            },
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
                                let adjusted_start = substring_offset(start);
                                when(adjusted_start.clone().lt(lit(0)))
                                    .then(e.clone().str().slice(lit(0), (length.clone() + adjusted_start.clone()).clip_min(lit(0))))
                                    .otherwise(e.str().slice(adjusted_start, length))
                            }
                        };
                        Ok(null_on_null_length(result, &raw_length))
                    }),
                    _ => polars_bail!(SQLSyntax: "SUBSTR expects 2-3 arguments (found {})", args.len()),
                }
            },
            Upper => self.visit_unary(|e| e.str().to_uppercase()),

            // ----
            // Aggregate functions
            // ----
            ArbitraryValue => self.visit_unary_aggregate(Expr::first_non_null),
            #[cfg(feature = "approx_unique")]
            ApproxCountDistinct => self.visit_approx_count_distinct(),
            #[cfg(feature = "approx_quantile")]
            ApproxQuantile => self.visit_approx_quantile(),
            Avg => self.visit_avg(),
            BoolAnd => self.visit_bool_aggregate(Expr::min),
            BoolOr => self.visit_bool_aggregate(Expr::max),
            Corr => self.visit_binary(sql_corr),
            Count => self.visit_count(),
            CountIf => self.visit_count_if(),
            CovarPop => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 0)),
            CovarSamp => self.visit_binary(|a, b| polars_lazy::dsl::cov(a, b, 1)),
            First => self.visit_unary_aggregate(Expr::first),
            Grouping | GroupingId => self.visit_grouping(),
            Last => self.visit_unary_aggregate(Expr::last),
            Max => self.visit_min_max(Expr::max),
            MaxBy => self.visit_binary(Expr::max_by),
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
            MinBy => self.visit_binary(Expr::min_by),
            PercentileCont | PercentileDisc => self.visit_percentile(&function_name),
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
            ArrayLength => {
                let args = extract_args(function)?;
                let schema = self.active_schema;
                match args.len() {
                    1 => self.visit_unary(|e| list_len(e, schema)),
                    2 => self.try_visit_binary(|e, dimension| match dimension {
                        Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(1))) => {
                            Ok(list_len(e, schema))
                        },
                        _ => polars_bail!(SQLInterface: "ARRAY_LENGTH only supports dimension 1 (found {})", args[1]),
                    }),
                    _ => polars_bail!(SQLSyntax: "ARRAY_LENGTH expects 1-2 arguments (found {})", args.len()),
                }
            },
            Cardinality => {
                let schema = self.active_schema;
                self.visit_unary(|e| list_len(e, schema))
            },
            ArrayMax => self.visit_unary(|e| e.list().max()),
            ArrayMean => self.visit_unary(|e| e.list().mean()),
            ArrayMin => self.visit_unary(|e| e.list().min()),
            ArrayReverse => self.visit_unary(|e| e.list().eval(element().reverse())),
            ArraySum => self.visit_unary(|e| e.list().sum()),
            ArrayToString => self.visit_arr_to_string(),
            ArrayUnique => self.visit_unary(|e| e.list().eval(element().unique_stable())),
            ListTransform => self.visit_list_lambda(false),
            ListFilter => self.visit_list_lambda(true),
            Explode => self.visit_unary(|e| {
                e.explode(UNNEST_OPTIONS)
            }),
            GenerateSeries => self.visit_series(true),
            Range => self.visit_series(false),

            // ----
            // Type functions
            // ----
            Typeof => self.visit_typeof(),

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
                                .filter(|name| rx.is_match(name) && !is_using_left_key(name))
                                .cloned()
                                .collect::<Vec<_>>();

                            Ok(if col_names.len() == 1 {
                                col(col_names.into_iter().next().unwrap())
                            } else {
                                by_name(col_names, true, false).as_expr()
                            })
                        } else {
                            Ok(Expr::Selector(Selector::Matches(pat.into())))
                        }
                    },
                    Expr::Selector(s) => {
                        let hidden: Vec<PlSmallStr> = active_schema
                            .iter()
                            .flat_map(|schema| schema.iter_names())
                            .filter(|name| is_using_left_key(name))
                            .cloned()
                            .collect();
                        Ok(match hidden.is_empty() {
                            true => s.as_expr(),
                            false => s.exclude_cols(hidden).as_expr(),
                        })
                    },
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
            #[cfg(feature = "approx_unique")]
            ApproxCountDistinct => true,
            ArbitraryValue | ArrayAgg | Corr | CovarPop | CovarSamp | Last | MaxBy | Median
            | MinBy | PercentileCont | PercentileDisc | QuantileCont | QuantileDisc | StdDev
            | StringAgg | Variance => true,
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

    /// `DATE_DIFF(part, start, end)`.
    fn visit_date_diff(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [
            FunctionArgExpr::Expr(part),
            FunctionArgExpr::Expr(start),
            FunctionArgExpr::Expr(end),
        ] = args.as_slice()
        else {
            return self.not_supported_error();
        };
        let unit = match self.parse_sql_arg(part)? {
            Expr::Literal(p) if p.is_null() => None,
            Expr::Literal(ref p) if let Some(p) = p.extract_str() => {
                Some(date_diff_unit(&p.to_ascii_lowercase()).ok_or_else(
                    || polars_err!(SQLSyntax: "DATE_DIFF does not support '{}' part", p),
                )?)
            },
            part => polars_bail!(SQLSyntax: "invalid 'part' for DATE_DIFF ({})", part),
        };
        let start = self.parse_sql_arg(start)?;
        let end = self.parse_sql_arg(end)?;
        let Some(unit) = unit else {
            return self.apply_window_spec(null_reading(Some(DataType::Int64), &[start, end]));
        };
        // A string literal is read as a value of the type of the other argument.
        let is_string_literal =
            |e: &Expr| matches!(e, Expr::Literal(lv) if lv.extract_str().is_some());
        let (start, end) = match (is_string_literal(&start), is_string_literal(&end)) {
            (true, false) => (
                temporal_string_as(&start, self.arg_dtype(&end)?.as_ref()),
                end,
            ),
            (false, true) => {
                let start_dtype = self.arg_dtype(&start)?;
                (start, temporal_string_as(&end, start_dtype.as_ref()))
            },
            _ => (start, end),
        };
        self.apply_window_spec(date_diff(start, end, unit))
    }

    /// FORMAT with `{}` placeholders, or PRINTF with `%` conversions.
    fn visit_format(&mut self, printf: bool) -> PolarsResult<Expr> {
        let name = if printf { "PRINTF" } else { "FORMAT" };
        let mut args = self.parse_variadic_args()?;
        polars_ensure!(!args.is_empty(), SQLSyntax: "{name} expects at least 1 argument (found 0)");
        let mut used = vec![false; args.len()];
        let (mut format, mut inputs) = match args.remove(0) {
            Expr::Literal(lv) if lv.is_null() => (
                "{0}".to_string(),
                vec![lit(LiteralValue::untyped_null()).cast(DataType::String)],
            ),
            Expr::Literal(lv) if lv.extract_str().is_some() => {
                let format = lv.extract_str().unwrap();
                if printf {
                    printf_to_format(format, &args, &mut used)?
                } else {
                    sql_format_to_format(format, &args, &mut used)?
                }
            },
            format => {
                polars_bail!(SQLSyntax: "{name} format must be a string literal (found {format})")
            },
        };
        // Unused arguments are written as empty strings, so that a null still gives a null and
        // aggregates are kept.
        for (arg, used) in args.into_iter().zip(used) {
            if !used {
                format.push_str(&format!("{{{}}}", inputs.len()));
                inputs.push(
                    when(arg.is_not_null())
                        .then(lit(""))
                        .otherwise(lit(LiteralValue::untyped_null())),
                );
            }
        }
        self.apply_window_spec(format_str(&format, &inputs)?)
    }

    fn visit_date_trunc(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [FunctionArgExpr::Expr(part), FunctionArgExpr::Expr(e)] = args.as_slice() else {
            return self.not_supported_error();
        };
        let part = self.parse_sql_arg(part)?;
        let e = self.parse_sql_arg(e)?;
        let every = match &part {
            Expr::Literal(p) if p.is_null() => None,
            Expr::Literal(p) if p.extract_str().is_some() => {
                let p = p.extract_str().unwrap();
                Some(match p.to_ascii_lowercase().as_str() {
                    "microsecond" | "microseconds" => "1us",
                    "millisecond" | "milliseconds" => "1ms",
                    "second" | "seconds" => "1s",
                    "minute" | "minutes" => "1m",
                    "hour" | "hours" => "1h",
                    "day" | "days" => "1d",
                    "week" | "weeks" => "1w",
                    "month" | "months" => "1mo",
                    "quarter" | "quarters" => "1q",
                    "year" | "years" => "1y",
                    _ => polars_bail!(SQLSyntax: "DATE_TRUNC does not support '{}' part", p),
                })
            },
            _ => polars_bail!(SQLSyntax: "invalid 'part' for DATE_TRUNC ({})", part),
        };
        // A date is truncated as a timestamp, which is also what is returned.
        let dtype = self.ctx.expr_dtype(&e, self.active_schema);
        let e = match dtype {
            Some(DataType::Date) => e.cast(DataType::Datetime(TimeUnit::Microseconds, None)),
            Some(_) => e,
            None if every.is_none() => e,
            None => polars_bail!(SQLInterface: "DATE_TRUNC cannot determine the type of {}", e),
        };
        let Some(every) = every else {
            return self.apply_window_spec(null_reading(None, &[e]));
        };
        self.apply_window_spec(e.dt().truncate(lit(every)))
    }

    /// `STRPTIME(s, format)`; a string that does not match the format raises if `strict`,
    /// else gives null.
    fn visit_strptime(&mut self, name: &str, strict: bool) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        polars_ensure!(
            args.len() == 2,
            SQLSyntax: "{} expects 2 arguments (found {})", name, args.len()
        );
        self.visit_binary(|e, opts: StrptimeOptions| {
            e.str().strptime(
                DataType::Datetime(TimeUnit::Microseconds, None),
                StrptimeOptions { strict, ..opts },
                lit("latest"),
            )
        })
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
        let mut parsed = parse_sql_expr(expr, self.ctx, self.active_schema)?;
        if std::mem::take(&mut self.string_input) {
            parsed = null_as_string(parsed);
        }
        if self.reads_rows {
            self.read_subqueries_per_row(&parsed);
        }
        Ok(if self.reads_rows && is_constant_key(&parsed) {
            ValueArg::Constant(parsed)
        } else {
            ValueArg::Rows(self.apply_filter(parsed))
        })
    }

    /// Record the scalar subqueries in `expr` as read once per row (see
    /// `GroupScope::subqueries_read_per_row`).
    fn read_subqueries_per_row(&mut self, expr: &Expr) {
        for e in expr {
            if let Expr::SubPlan(_, names) = e {
                let read_per_row = &mut self.ctx.group_scope.subqueries_read_per_row;
                read_per_row.extend(names.iter().map(|(name, _)| name.clone()));
            }
        }
    }

    /// The number of rows an aggregate reads: all rows, or those that pass FILTER.
    fn rows_read(&self) -> Expr {
        match &self.filter {
            // A constant predicate holds for all rows or none.
            Some(pred) if is_constant_key(pred) => when(pred.clone()).then(len()).otherwise(lit(0)),
            Some(pred) => pred.clone().sum(),
            None => len(),
        }
    }

    /// `PERCENTILE_CONT(p)` / `PERCENTILE_DISC(p)` `WITHIN GROUP (ORDER BY value)`, the
    /// `QUANTILE_CONT` / `QUANTILE_DISC` of the values.
    fn visit_percentile(&mut self, function_name: &PolarsSQLFunctions) -> PolarsResult<Expr> {
        let (fname, method) = match function_name {
            PolarsSQLFunctions::PercentileCont => ("PERCENTILE_CONT", QuantileMethod::Linear),
            _ => ("PERCENTILE_DISC", QuantileMethod::Equiprobable),
        };
        let args = extract_args(self.func)?;
        let ([FunctionArgExpr::Expr(quantile)], [order_by]) =
            (args.as_slice(), self.func.within_group.as_slice())
        else {
            polars_bail!(
                SQLSyntax: "{} expects one argument and WITHIN GROUP (ORDER BY <value>)", fname
            )
        };
        polars_ensure!(
            matches!(order_by.options.sort, None | Some(OrderBySort::Asc)),
            SQLInterface: "{} only supports an ascending WITHIN GROUP order", fname
        );
        // Parameters are not subject to an active FILTER clause; only the values are.
        let quantile = parse_sql_expr(quantile, self.ctx, self.active_schema)?;
        let quantile = parse_quantile_literal(quantile, fname, args[0])?;
        let e = self.aggregate_value(&order_by.expr, |e| e.quantile(quantile.clone(), method))?;
        self.apply_window_spec(e)
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

    fn visit_series(&mut self, include_end: bool) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let args = args
            .into_iter()
            .map(|arg| match arg {
                FunctionArgExpr::Expr(sql_expr) => Ok(sql_expr),
                _ => self.not_supported_error(),
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let fname = self.func.name.to_string();
        series_expr(&fname, &args, include_end, self.active_schema, |e| {
            self.parse_sql_arg(e)
        })
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

    /// Like `visit_unary`, for STDDEV or VARIANCE: over two or more rows, the sample spread of
    /// a constant is that of two copies of it (0, or NaN for a value that is not finite).
    fn visit_spread_aggregate(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let sql_expr = match args.as_slice() {
            [FunctionArgExpr::Expr(sql_expr)] => sql_expr,
            [FunctionArgExpr::Wildcard] => &SQLExpr::Wildcard(AttachedToken::empty()),
            _ => return self.not_supported_error(),
        };
        let e = match self.parse_value_arg(sql_expr)? {
            ValueArg::Rows(e) => f(e),
            ValueArg::Constant(c) => when(self.rows_read().gt(lit(1)))
                .then(f(polars_lazy::dsl::repeat(c, lit(2))))
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
        let expr_args = self.parse_variadic_args()?;
        f(&expr_args).and_then(|e| self.apply_window_spec(e))
    }

    /// [`Self::visit_variadic`] for a function whose arguments share one type: a string
    /// literal is read as a value of the common temporal type of the other arguments, as
    /// in `COALESCE(date_col, '2020-01-01')`.
    fn visit_temporal_variadic(
        &mut self,
        f: impl Fn(&[Expr]) -> PolarsResult<Expr>,
    ) -> PolarsResult<Expr> {
        let mut expr_args = self.parse_variadic_args()?;
        polars_ensure!(
            !expr_args.is_empty(),
            SQLSyntax: "{} expects at least 1 argument (found 0)",
            self.func.name.to_string().to_uppercase()
        );
        let is_string_literal =
            |e: &Expr| matches!(e, Expr::Literal(lv) if lv.extract_str().is_some());
        if expr_args.iter().any(is_string_literal) {
            let mut dtypes = expr_args
                .iter()
                .filter(|e| !is_string_literal(e))
                .filter_map(|e| self.ctx.expr_dtype(e, self.active_schema))
                .filter(|dtype| dtype.is_temporal());
            let dtype = dtypes.next().and_then(|first| {
                dtypes.try_fold(first, |acc, dtype| try_get_supertype(&acc, &dtype).ok())
            });
            if let Some(dtype) = dtype {
                for arg in &mut expr_args {
                    *arg = temporal_string_as(arg, Some(&dtype));
                }
            }
        }
        self.apply_window_spec(f(&expr_args)?)
    }

    fn parse_variadic_args(&mut self) -> PolarsResult<Vec<Expr>> {
        let args = extract_args(self.func)?;
        let mut expr_args = vec![];
        for arg in args {
            if let FunctionArgExpr::Expr(sql_expr) = arg {
                expr_args.push(self.parse_sql_arg(sql_expr)?);
            } else {
                return self.not_supported_error();
            };
        }
        Ok(expr_args)
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
                _ => {
                    polars_bail!(SQLInterface: "'{}' is not supported in {}", clause, func_name)
                },
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
                // NULL, not an empty list, when no row is read.
                let array = when(self.rows_read().gt(lit(0)))
                    .then(base.implode(true))
                    .otherwise(Expr::Literal(LiteralValue::untyped_null()));
                self.apply_window_spec(array)
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

    /// BOOL_AND or BOOL_OR, as the MIN or MAX of a boolean. DISTINCT doesn't change them.
    fn visit_bool_aggregate(&mut self, f: impl Fn(Expr) -> Expr) -> PolarsResult<Expr> {
        let (args, _) = extract_args_distinct(self.func)?;
        let [FunctionArgExpr::Expr(sql_expr)] = args.as_slice() else {
            return self.not_supported_error();
        };
        let e = self.aggregate_value(sql_expr, f)?;
        self.boolean_arg(e)
    }

    /// `e` as a boolean: a NULL is a boolean NULL, and other types raise.
    pub(crate) fn boolean_arg(&mut self, e: Expr) -> PolarsResult<Expr> {
        match self.arg_dtype(&e)? {
            Some(DataType::Boolean) => Ok(e),
            None | Some(DataType::Null | DataType::Unknown(UnknownKind::Any)) => {
                Ok(e.cast(DataType::Boolean))
            },
            Some(dtype) => polars_bail!(
                InvalidOperation: "{} expects a boolean argument, got `{}`", self.func.name, dtype
            ),
        }
    }

    /// The dtype of `e`, with each subquery typed by its result.
    fn arg_dtype(&mut self, e: &Expr) -> PolarsResult<Option<DataType>> {
        let typed = self.ctx.with_typed_subqueries(e)?;
        Ok(self.ctx.expr_dtype(&typed, self.active_schema))
    }

    fn visit_count_if(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [FunctionArgExpr::Expr(sql_expr)] = args.as_slice() else {
            return self.not_supported_error();
        };
        let count = match self.parse_value_arg(sql_expr)? {
            ValueArg::Rows(e) => self.boolean_arg(e)?.sum(),
            ValueArg::Constant(c) => when(self.boolean_arg(c)?)
                .then(self.rows_read())
                .otherwise(lit(0)),
        };
        Ok(count.cast(DataType::Int64))
    }

    #[cfg(feature = "approx_unique")]
    fn visit_approx_count_distinct(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [FunctionArgExpr::Expr(sql_expr)] = args.as_slice() else {
            return self.not_supported_error();
        };
        let count = match self.parse_value_arg(sql_expr)? {
            ValueArg::Rows(e) => e.drop_nulls().approx_n_unique(),
            ValueArg::Constant(c) => when(self.rows_read().gt(lit(0)).and(c.is_not_null()))
                .then(lit(1))
                .otherwise(lit(0)),
        };
        self.apply_window_spec(count.cast(DataType::Int64))
    }

    /// REGEXP_EXTRACT(value, pattern [, group]): the group of the first match, or an empty
    /// string if there is none.
    fn visit_regexp_extract(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let (value, pattern, group) = match args.as_slice() {
            [FunctionArgExpr::Expr(value), FunctionArgExpr::Expr(pattern)] => (value, pattern, 0),
            [
                FunctionArgExpr::Expr(value),
                FunctionArgExpr::Expr(pattern),
                FunctionArgExpr::Expr(group),
            ] => {
                let group = match parse_sql_expr(group, self.ctx, self.active_schema)? {
                    Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) => {
                        usize::try_from(n).ok()
                    },
                    _ => None,
                };
                let Some(group) = group else {
                    polars_bail!(
                        SQLSyntax: "REGEXP_EXTRACT expects a non-negative integer literal as the group; found {}",
                        args[2]
                    )
                };
                (value, pattern, group)
            },
            _ => {
                polars_bail!(SQLSyntax: "REGEXP_EXTRACT expects 2-3 arguments (found {})", args.len())
            },
        };
        let value = self.string_arg(value)?;
        let pattern = self.string_arg(pattern)?;
        let extracted = value.clone().str().extract(pattern.clone(), group);
        let e = when(value.is_not_null().and(pattern.is_not_null()))
            .then(extracted.fill_null(lit("")))
            .otherwise(Expr::Literal(LiteralValue::untyped_null()));
        self.apply_window_spec(e)
    }

    /// `REGEXP_SUBSTR(string, pattern [, 1 [, n [, flags [, subexpr]]]])`, as in Postgres.
    fn visit_regexp_substr(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let exprs: Vec<&SQLExpr> = args
            .iter()
            .map(|arg| match arg {
                FunctionArgExpr::Expr(e) => Ok(e),
                _ => polars_bail!(SQLSyntax: "REGEXP_SUBSTR expects expressions; found {}", arg),
            })
            .collect::<PolarsResult<_>>()?;
        let (value, pattern, rest) = match exprs.as_slice() {
            [value, pattern, rest @ ..] if rest.len() <= 4 => (value, pattern, rest),
            _ => {
                polars_bail!(SQLSyntax: "REGEXP_SUBSTR expects 2-6 arguments (found {})", args.len())
            },
        };
        // An argument left out is `min`.
        let int_arg = |this: &mut Self, i: usize, name: &str, min: i64| -> PolarsResult<i64> {
            let Some(arg) = rest.get(i) else {
                return Ok(min);
            };
            match parse_sql_expr(arg, this.ctx, this.active_schema)? {
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(n))) if n >= min as i128 => {
                    Ok(n as i64)
                },
                _ => polars_bail!(
                    SQLSyntax: "REGEXP_SUBSTR expects an integer literal of at least {} as `{}`; found {}",
                    min, name, arg
                ),
            }
        };
        let start = int_arg(self, 0, "start", 1)?;
        let n = int_arg(self, 1, "n", 1)?;
        let subexpr = int_arg(self, 3, "subexpr", 0)?;
        let mut case_insensitive = false;
        if let Some(flags) = rest.get(2) {
            let SQLExpr::Value(ValueWithSpan {
                value: SQLValue::SingleQuotedString(flags),
                ..
            }) = flags
            else {
                polars_bail!(SQLSyntax: "REGEXP_SUBSTR expects literal flags; found {}", flags)
            };
            // As in Postgres, the last of 'c' and 'i' decides.
            for flag in flags.chars() {
                match flag {
                    'i' => case_insensitive = true,
                    'c' => case_insensitive = false,
                    _ => {
                        polars_bail!(SQLSyntax: "REGEXP_SUBSTR supports the flags 'c' and 'i'; found '{}'", flag)
                    },
                }
            }
        }
        // Another start would need `^` to anchor at the start of the whole string.
        polars_ensure!(
            start == 1,
            SQLInterface: "REGEXP_SUBSTR supports a `start` of 1 only (found {})", start
        );
        // As in Postgres, a pattern without capture groups is its own group 1.
        let groups = match pattern {
            SQLExpr::Value(ValueWithSpan {
                value: SQLValue::SingleQuotedString(p),
                ..
            }) => polars_utils::regex_cache::compile_regex(p)
                .ok()
                .map(|re| re.captures_len() - 1),
            _ => None,
        };
        let subexpr = match (subexpr, groups) {
            (1, Some(0)) => 0,
            (subexpr, _) => subexpr,
        };
        let value = self.string_arg(value)?;
        let mut pattern = self.string_arg(pattern)?;
        if case_insensitive {
            pattern = concat_str([lit("(?i)"), pattern], "", false);
        }
        let e = match (n, subexpr) {
            (1, group) => value.str().extract(pattern, group as usize),
            (n, 0) => value
                .str()
                .extract_all(pattern)
                .list()
                .get(lit(n - 1), true),
            _ => polars_bail!(
                SQLInterface: "REGEXP_SUBSTR of a capture group (`subexpr`) is supported for the first match only"
            ),
        };
        self.apply_window_spec(e)
    }

    /// `JSON_EXTRACT_PATH[_TEXT](json, key, ...)`: `json #> '{key,...}'` (or `#>>`).
    fn visit_json_extract_path(&mut self, text: bool) -> PolarsResult<Expr> {
        let name = match text {
            true => "JSON_EXTRACT_PATH_TEXT",
            false => "JSON_EXTRACT_PATH",
        };
        let args = extract_args(self.func)?;
        let [FunctionArgExpr::Expr(json), keys @ ..] = args.as_slice() else {
            polars_bail!(SQLSyntax: "{} expects a JSON value and a path of keys", name)
        };
        polars_ensure!(!keys.is_empty(), SQLSyntax: "{} expects at least one key", name);
        let elements = keys
            .iter()
            .map(|key| {
                json_path_element(key).ok_or_else(
                    || polars_err!(SQLSyntax: "{} expects literal keys; found {}", name, key),
                )
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let path = format!("{{{}}}", elements.join(","));
        let access = SQLExpr::BinaryOp {
            left: Box::new(json.clone()),
            op: match text {
                true => SQLBinaryOperator::HashLongArrow,
                false => SQLBinaryOperator::HashArrow,
            },
            right: Box::new(SQLExpr::Value(ValueWithSpan {
                value: SQLValue::SingleQuotedString(path),
                span: Span::empty(),
            })),
        };
        let e = self.parse_sql_arg(&access)?;
        self.apply_window_spec(e)
    }

    /// A string argument, where a NULL is a string NULL.
    fn string_arg(&mut self, e: &SQLExpr) -> PolarsResult<Expr> {
        let e = self.parse_sql_arg(e)?;
        Ok(match self.arg_dtype(&e)? {
            Some(DataType::Null) => e.cast(DataType::String),
            _ => e,
        })
    }

    fn visit_regexp_matches(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let (value, pattern, options) = match args.as_slice() {
            [FunctionArgExpr::Expr(value), FunctionArgExpr::Expr(pattern)] => {
                (value, pattern, None)
            },
            [
                FunctionArgExpr::Expr(value),
                FunctionArgExpr::Expr(pattern),
                FunctionArgExpr::Expr(options),
            ] => (value, pattern, Some(options)),
            _ => {
                polars_bail!(SQLSyntax: "REGEXP_MATCHES expects 2-3 arguments (found {})", args.len())
            },
        };
        let options = match options {
            None => RegexpOptions::default(),
            Some(options) => match self.parse_sql_arg(options)? {
                Expr::Literal(lv) if lv.extract_str().is_some() => {
                    RegexpOptions::new(lv.extract_str().unwrap())?
                },
                _ => {
                    polars_bail!(SQLSyntax: "REGEXP_MATCHES expects a string literal as the options (found {})", options)
                },
            },
        };
        let value = self.string_arg(value)?;
        let pattern = self.string_arg(pattern)?;
        let e = options.contains(value, pattern);
        self.apply_window_spec(e)
    }

    /// TO_DATE or TO_TIMESTAMP, where `parse` converts the string. With `from_epoch`, a
    /// number is the seconds since the Unix epoch, giving a UTC datetime.
    fn visit_parse_template(
        &mut self,
        function: &str,
        from_epoch: bool,
        parse: impl Fn(Expr, StrptimeOptions) -> Expr,
    ) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let (value, template) = match args.as_slice() {
            [FunctionArgExpr::Expr(value)] => (value, None),
            [
                FunctionArgExpr::Expr(value),
                FunctionArgExpr::Expr(template),
            ] => (value, Some(String::from_sql_arg(template, self)?)),
            _ => {
                polars_bail!(SQLSyntax: "{} expects 1-2 arguments (found {})", function, args.len())
            },
        };
        let value = self.parse_sql_arg(value)?;
        let mut value = match self.arg_dtype(&value)? {
            Some(DataType::Null) => value.cast(DataType::String),
            Some(dtype)
                if from_epoch
                    && template.is_none()
                    && (dtype.is_primitive_numeric() || dtype.is_decimal()) =>
            {
                let micros = if dtype.is_float() {
                    (value.cast(DataType::Float64) * lit(1_000_000.0))
                        .round(0, RoundMode::HalfToEven)
                } else {
                    // The cast rounds half to even; a value out of range becomes null.
                    value.cast(DataType::Decimal(38, 6)) * lit(1_000_000i64)
                };
                let micros = micros.cast(DataType::Int64);
                let dtype = DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC));
                return self.apply_window_spec(micros.cast(dtype));
            },
            _ => value,
        };
        let mut options = StrptimeOptions::default();
        if let Some(template) = template {
            let ParseFormat { format, defaults } = parse_format(&template, function)?;
            if !defaults.is_empty() {
                value = concat_str([value, lit(defaults)], "", false);
            }
            options.format = Some(format.into());
        }
        let e = parse(value, options);
        self.apply_window_spec(e)
    }

    fn visit_to_char(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [
            FunctionArgExpr::Expr(value),
            FunctionArgExpr::Expr(template),
        ] = args.as_slice()
        else {
            polars_bail!(SQLSyntax: "TO_CHAR expects 2 arguments (found {})", args.len())
        };
        let value = self.parse_sql_arg(value)?;
        let template = String::from_sql_arg(template, self)?;
        let value = match self.arg_dtype(&value)? {
            // A date is written as a datetime at midnight, so that time patterns work.
            Some(DataType::Date | DataType::Null) => {
                value.cast(DataType::Datetime(TimeUnit::Microseconds, None))
            },
            Some(dtype) if dtype.is_primitive_numeric() || dtype.is_decimal() => {
                polars_bail!(SQLInterface: "TO_CHAR does not support numbers (found {})", dtype)
            },
            _ => value,
        };
        let e = format_with_template(value, &template)?;
        self.apply_window_spec(e)
    }

    fn visit_typeof(&mut self) -> PolarsResult<Expr> {
        let args = extract_args(self.func)?;
        let [FunctionArgExpr::Expr(arg)] = args.as_slice() else {
            polars_bail!(SQLSyntax: "TYPEOF expects 1 argument (found {})", args.len())
        };
        let e = self.parse_sql_arg(arg)?;
        let Some(dtype) = self.arg_dtype(&e)? else {
            polars_bail!(SQLInterface: "TYPEOF cannot determine the type of {}", arg)
        };
        let name = lit(sql_type_name(&dtype.materialize_unknown(true)?));
        // The argument is kept for its number of rows and its checks against the GROUP BY, except
        // when it is an aggregate, whose value need not be computed.
        let e = match e {
            Expr::Literal(_) => name,
            e if is_marked_aggregate(&e) => mark_aggregate(name.first()),
            e => when(e.is_null()).then(name.clone()).otherwise(name),
        };
        self.apply_window_spec(e)
    }

    fn visit_count(&mut self) -> PolarsResult<Expr> {
        let (args, is_distinct) = extract_args_distinct(self.func)?;
        let distinct_count = |e: Expr| e.drop_nulls().n_unique();
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

    /// SUM of a constant: the SUM of the constant alone, times the number of rows read without
    /// DISTINCT, in the dtype of that SUM. NULL when no row is read or the constant is NULL.
    fn sum_of_constant(&self, c: Expr, is_distinct: bool) -> Expr {
        // Integer literals are summed as Int64, as in `literal_sum`.
        let c = match c {
            e @ Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(_))) => {
                e.cast(DataType::Int64)
            },
            e => e,
        };
        let rows = self.rows_read();
        let sum = c.clone().sum();
        let total = if is_distinct {
            sum
        } else {
            // Signed: an integer constant takes the other operand's type.
            let rows = rows.clone().cast(DataType::Int64);
            (sum.clone() * rows).cast(DataTypeExpr::OfExpr(Box::new(sum)))
        };
        when(rows.gt(lit(0)).and(c.is_not_null()))
            .then(total)
            .otherwise(Expr::Literal(LiteralValue::untyped_null()))
    }

    fn apply_order_by(&mut self, expr: Expr, order_by: &[OrderByExpr]) -> PolarsResult<Expr> {
        let mut by = Vec::with_capacity(order_by.len());
        let mut descending = Vec::with_capacity(order_by.len());
        let mut nulls_last = Vec::with_capacity(order_by.len());

        for ob in order_by {
            // Note: ORDER BY exprs share their length with the (possibly filtered) base,
            // so they have to go through `parse_value_arg` to apply any active FILTER.
            match self.parse_value_arg(&ob.expr)? {
                ValueArg::Rows(e) => by.push(e),
                // A constant key does not change the order.
                ValueArg::Constant(_) => continue,
            }

            let options = order_by_sort_options(&ob.options);
            nulls_last.push(options.nulls_last);
            descending.push(options.descending);
        }
        if by.is_empty() {
            return Ok(expr);
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
        // The distinct values have no separate draw of `RANDOM()` to sort by.
        polars_ensure!(
            !order_by.iter().any(|ob| contains_random(&ob.expr, true)),
            SQLInterface: "RANDOM() in the ORDER BY of a DISTINCT aggregate is not supported"
        );
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
        let expr = match (partition_by, order_by) {
            (None, None) if self.ctx.group_scope.mark_whole_frame_windows => {
                expr.over([col(self.ctx.whole_frame_partition())])?
            },
            (None, None) => expr,
            (Some(part), None) => expr.over(part)?,
            (part, Some(order)) => expr.over_with_options(part, Some(order), Default::default())?,
        };
        // SQL results have no row order, so the window may reorder the rows.
        Ok(match expr {
            Expr::Over {
                function,
                partition_by,
                order_by,
                mapping,
                may_reorder: _,
            } => Expr::Over {
                function,
                partition_by,
                order_by,
                mapping,
                may_reorder: true,
            },
            expr => expr,
        })
    }

    /// Visit `list_transform(list, lambda x: body)`, or `list_filter` if `filter`.
    fn visit_list_lambda(&mut self, filter: bool) -> PolarsResult<Expr> {
        let name = self.func.name.to_string().to_uppercase();
        let FunctionArguments::List(FunctionArgumentList {
            args,
            duplicate_treatment: None,
            clauses,
        }) = &self.func.args
        else {
            polars_bail!(SQLSyntax: "{name} expects a list and a lambda")
        };
        let ([FunctionArg::Unnamed(FunctionArgExpr::Expr(list)), lambda], true) =
            (args.as_slice(), clauses.is_empty())
        else {
            polars_bail!(SQLSyntax: "{name} expects a list and a lambda")
        };
        let Some((n_params, body)) = lambda_arg(lambda) else {
            polars_bail!(SQLSyntax: "the second argument of {name} must be a lambda, as in `lambda x: x + 1`")
        };
        polars_ensure!(
            n_params == 1,
            SQLInterface: "the lambda of {name} takes one parameter (found {n_params})"
        );
        let mut list = self.parse_sql_arg(list)?;
        let element_dtype = match self.arg_dtype(&list)? {
            Some(DataType::Array(inner, _)) => {
                list = list.arr().to_list();
                Some(*inner)
            },
            Some(DataType::Null) => {
                list = list.cast(DataType::List(Box::new(DataType::Null)));
                Some(DataType::Null)
            },
            Some(DataType::List(inner)) => Some(*inner),
            Some(dtype) => polars_bail!(
                SQLSyntax: "{name} expects a list; found {}", sql_type_name(&dtype)
            ),
            // Without a schema (as in `sql_expr`) no type is known.
            None => {
                polars_ensure!(
                    self.active_schema.is_none(),
                    SQLInterface: "cannot determine the type of the list of {name}; cast it, as in `CAST(... AS INT[])`"
                );
                None
            },
        };
        polars_ensure!(
            !contains_random(body, true),
            SQLInterface: "RANDOM() is not supported in a lambda"
        );
        // The body reads the element as a column, so that its type is known while parsing. The
        // other columns it reads are the values of the row.
        self.ctx.lambda_depth += 1;
        let element_name = lambda_element_name(self.ctx.lambda_depth);
        let mut schema = self.active_schema.cloned().unwrap_or_default();
        if let Some(dtype) = element_dtype {
            schema.with_column(element_name.clone(), dtype);
        }
        let body = parse_sql_expr(body, self.ctx, Some(&schema));
        self.ctx.lambda_depth -= 1;
        let body = body?.map_expr(|e| match e {
            Expr::Column(name) if name == element_name => element(),
            e => e,
        });
        let body = if filter {
            element().filter(body.cast(DataType::Boolean))
        } else {
            body
        };
        self.apply_window_spec(list.list().eval(body))
    }

    pub(crate) fn not_supported_error<T>(&self) -> PolarsResult<T> {
        polars_bail!(
            SQLInterface:
            "no function matches the given name and arguments: `{}`",
            self.func.to_string()
        );
    }
}

/// A pseudo-random value in [0, 1) per row, from a hash of the row index with a new seed. The
/// top 53 bits of the hash are exact in a `Float64`.
fn random() -> Expr {
    let seed = polars_core::random::get_global_random_u64();
    int_range(lit(0), len(), 1, polars_core::datatypes::IDX_DTYPE)
        .hash(seed)
        .floor_div(lit(1u64 << 11))
        .cast(DataType::Float64)
        * lit(1.0 / (1u64 << 53) as f64)
}

/// Whether `expr` is the hash of the row index made by [`random`]. Its values depend on the
/// rows it runs on, so it must run on the frame the SQL reads it from.
pub(crate) fn is_random(expr: &Expr) -> bool {
    match expr {
        Expr::Function {
            input,
            function: FunctionExpr::Hash(_),
        } => matches!(
            input.as_slice(),
            [Expr::Function {
                input,
                function: FunctionExpr::Range(RangeFunction::IntRange { .. }),
            }] if matches!(input.as_slice(), [_, Expr::Len])
        ),
        _ => false,
    }
}

/// Make each [`random`] in `expr` one value, the one it gives the first row, with the same seed.
pub(crate) fn random_once(expr: Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::Function {
            function: FunctionExpr::Hash(seed),
            ..
        } if is_random(&e) => lit(Scalar::new_idxsize(0)).hash(seed),
        e => e,
    })
}

/// Whether a window key is a scalar that doesn't depend on the input, as `1` or `LOWER('A')`.
pub(crate) fn is_constant_key(key: &Expr) -> bool {
    matches!(key.clone().meta().is_input_independent_scalar(), Ok(true))
}

/// Whether `expr` reads no column of the input. Unlike [`is_constant_key`], this also holds for
/// an aggregate or membership test of constants, as `MAX(1)` or `1 IN (1, 2)`.
pub(crate) fn reads_no_input(expr: &Expr) -> bool {
    !expr
        .into_iter()
        .any(|e| matches!(e, Expr::Column(_) | Expr::Selector(_) | Expr::Len))
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
                FunctionExpr::RowEncode {
                    variant: RowEncodingVariant::Ordered {
                        descending: Some(descending),
                        nulls_last: Some(nulls_last),
                    },
                    output: RowEncodingOutputType::AlwaysBinaryOffset,
                    encode_nulls: None,
                    propagate_nulls: false,
                },
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

/// Parses a quantile argument, which must be a literal, and checks it like any `quantile`
/// fraction.
fn parse_quantile_literal(
    quantile: Expr,
    fname: &str,
    arg: &FunctionArgExpr,
) -> PolarsResult<Expr> {
    let fraction = match &quantile {
        Expr::Literal(lv) => lv.to_quantile_fraction()?,
        _ => None,
    };
    let Some(fraction) = fraction else {
        polars_bail!(SQLSyntax: "invalid value for {} ({})", fname, arg)
    };
    let quantiles = extract_non_null_quantiles(&fraction, QuantileFractionDtypes::Number)?;
    Ok(Expr::from(quantiles[0]))
}

/// SQL `REGEXP_REPLACE(e, pattern, replacement [, flags])`. The 'g' flag replaces every
/// match; the other flags are inline regex flags.
fn regexp_replace(
    e: &Expr,
    pat: &Expr,
    replacement: &Expr,
    flags: Option<&Expr>,
) -> PolarsResult<Expr> {
    let is_null = |e: &Expr| matches!(e, Expr::Literal(lv) if lv.is_null());
    if is_null(e) || is_null(pat) || is_null(replacement) || flags.is_some_and(is_null) {
        return Ok(lit(LiteralValue::untyped_null()).cast(DataType::String));
    }
    let replacement = match replacement {
        Expr::Literal(lv) if lv.extract_str().is_some() => {
            lit(sql_regex_replacement(lv.extract_str().unwrap()))
        },
        _ => {
            polars_bail!(SQLSyntax: "REGEXP_REPLACE expects a string literal replacement (found {})", replacement)
        },
    };
    let flags = match flags {
        None => "",
        Some(Expr::Literal(lv)) if lv.extract_str().is_some() => lv.extract_str().unwrap(),
        Some(flags) => polars_bail!(SQLSyntax: "invalid 'flags' for REGEXP_REPLACE ({})", flags),
    };
    let inline_flags: String = flags.chars().filter(|&c| c != 'g').collect();
    let pat = match pat {
        Expr::Literal(lv) if lv.extract_str().is_some() => {
            let pat = lv.extract_str().unwrap();
            if inline_flags.is_empty() {
                lit(pat)
            } else {
                lit(format!("(?{inline_flags}){pat}"))
            }
        },
        _ => {
            polars_bail!(SQLSyntax: "REGEXP_REPLACE expects a string literal pattern (found {})", pat)
        },
    };
    let e = e.clone().str();
    Ok(if flags.contains('g') {
        e.replace_all(pat, replacement, false)
    } else {
        e.replace(pat, replacement, false)
    })
}

/// Convert a SQL regex replacement string to the Polars syntax: `\1` and `\&` refer to
/// capture groups, and `$` is a literal character.
fn sql_regex_replacement(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.peek() {
                Some(&d) if d.is_ascii_digit() => {
                    out.push_str("${");
                    out.push(d);
                    out.push('}');
                    chars.next();
                },
                Some('&') => {
                    out.push_str("${0}");
                    chars.next();
                },
                Some('\\') => {
                    out.push('\\');
                    chars.next();
                },
                _ => out.push('\\'),
            },
            '$' => out.push_str("$$"),
            _ => out.push(c),
        }
    }
    out
}

/// A list from start to stop for SQL `generate_series` (stop included) and `range` (stop not
/// included). Takes `(stop)`, `(start, stop)` or `(start, stop, step)`; a date or timestamp
/// series takes an INTERVAL step and returns timestamps.
pub(crate) fn series_expr(
    fname: &str,
    args: &[&SQLExpr],
    include_end: bool,
    schema: Option<&Schema>,
    mut parse: impl FnMut(&SQLExpr) -> PolarsResult<Expr>,
) -> PolarsResult<Expr> {
    let int_series = |start: Expr, stop: Expr, step: Expr| {
        let stop = if include_end {
            stop + step.clone().sign()
        } else {
            stop
        };
        int_ranges(start, stop, step, DataType::Int64)
    };
    let temporal_series = |start: Expr, stop: Expr, interval: Duration| {
        polars_ensure!(
            !interval.negative(),
            SQLInterface: "{} does not support negative INTERVAL steps", fname
        );
        let closed = if include_end {
            ClosedWindow::Both
        } else {
            ClosedWindow::Left
        };
        datetime_ranges(
            Some(start),
            Some(stop),
            Some(interval),
            None,
            closed,
            None,
            None,
        )
    };
    match args {
        [start, stop, SQLExpr::Interval(interval)] => temporal_series(
            parse(start)?,
            parse(stop)?,
            interval_to_duration(interval, false)?,
        ),
        [_, _, SQLExpr::UnaryOp { expr, .. }] if matches!(**expr, SQLExpr::Interval(_)) => {
            polars_bail!(SQLInterface: "{} does not support negative INTERVAL steps", fname)
        },
        [stop] => Ok(int_series(lit(0i64), parse(stop)?, lit(1i64))),
        [start, stop] => Ok(int_series(parse(start)?, parse(stop)?, lit(1i64))),
        [start, stop, step] => {
            let (start, stop, step) = (parse(start)?, parse(stop)?, parse(step)?);
            if !matches!(
                resolve_expr_dtype(&step, schema),
                Some(DataType::Duration(_))
            ) {
                return Ok(int_series(start, stop, step));
            }
            polars_ensure!(
                is_constant_key(&step),
                SQLInterface: "{} takes only a constant INTERVAL step", fname
            );
            let step = fold_scalar(step.strict_cast(DataType::Duration(TimeUnit::Nanoseconds)))?;
            let Expr::Literal(step) = step else {
                unreachable!()
            };
            match step.to_any_value() {
                Some(AnyValue::Duration(ns, _)) => {
                    polars_ensure!(
                        ns >= 0,
                        SQLInterface: "{} does not support negative INTERVAL steps", fname
                    );
                    temporal_series(start, stop, Duration::from_nsecs(ns))
                },
                // A NULL step gives a NULL series.
                _ => {
                    let null_stop = when(lit(false))
                        .then(stop)
                        .otherwise(lit(LiteralValue::untyped_null()));
                    temporal_series(start, null_stop, Duration::new_days(1))
                },
            }
        },
        _ => polars_bail!(SQLSyntax: "{} expects 1 to 3 arguments; found {}", fname, args.len()),
    }
}

/// The number of elements of the list `e`; a NULL value has no list type to read.
fn list_len(e: Expr, schema: Option<&Schema>) -> Expr {
    match resolve_expr_dtype(&e, schema) {
        Some(DataType::Null) => e.cast(DataType::Int64),
        _ => e.list().len().cast(DataType::Int64),
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
                polars_ensure!(
                    !args.iter().any(is_lambda_arg),
                    SQLSyntax: "a lambda is only supported in LIST_TRANSFORM and LIST_FILTER"
                );
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

/// The options of `REGEXP_MATCHES`, as in DuckDB.
#[derive(Default)]
struct RegexpOptions {
    case_insensitive: bool,
    dot_matches_new_line: bool,
    literal: bool,
}

impl RegexpOptions {
    fn new(options: &str) -> PolarsResult<Self> {
        let mut out = Self::default();
        for c in options.chars() {
            match c {
                'c' => out.case_insensitive = false,
                'i' => out.case_insensitive = true,
                'l' => out.literal = true,
                'm' | 'n' | 'p' => out.dot_matches_new_line = false,
                's' => out.dot_matches_new_line = true,
                'g' => polars_bail!(SQLSyntax: "REGEXP_MATCHES does not take the 'g' option"),
                c => polars_bail!(SQLSyntax: "invalid REGEXP_MATCHES option '{}'", c),
            }
        }
        Ok(out)
    }

    /// Whether `pattern` matches somewhere in `value`.
    fn contains(&self, value: Expr, pattern: Expr) -> Expr {
        let pattern = match (self.literal, self.case_insensitive) {
            (true, false) => return value.str().contains_literal(pattern),
            (true, true) => pattern.str().escape_regex(),
            (false, _) => pattern,
        };
        let flags = match (
            self.case_insensitive,
            self.dot_matches_new_line && !self.literal,
        ) {
            (false, false) => return value.str().contains(pattern, true),
            (true, false) => "(?i)",
            (false, true) => "(?s)",
            (true, true) => "(?is)",
        };
        value
            .str()
            .contains(concat_str([lit(flags), pattern], "", false), true)
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

/// An error for a format that has an unknown or incomplete `%` specifier.
fn check_strptime_format(format: &str) -> PolarsResult<()> {
    polars_ensure!(
        chrono::format::StrftimeItems::new(format).parse().is_ok(),
        SQLInterface: "invalid date/time format '{}'", format
    );
    Ok(())
}

impl FromSQLExpr for StrptimeOptions {
    fn from_sql_expr(expr: &SQLExpr, _: &mut SQLContext) -> PolarsResult<Self>
    where
        Self: Sized,
    {
        match expr {
            SQLExpr::Value(ValueWithSpan { value: v, .. }) => match v {
                SQLValue::SingleQuotedString(s) => {
                    check_strptime_format(s)?;
                    Ok(StrptimeOptions {
                        format: Some(PlSmallStr::from_str(s)),
                        ..StrptimeOptions::default()
                    })
                },
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

/// Translates a printf format string into a [`format_str`] format string and its arguments.
fn printf_to_format(
    printf: &str,
    args: &[Expr],
    used: &mut [bool],
) -> PolarsResult<(String, Vec<Expr>)> {
    let bytes = printf.as_bytes();
    let mut format = String::with_capacity(printf.len());
    let mut inputs = vec![];
    let mut next_arg = 0;
    let mut numbered = None;
    let mut literal_start = 0;
    let mut i = 0;
    let digits = |i: &mut usize| {
        let start = *i;
        while bytes.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        (start < *i).then(|| &printf[start..*i])
    };
    while i < bytes.len() {
        let (escaped, n_bytes) = match bytes[i] {
            b'{' => ("{{", 1),
            b'}' => ("}}", 1),
            b'%' if bytes.get(i + 1) == Some(&b'%') => ("%", 2),
            b'%' => ("", 1),
            _ => {
                i += 1;
                continue;
            },
        };
        format.push_str(&printf[literal_start..i]);
        format.push_str(escaped);
        let start = i;
        i += n_bytes;
        literal_start = i;
        if !escaped.is_empty() {
            continue;
        }

        // An argument number, as in `%2$s`.
        let mut position = None;
        let mut j = i;
        if let Some(n) = digits(&mut j)
            && bytes.get(j) == Some(&b'$')
        {
            position = Some(n.parse::<usize>().ok().filter(|n| *n > 0));
            i = j + 1;
        }
        let (mut left, mut zero, mut plus, mut space, mut alternate, mut comma) =
            (false, false, false, false, false, false);
        while let Some(flag) = bytes.get(i) {
            match flag {
                b'-' => left = true,
                b'0' => zero = true,
                b'+' => plus = true,
                b' ' => space = true,
                b'#' => alternate = true,
                b',' => comma = true,
                _ => break,
            }
            i += 1;
        }
        let width = digits(&mut i);
        let precision = if bytes.get(i) == Some(&b'.') {
            i += 1;
            Some(digits(&mut i).unwrap_or("0"))
        } else {
            None
        };
        // Length modifiers such as `ld` don't change the output.
        while matches!(
            bytes.get(i),
            Some(b'h' | b'l' | b'L' | b'q' | b'j' | b'z' | b't')
        ) {
            i += 1;
        }
        let conversion = bytes.get(i).copied();
        i += 1;
        literal_start = i;
        let spec_end = (i..printf.len())
            .find(|end| printf.is_char_boundary(*end))
            .unwrap_or(printf.len());
        let spec_str = &printf[start..spec_end];
        let ty = match conversion {
            Some(b'd' | b'i' | b'u') if precision.is_none() => 'd',
            Some(b'o') if !alternate => 'o',
            Some(t @ (b'c' | b'e' | b'E' | b'f' | b'F' | b'g' | b'G' | b's' | b'x' | b'X')) => {
                char::from(t)
            },
            _ => polars_bail!(SQLSyntax: "PRINTF does not support '{spec_str}'"),
        };

        // As in C, characters and strings ignore the numeric flags, and unsigned conversions
        // ignore the sign flags.
        let numeric = !matches!(ty, 'c' | 's');
        let zero = zero && !left && numeric;
        let build_spec = |zero: bool| {
            let mut spec = String::new();
            if left {
                spec.push('<');
            } else if ty == 's' {
                spec.push('>');
            }
            if numeric {
                let unsigned = matches!(conversion, Some(b'u' | b'o' | b'x' | b'X'));
                if plus && !unsigned {
                    spec.push('+');
                } else if space && !unsigned {
                    spec.push(' ');
                }
                if alternate {
                    spec.push('#');
                }
                if zero {
                    spec.push('0');
                }
            }
            spec.push_str(width.unwrap_or(""));
            if comma && numeric {
                spec.push(',');
            }
            // `%f` always has 6 decimals, also for decimals.
            if let Some(precision) = precision.or((ty == 'f' || ty == 'F').then_some("6")) {
                spec.push('.');
                spec.push_str(precision);
            }
            spec.push(ty);
            spec
        };
        let spec = build_spec(zero);

        let index = match position {
            Some(position) => {
                polars_ensure!(numbered != Some(false), SQLSyntax: "PRINTF cannot mix numbered and unnumbered arguments");
                numbered = Some(true);
                let Some(position) = position else {
                    polars_bail!(SQLSyntax: "PRINTF does not support '{spec_str}'");
                };
                position - 1
            },
            None => {
                polars_ensure!(numbered != Some(true), SQLSyntax: "PRINTF cannot mix numbered and unnumbered arguments");
                numbered = Some(false);
                next_arg += 1;
                next_arg - 1
            },
        };
        let Some(arg) = args.get(index) else {
            polars_bail!(SQLSyntax: "PRINTF format '{printf}' needs more arguments (found {})", args.len());
        };
        used[index] = true;
        if zero && matches!(ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G') {
            // C pads infinity and NaN with spaces, also with the `0` flag.
            let finite = arg.clone().cast(DataType::Float64).is_finite();
            let padded =
                |spec: &str| format_str(&format!("{{:{spec}}}"), std::slice::from_ref(arg));
            format.push_str(&format!("{{{}}}", inputs.len()));
            inputs.push(
                when(finite)
                    .then(padded(&spec)?)
                    .otherwise(padded(&build_spec(false))?),
            );
        } else {
            format.push_str(&format!("{{{}:{spec}}}", inputs.len()));
            inputs.push(arg.clone());
        }
    }
    format.push_str(&printf[literal_start..]);
    Ok((format, inputs))
}

/// Translates a FORMAT format string, with `{}` and `{0}` placeholders, into a [`format_str`]
/// format string and its arguments.
fn sql_format_to_format(
    sql_format: &str,
    args: &[Expr],
    used: &mut [bool],
) -> PolarsResult<(String, Vec<Expr>)> {
    let mut format = String::with_capacity(sql_format.len());
    let mut inputs = vec![];
    let mut next_arg = 0;
    let mut numbered = None;
    let mut rest = sql_format;
    while let Some(i) = rest.find(['{', '}']) {
        format.push_str(&rest[..i]);
        let (brace, after) = (rest.as_bytes()[i], &rest[i + 1..]);
        if after.as_bytes().first() == Some(&brace) {
            format.push_str(if brace == b'{' { "{{" } else { "}}" });
            rest = &after[1..];
            continue;
        }
        let close = after.find('}').filter(|_| brace == b'{').ok_or_else(
            || polars_err!(SQLSyntax: "unmatched brace in FORMAT format '{sql_format}'"),
        )?;
        let (arg, spec) = after[..close]
            .split_once(':')
            .unwrap_or((&after[..close], ""));
        let index = if arg.is_empty() {
            polars_ensure!(numbered != Some(true), SQLSyntax: "FORMAT cannot mix numbered and unnumbered placeholders");
            numbered = Some(false);
            next_arg += 1;
            next_arg - 1
        } else {
            polars_ensure!(numbered != Some(false), SQLSyntax: "FORMAT cannot mix numbered and unnumbered placeholders");
            numbered = Some(true);
            let valid =
                arg.bytes().all(|b| b.is_ascii_digit()) && (arg == "0" || !arg.starts_with('0'));
            arg.parse::<usize>().ok().filter(|_| valid).ok_or_else(
                || polars_err!(SQLSyntax: "FORMAT does not support the placeholder '{{{arg}}}'"),
            )?
        };
        let Some(arg) = args.get(index) else {
            polars_bail!(SQLSyntax: "FORMAT format '{sql_format}' needs more arguments (found {})", args.len());
        };
        used[index] = true;
        format.push_str(&format!("{{{}:{spec}}}", inputs.len()));
        inputs.push(arg.clone());
        rest = &after[close + 1..];
    }
    format.push_str(rest);
    Ok((format, inputs))
}

fn date_diff_unit(part: &str) -> Option<DateDiffUnit> {
    Some(match part {
        "year" | "years" | "yr" | "yrs" | "y" => DateDiffUnit::Year,
        "quarter" | "quarters" => DateDiffUnit::Quarter,
        "month" | "months" | "mon" | "mons" => DateDiffUnit::Month,
        "week" | "weeks" | "w" => DateDiffUnit::Week,
        "day" | "days" | "d" => DateDiffUnit::Day,
        "hour" | "hours" | "hr" | "hrs" | "h" => DateDiffUnit::Hour,
        "minute" | "minutes" | "min" | "mins" | "m" => DateDiffUnit::Minute,
        "second" | "seconds" | "sec" | "secs" | "s" => DateDiffUnit::Second,
        "millisecond" | "milliseconds" | "ms" => DateDiffUnit::Millisecond,
        "microsecond" | "microseconds" | "us" => DateDiffUnit::Microsecond,
        "nanosecond" | "nanoseconds" | "ns" => DateDiffUnit::Nanosecond,
        _ => return None,
    })
}

/// The name Polars SQL gives a function that other dialects name `name`.
fn spelling_of(name: &str) -> Option<&'static str> {
    match name {
        "len" => Some("LENGTH"),
        _ => None,
    }
}
