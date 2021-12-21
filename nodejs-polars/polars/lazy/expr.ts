import {DataType} from "../datatypes";
import pli from "../internals/polars_internal";
import {col, lit} from "./lazy_functions";
import {
  ExprOrString,
  FillNullStrategy,
  isExpr,
  RankMethod,
  regexToString,
  RollingOptions,
  selectionToExprList
} from "../utils";
import {Series} from "../series";
const inspect = Symbol.for("nodejs.util.inspect.custom");

type JsExpr = any;

/**
 * namespace containing expr list functions
 */
export interface ExprListFunctions {
  /**
   * Get the value by index in the sublists.
   * So index `0` would return the first item of every sublist
   * and index `-1` would return the last item of every sublist
   * if an index is out of bounds, it will return a `null`.
   */
  get(index: number): Expr
  /** Get the first value of the sublists. */
  first(): Expr
  /** Get the last value of the sublists. */
  last(): Expr
  lengths(): Expr;
  max(): Expr;
  mean(): Expr;
  min(): Expr;
  reverse(): Expr;
  sort(reverse?: boolean): Expr;
  sort(opt: {reverse: boolean}): Expr;
  sum(): Expr;
  unique(): Expr;
}

/**
 * namespace containing expr string functions
 */
export interface ExprStringFunctions {
  /**
   * Vertically concat the values in the Series to a single string value.
   * @example
   * ```
   * >>> df = pl.DataFrame({"foo": [1, null, 2]})
   * >>> df = df.select(pl.col("foo").str.concat("-"))
   * >>> df
   * shape: (1, 1)
   * ┌──────────┐
   * │ foo      │
   * │ ---      │
   * │ str      │
   * ╞══════════╡
   * │ 1-null-2 │
   * └──────────┘
   * ```
   */
  concat(delimiter: string): Expr;
  /** Check if strings in Series contain regex pattern. */
  contains(pat: string | RegExp): Expr;
  /**
   * Extract the target capture group from provided patterns.
   * @param pattern A valid regex pattern
   * @param groupIndex Index of the targeted capture group.
   * Group 0 mean the whole pattern, first group begin at index 1
   * Default to the first capture group
   * @returns Utf8 array. Contain null if original value is null or regex capture nothing.
   * @example
   * ```
   * > df = pl.DataFrame({
   * ...   'a': [
   * ...       'http://vote.com/ballon_dor?candidate=messi&ref=polars',
   * ...       'http://vote.com/ballon_dor?candidat=jorginho&ref=polars',
   * ...       'http://vote.com/ballon_dor?candidate=ronaldo&ref=polars'
   * ...   ]})
   * > df.select(pl.col('a').str.extract(/candidate=(\w+)/, 1))
   * shape: (3, 1)
   * ┌─────────┐
   * │ a       │
   * │ ---     │
   * │ str     │
   * ╞═════════╡
   * │ messi   │
   * ├╌╌╌╌╌╌╌╌╌┤
   * │ null    │
   * ├╌╌╌╌╌╌╌╌╌┤
   * │ ronaldo │
   * └─────────┘
   * ```
   */
  extract(pat: string | RegExp, groupIndex: number): Expr;
  /**
   * Extract the first match of json string with provided JSONPath expression.
   * Throw errors if encounter invalid json strings.
   * All return value will be casted to Utf8 regardless of the original value.
   * @see https://goessner.net/articles/JsonPath/
   * @param jsonPath - A valid JSON path query string
   * @returns Utf8 array. Contain null if original value is null or the `jsonPath` return nothing.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...   'json_val': [
   * ...     '{"a":"1"}',
   * ...     null,
   * ...     '{"a":2}',
   * ...     '{"a":2.1}',
   * ...     '{"a":true}'
   * ...   ]
   * ... })
   * >>> df.select(pl.col('json_val').str.jsonPathMatch('$.a')
   * shape: (5,)
   * Series: 'json_val' [str]
   * [
   *     "1"
   *     null
   *     "2"
   *     "2.1"
   *     "true"
   * ]
   * ```
   */
  jsonPathMatch(pat: string): Expr;
  /**  Get length of the string values in the Series. */
  lengths(): Expr;
  /** Remove leading whitespace. */
  lstrip(): Expr
  /** Replace first regex match with a string value. */
  replace(pat: string | RegExp, val: string): Expr;
  /** Replace all regex matches with a string value. */
  replaceAll(pat: string | RegExp, val: string): Expr;
  /** Modify the strings to their lowercase equivalent. */
  toLowerCase(): Expr;
  /** Modify the strings to their uppercase equivalent. */
  toUpperCase(): Expr;
  /** Remove trailing whitespace. */
  rstrip(): Expr
  /**
   * Create subslices of the string values of a Utf8 Series.
   * @param start - Start of the slice (negative indexing may be used).
   * @param length - Optional length of the slice.
   */
  slice(start: number, length?: number): Expr;
    /**
   * Parse a Series of dtype Utf8 to a Date/Datetime Series.
   * @param datatype Date or Datetime.
   * @param fmt formatting syntax. [Read more](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html)
   */
  strftime(datatype: DataType.Date, fmt?: string): Expr
  strftime(datatype: DataType.Datetime, fmt?: string): Expr
}

export interface ExprDateTimeFunctions {
  /**
   * Extract day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of month starting from 1.
   * The return value ranges from 1 to 31. (The last day of month differs by months.)
   * @returns day as pl.UInt32
   */
  day(): Expr;
  /**
   * Extract hour from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the hour number from 0 to 23.
   * @returns Hour as UInt32
   */
  hour(): Expr;
  /**
   * Extract minutes from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the minute number from 0 to 59.
   * @returns minute as UInt32
   */
  minute(): Expr;
  /**
   * Extract month from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the month number starting from 1.
   * The return value ranges from 1 to 12.
   * @returns Month as UInt32
   */
  month(): Expr;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the number of nanoseconds since the whole non-leap second.
   * The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
   * @returns Nanosecond as UInt32
   */
  nanosecond(): Expr;
  /**
   * Extract ordinal day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of year starting from 1.
   * The return value ranges from 1 to 366. (The last day of year differs by years.)
   * @returns Day as UInt32
   */
  ordinalDay(): Expr;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the second number from 0 to 59.
   * @returns Second as UInt32
   */
  second(): Expr;
  /**
   * Format Date/datetime with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
   */
  strftime(fmt: string): Expr;
  /** Return timestamp in ms as Int64 type. */
  timestamp(): Expr;
  /**
   * Extract the week from the underlying Date representation.
   * Can be performed on Date and Datetime
   *
   * Returns the ISO week number starting from 1.
   * The return value ranges from 1 to 53. (The last week of year differs by years.)
   * @returns Week number as UInt32
   */
  week(): Expr;
  /**
   * Extract the week day from the underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the weekday number where monday = 0 and sunday = 6
   * @returns Week day as UInt32
   */
  weekday(): Expr;
  /**
   * Extract year from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the year number in the calendar date.
   * @returns Year as Int32
   */
  year(): Expr;
}

export interface Expr {
  _expr: any;
  get date(): ExprDateTimeFunctions;
  get str(): ExprStringFunctions;
  get lst(): ExprListFunctions;
  [inspect](): string;
  /** Take absolute values */
  abs(): Expr
  aggGroups(): Expr
  /**
   * Rename the output of an expression.
   * @param name new name
   * @see {@link Expr.as}
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...   "a": [1, 2, 3],
   * ...   "b": ["a", "b", None],
   * ... })
   * >>> df
   * shape: (3, 2)
   * ╭─────┬──────╮
   * │ a   ┆ b    │
   * │ --- ┆ ---  │
   * │ i64 ┆ str  │
   * ╞═════╪══════╡
   * │ 1   ┆ "a"  │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 2   ┆ "b"  │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 3   ┆ null │
   * ╰─────┴──────╯
   * >>> df.select([
   * ...   pl.col("a").alias("bar"),
   * ...   pl.col("b").alias("foo"),
   * ... ])
   * shape: (3, 2)
   * ╭─────┬──────╮
   * │ bar ┆ foo  │
   * │ --- ┆ ---  │
   * │ i64 ┆ str  │
   * ╞═════╪══════╡
   * │ 1   ┆ "a"  │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 2   ┆ "b"  │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 3   ┆ null │
   * ╰─────┴──────╯
   *```
   */
  alias(name: string): Expr
  and(other: any): Expr
  /** Get the index of the maximal value. */
  argMax(): Expr
  /** Get the index of the minimal value. */
  argMin(): Expr
  /**
   * Get the index values that would sort this column.
   * @param reverse
   *     - false -> order from small to large.
   *     - true -> order from large to small.
   * @returns UInt32 Series
   */
  argSort(reverse?: boolean): Expr
  argSort({reverse}: {reverse: boolean}): Expr
  /** Get index of first unique value. */
  argUnique(): Expr
  /** @see {@link Expr.alias} */
  as(name: string): Expr
  /** Fill missing values with the next to be seen values */
  backwardFill(): Expr
  /** Cast between data types. */
  cast(dtype: DataType, strict?: boolean): Expr
  /** Count the number of values in this expression */
  count(): Expr
  /** Get an array with the cumulative count computed at every element. */
  cumCount(reverse?: boolean): Expr
  cumCount({reverse}: {reverse: boolean}): Expr
  /** Get an array with the cumulative max computed at every element. */
  cumMax(reverse?: boolean): Expr
  cumMax({reverse}: {reverse: boolean}): Expr
  /** Get an array with the cumulative min computed at every element. */
  cumMin(reverse?: boolean): Expr
  cumMin({reverse}: {reverse: boolean}): Expr
  /**
   * Get an array with the cumulative product computed at every element.
   *
   * @notes
   * *Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
   * Int64 before summing to prevent overflow issues.*
   */
  cumProd(reverse?: boolean): Expr
  cumProd({reverse}: {reverse: boolean}): Expr
  /**
   * Get an array with the cumulative sum computed at every element.
   *
   * @notes
   * *Dtypes in {Int8, UInt8, Int16, UInt16} are cast to
   * Int64 before summing to prevent overflow issues.*
   */
  cumSum(reverse?: boolean): Expr
  cumSum({reverse}: {reverse: boolean}): Expr
  /** Calculate the n-th discrete difference.
   *
   * @param n number of slots to shift
   * @param nullBehavior ignore or drop
   */
  diff(n: number, nullBehavior: "ignore" | "drop"): Expr
  diff(o: {n: number, nullBehavior: "ignore" | "drop"}): Expr
  /**
   * Compute the dot/inner product between two Expressions
   * @param other Expression to compute dot product with
   */
  dot(other: any): Expr
  eq(other: any): Expr
  /**
   * Exclude certain columns from a wildcard/regex selection.
   *
   * You may also use regexes in the exclude list. They must start with `^` and end with `$`.
   *
   * @param columns Column(s) to exclude from selection
   * @example
   * ```
   *  >>> df = pl.DataFrame({
   *  ...   "a": [1, 2, 3],
   *  ...   "b": ["a", "b", None],
   *  ...   "c": [None, 2, 1],
   *  ...})
   *  >>> df
   *  shape: (3, 3)
   *  ╭─────┬──────┬──────╮
   *  │ a   ┆ b    ┆ c    │
   *  │ --- ┆ ---  ┆ ---  │
   *  │ i64 ┆ str  ┆ i64  │
   *  ╞═════╪══════╪══════╡
   *  │ 1   ┆ "a"  ┆ null │
   *  ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   *  │ 2   ┆ "b"  ┆ 2    │
   *  ├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   *  │ 3   ┆ null ┆ 1    │
   *  ╰─────┴──────┴──────╯
   *  >>> df.select(
   *  ...   pl.col("*").exclude("b"),
   *  ... )
   * shape: (3, 2)
   * ╭─────┬──────╮
   * │ a   ┆ c    │
   * │ --- ┆ ---  │
   * │ i64 ┆ i64  │
   * ╞═════╪══════╡
   * │ 1   ┆ null │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 2   ┆ 2    │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 3   ┆ 1    │
   * ╰─────┴──────╯
   * ```
   */
  exclude(column: string, ...columns: string[]): Expr
  /**
   * Explode a list or utf8 Series.
   *
   * This means that every item is expanded to a new row.
   */
  explode(): Expr
  /** Fill nan value with a fill value */
  fillNan(other: any): Expr
  /** Fill null value with a fill value or strategy */
  fillNull(other: any | FillNullStrategy): Expr
  /**
   * Filter a single column.
   *
   * Mostly useful in in aggregation context.
   * If you want to filter on a DataFrame level, use `LazyFrame.filter`.
   * @param predicate Boolean expression.
   */
  filter(predicate: Expr): Expr
  /** Get the first value. */
  first(): Expr
  /** @see {@link Expr.explode} */
  flatten(): Expr
  /**
   * Floor underlying floating point array to the lowest integers smaller or equal to the float value.
   *
   * Only works on floating point Series
   */
  floor(): Expr
  /** Fill missing values with the latest seen values */
  forwardFill(): Expr
  gt(other: any): Expr
  gtEq(other: any): Expr
  /** Hash the Series. */
  hash(k0?: number, k1?: number, k2?: number, k3?: number): Expr
  hash({k0, k1, k2, k3}: {k0?: number, k1?: number, k2?: number, k3?: number}): Expr
  /** Take the first n values.  */
  head(length?: number): Expr
  head({length}: {length: number}): Expr
  /** Interpolate intermediate values. The interpolation method is linear. */
  interpolate(): Expr
  /** Get mask of duplicated values. */
  isDuplicated(): Expr
  /** Create a boolean expression returning `true` where the expression values are finite. */
  isFinite(): Expr
  /** Get a mask of the first unique value. */
  isFirst(): Expr
  /**
   * Check if elements of this Series are in the right Series, or List values of the right Series.
   *
   * @param other Series of primitive type or List type.
   * @returns Expr that evaluates to a Boolean Series.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...   "sets": [[1, 2, 3], [1, 2], [9, 10]],
   * ...    "optional_members": [1, 2, 3]
   * ... })
   * >>> df.select(
   * ...   pl.col("optional_members").isIn("sets").alias("contains")
   * ... )
   * shape: (3, 1)
   * ┌──────────┐
   * │ contains │
   * │ ---      │
   * │ bool     │
   * ╞══════════╡
   * │ true     │
   * ├╌╌╌╌╌╌╌╌╌╌┤
   * │ true     │
   * ├╌╌╌╌╌╌╌╌╌╌┤
   * │ false    │
   * └──────────┘
   * ```
   */
  isIn(other): Expr
  /** Create a boolean expression returning `true` where the expression values are infinite. */
  isInfinite(): Expr
  /** Create a boolean expression returning `true` where the expression values are NaN (Not A Number). */
  isNan(): Expr
  /** Create a boolean expression returning `true` where the expression values are not NaN (Not A Number). */
  isNotNan(): Expr
  /** Create a boolean expression returning `true` where the expression does not contain null values. */
  isNotNull(): Expr
  /** Create a boolean expression returning `True` where the expression contains null values. */
  isNull(): Expr
  /** Get mask of unique values. */
  isUnique(): Expr
  /**
   *  Keep the original root name of the expression.
   *
   * A groupby aggregation often changes the name of a column.
   * With `keepName` we can keep the original name of the column
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...   "a": [1, 2, 3],
   * ...   "b": ["a", "b", None],
   * ... })
   *
   * >>> df
   * ...   .groupBy("a")
   * ...   .agg(pl.col("b").list())
   * ...   .sort({by:"a"})
   *
   * shape: (3, 2)
   * ╭─────┬────────────╮
   * │ a   ┆ b_agg_list │
   * │ --- ┆ ---        │
   * │ i64 ┆ list [str] │
   * ╞═════╪════════════╡
   * │ 1   ┆ [a]        │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 2   ┆ [b]        │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 3   ┆ [null]     │
   * ╰─────┴────────────╯
   *
   * Keep the original column name:
   *
   * >>> df
   * ...   .groupby("a")
   * ...   .agg(col("b").list().keepName())
   * ...   .sort({by:"a"})
   *
   * shape: (3, 2)
   * ╭─────┬────────────╮
   * │ a   ┆ b          │
   * │ --- ┆ ---        │
   * │ i64 ┆ list [str] │
   * ╞═════╪════════════╡
   * │ 1   ┆ [a]        │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 2   ┆ [b]        │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 3   ┆ [null]     │
   * ╰─────┴────────────╯
   * ```
   */
  keepName(): Expr
  kurtosis(): Expr
  kurtosis(fisher: boolean, bias?: boolean): Expr
  kurtosis({fisher, bias}: {fisher?: boolean, bias?: boolean}): Expr
  /** Get the last value.  */
  last(): Expr
  /** Aggregate to list. */
  list(): Expr
  /** Returns a unit Series with the lowest value possible for the dtype of this expression. */
  lowerBound(): Expr
  lt(other: any): Expr
  ltEq(other: any): Expr
  /** Compute the max value of the arrays in the list */
  max(): Expr
  /** Compute the mean value of the arrays in the list */
  mean(): Expr
  /** Get median value. */
  median(): Expr
  /** Get minimum value. */
  min(): Expr
  /** Compute the most occurring value(s). Can return multiple Values */
  mode(): Expr
  neq(other: any): Expr
  /** Negate a boolean expression. */
  not(): Expr
  /** Count unique values. */
  nUnique(): Expr
  or(other: any): Expr
  /**
   * Apply window function over a subgroup.
   *
   * This is similar to a groupby + aggregation + self join.
   * Or similar to [window functions in Postgres](https://www.postgresql.org/docs/9.1/tutorial-window.html)
   * @param partitionBy Column(s) to partition by.
   *
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...  "groups": [1, 1, 2, 2, 1, 2, 3, 3, 1],
   * ...  "values": [1, 2, 3, 4, 5, 6, 7, 8, 8],
   * ... })
   * >>> df.select(
   * ...     pl.col("groups").sum().over("groups")
   * ... )
   *     ╭────────┬────────╮
   *     │ groups ┆ values │
   *     │ ---    ┆ ---    │
   *     │ i32    ┆ i32    │
   *     ╞════════╪════════╡
   *     │ 1      ┆ 16     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 1      ┆ 16     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 2      ┆ 13     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 2      ┆ 13     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ ...    ┆ ...    │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 1      ┆ 16     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 2      ┆ 13     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 3      ┆ 15     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 3      ┆ 15     │
   *     ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   *     │ 1      ┆ 16     │
   *     ╰────────┴────────╯
   * ```
   */
  over(by: ExprOrString, ...partitionBy: ExprOrString[]): Expr
  /** Raise expression to the power of exponent. */
  pow(exponent: number): Expr
  pow({exponent}: {exponent: number}): Expr
  /**
   * Add a prefix the to root column name of the expression.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * ...   "A": [1, 2, 3, 4, 5],
   * ...   "fruits": ["banana", "banana", "apple", "apple", "banana"],
   * ...   "B": [5, 4, 3, 2, 1],
   * ...   "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
   * ... })
   * shape: (5, 4)
   * ╭─────┬──────────┬─────┬──────────╮
   * │ A   ┆ fruits   ┆ B   ┆ cars     │
   * │ --- ┆ ---      ┆ --- ┆ ---      │
   * │ i64 ┆ str      ┆ i64 ┆ str      │
   * ╞═════╪══════════╪═════╪══════════╡
   * │ 1   ┆ "banana" ┆ 5   ┆ "beetle" │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
   * │ 2   ┆ "banana" ┆ 4   ┆ "audi"   │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
   * │ 3   ┆ "apple"  ┆ 3   ┆ "beetle" │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
   * │ 4   ┆ "apple"  ┆ 2   ┆ "beetle" │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
   * │ 5   ┆ "banana" ┆ 1   ┆ "beetle" │
   * ╰─────┴──────────┴─────┴──────────╯
   * >>> df.select(
   * ...   pl.all().reverse().prefix("reverse_"),
   * ... )
   * shape: (5, 8)
   * ╭───────────┬────────────────┬───────────┬──────────────╮
   * │ reverse_A ┆ reverse_fruits ┆ reverse_B ┆ reverse_cars │
   * │ ---       ┆ ---            ┆ ---       ┆ ---          │
   * │ i64       ┆ str            ┆ i64       ┆ str          │
   * ╞═══════════╪════════════════╪═══════════╪══════════════╡
   * │ 5         ┆ "banana"       ┆ 1         ┆ "beetle"     │
   * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 4         ┆ "apple"        ┆ 2         ┆ "beetle"     │
   * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 3         ┆ "apple"        ┆ 3         ┆ "beetle"     │
   * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 2         ┆ "banana"       ┆ 4         ┆ "audi"       │
   * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 1         ┆ "banana"       ┆ 5         ┆ "beetle"     │
   * ╰───────────┴────────────────┴───────────┴──────────────╯
   * ```
   */
  prefix(prefix: string): Expr
  /** Get quantile value. */
  quantile(quantile: number): Expr
  /** Assign ranks to data, dealing with ties appropriately. */
  rank(method?: RankMethod): Expr
  rank({method}: {method: string}): Expr
  reinterpret(signed?: boolean): Expr
  reinterpret({signed}: {signed: boolean}): Expr
  /**
   * Repeat the elements in this Series `n` times by dictated by the number given by `by`.
   * The elements are expanded into a `List`
   * @param by Numeric column that determines how often the values will be repeated.
   *
   * The column will be coerced to UInt32. Give this dtype to make the coercion a no-op.
   */
  repeatBy(by: Expr | string): Expr
  /** Reverse the arrays in the list */
  reverse(): Expr
    /**
   * __Apply a rolling max (moving max) over the values in this Series.__
   *
   * A window of length `window_size` will traverse the series. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector.
   *
   * The resulting parameters' values will be aggregated into their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   * @see {@link rollingMean}, {@link rollingMin}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMax(options: RollingOptions): Expr
  rollingMax(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): Expr
  /**
   * __Apply a rolling mean (moving mean) over the values in this Series.__
   *
   * A window of length `window_size` will traverse the series. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector.
   *
   * The resulting parameters' values will be aggregated into their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   * @see {@link rollingMax}, {@link rollingMin}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMean(options: RollingOptions): Expr
  rollingMean(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): Expr
  /**
   * __Apply a rolling min (moving min) over the values in this Series.__
   *
   * A window of length `window_size` will traverse the series. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector.
   *
   * The resulting parameters' values will be aggregated into their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   * @see {@link rollingMax}, {@link rollingMean}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMin(options: RollingOptions): Expr
  rollingMin(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): Expr
  /**
   * __Apply a rolling sum (moving sum) over the values in this Series.__
   *
   * A window of length `window_size` will traverse the series. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector.
   *
   * The resulting parameters' values will be aggregated into their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   */
  rollingSum(options: RollingOptions): Expr
  rollingSum(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): Expr
  /**
   * __Compute a rolling variance.__
   *
   * A window of length `window_size` will traverse the series. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector.
   *
   * The resulting parameters' values will be aggregated into their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   * @see {@link rollingMax}, {@link rollingMin}, {@link rollingMean}, {@link rollingSum}
   */
  rollingVar(options: RollingOptions): Expr
  rollingVar(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): Expr
  /** Compute a rolling median */
  rollingMedian(windowSize: number): Expr
  rollingMedian({windowSize}: {windowSize: number}): Expr
  /**
   * Compute a rolling quantile
   * @param windowSize Size of the rolling window
   * @param quantile quantile to compute
   */
  rollingQuantile(windowSize: number, quantile: number): Expr
  rollingQuantile({windowSize, quantile}: {windowSize: number, quantile: number}): Expr
  /**
   * Compute a rolling skew
   * @param windowSize Size of the rolling window
   * @param bias If false, then the calculations are corrected for statistical bias.
   */
  rollingSkew(windowSize: number, bias?: boolean): Expr
  rollingSkew({windowSize, bias}: {windowSize: number, bias?: boolean}): Expr
  /**
   * Round underlying floating point data by `decimals` digits.
   * @param decimals Number of decimals to round by.
   */
  round(decimals?: number): Expr
  round({decimals}: {decimals: number}): Expr
  /**
   * Shift the values by a given period and fill the parts that will be empty due to this operation
   * @param periods number of places to shift (may be negative).
   */
  shift(periods?: number): Expr
  shift({periods}: {periods: number}): Expr
  /**
   * Shift the values by a given period and fill the parts that will be empty due to this operation
   * @param periods Number of places to shift (may be negative).
   * @param fillValue Fill null values with the result of this expression.
   */
  shiftAndFill(periods: number, fillValue: Expr): Expr
  shiftAndFill({periods, fillValue}: {periods: number, fillValue: Expr}): Expr
  /**
   * Compute the sample skewness of a data set.
   * For normally distributed data, the skewness should be about zero. For
   * unimodal continuous distributions, a skewness value greater than zero means
   * that there is more weight in the right tail of the distribution.
   * ___
   * @param bias If False, then the calculations are corrected for statistical bias.
   */
  skew(bias?: boolean): Expr
  skew({bias}: {bias: boolean}): Expr
  /** Slice the Series. */
  slice(offset: number, length: number): Expr
  slice({offset, length}: {offset: number, length: number}): Expr
  /**
   * Sort this column. In projection/ selection context the whole column is sorted.
   * @param reverse
   * * false -> order from small to large.
   * * true -> order from large to small.
   * @param nullsLast If true nulls are considered to be larger than any valid value
   */
  sort(reverse?: boolean, nullsLast?: boolean): Expr
  sort({reverse, nullsLast}: {reverse?: boolean, nullsLast?: boolean}): Expr
  /** Get standard deviation. */
  std(): Expr
  /** Add a suffix the to root column name of the expression. */
  suffix(suffix: string): Expr
  /**
   * Get sum value.
   * @note
   * Dtypes in {Int8, UInt8, Int16, UInt16} are cast to Int64 before summing to prevent overflow issues.
   */
  sum(): Expr
  /** Take the last n values. */
  tail(length?: number): Expr
  tail({length}: {length: number}): Expr
  /**
   * Take values by index.
   * @param index An expression that leads to a UInt32 dtyped Series.
   */
  take(index: Expr | number[] | Series<number>): Expr
  take({index}: {index: Expr | number[] | Series<number>}): Expr
  /** Take every nth value in the Series and return as a new Series. */
  takeEvery(n: number): Expr
  /** Get the unique/distinct values in the list */
  unique(): Expr
  /** Returns a unit Series with the highest value possible for the dtype of this expression. */
  upperBound(): Expr
  /** Get variance. */
  var(): Expr
  /** Alais for filter: @see {@link filter} */
  where(predicate: Expr): Expr
}

const ExprListFunctions = (_expr: JsExpr): ExprListFunctions => {
  const wrap = (method, args?): Expr => {

    return Expr(pli.expr.lst[method]({_expr, ...args }));
  };

  return {
    get(index: number) {
      return wrap("get", {index});
    },
    first() {
      return wrap("get", {index:0});
    },
    last() {
      return wrap("get", {index:-1});
    },
    lengths() {
      return wrap("lengths");
    },
    max() {
      return wrap("max");
    },
    mean() {
      return wrap("mean");
    },
    min() {
      return wrap("min");
    },
    reverse() {
      return wrap("reverse");
    },
    sort(reverse: any = false) {
      return typeof reverse === "boolean" ?
        wrap("sort", {reverse}) :
        wrap("sort", reverse);
    },
    sum() {
      return wrap("sum");
    },
    unique() {
      return wrap("unique");
    },
  };
};

const ExprDateTimeFunctions = (_expr: JsExpr): ExprDateTimeFunctions => {
  const wrap = (method, args?): Expr => {
    return Expr(pli.expr.date[method]({_expr, ...args }));
  };

  const wrapNullArgs = (method: string) => () => wrap(method);

  return {
    day: wrapNullArgs("day"),
    hour: wrapNullArgs("hour"),
    minute: wrapNullArgs("minute"),
    month: wrapNullArgs("month"),
    nanosecond: wrapNullArgs("nanosecond"),
    ordinalDay: wrapNullArgs("ordinalDay"),
    second: wrapNullArgs("second"),
    strftime: (fmt) => wrap("strftime", {fmt}),
    timestamp: wrapNullArgs("timestamp"),
    week: wrapNullArgs("week"),
    weekday: wrapNullArgs("weekday"),
    year: wrapNullArgs("year"),
  };
};

const ExprStringFunctions = (_expr: JsExpr): ExprStringFunctions => {
  const wrap = (method, args?): Expr => {

    return Expr(pli.expr.str[method]({_expr, ...args }));
  };

  return {
    concat(delimiter: string) {
      return wrap("concat", {delimiter});
    },
    contains(pat: string | RegExp) {
      return wrap("contains", {pat: regexToString(pat)});
    },
    extract(pat: string | RegExp, groupIndex: number) {
      return wrap("extract", {pat: regexToString(pat), groupIndex});
    },
    jsonPathMatch(pat: string) {
      return wrap("jsonPathMatch", {pat});
    },
    lengths() {
      return wrap("lengths");
    },
    lstrip() {
      return wrap("replace", {pat: /^\s*/.source, val: ""});
    },
    replace(pat: RegExp, val: string) {
      return wrap("replace", {pat: regexToString(pat), val});
    },
    replaceAll(pat: RegExp, val: string) {
      return wrap("replaceAll", {pat: regexToString(pat), val});
    },
    rstrip() {
      return wrap("replace", {pat: /[ \t]+$/.source, val: ""});
    },
    slice(start: number, length?: number) {
      return wrap("slice", {start, length});
    },
    strftime(dtype, fmt?) {
      if (dtype === DataType.Date) {
        return wrap("parseDate", {fmt});
      } else if (dtype === DataType.Datetime) {
        return wrap("parseDateTime", {fmt});
      } else {
        throw new Error(`only "DataType.Date" and "DataType.Datetime" are supported`);
      }
    },
    toLowerCase() {
      return wrap("toLowerCase");
    },
    toUpperCase() {
      return wrap("toUpperCase");
    },
  };
};

export const Expr = (_expr: JsExpr): Expr => {

  const wrap = (method, args?): Expr => {

    return Expr(pli.expr[method]({_expr, ...args }));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);
  const wrapExprArg = (method: string, lit=false) => (other: any) => {

    const expr = exprToLitOrExpr(other, lit)._expr;

    return wrap(method, {other: expr});
  };
  const wrapUnary = (method: string, key: string) => (val) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapUnaryWithDefault = (method: string, key: string, otherwise) => (val=otherwise) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapBinary = (method: string, key0: string, key1: string) => (val0, val1) => {
    if(val0[key0] !== undefined) {
      return wrap(method, val0);
    }

    return wrap(
      method, {
        [key0]: val0,
        [key1]: val1
      }
    );
  };
  const wrapUnaryNumber = (method: string, key: string) => {
    const f = (val) => {
      if(typeof val === "number") {

        return f({[key]: val});
      }

      return wrap(method, val);
    };

    return f;
  };
  const exclude = (column, ...columns) => {
    return wrap("exclude", {columns: [column, ...columns]});
  };
  const fillNull = (fillValue) => {
    if(["backward", "forward", "mean", "min", "max", "zero", "one"].includes(fillValue)) {
      return wrap("fillNullWithStrategy", {strategy: fillValue});
    }

    const expr = exprToLitOrExpr(fillValue)._expr;

    return wrap("fillNull", {other: expr});
  };
  const isIn = (other) => {
    if(Array.isArray(other)) {
      other = lit(Series(other));
    } else {
      other = exprToLitOrExpr(other, false);
    }

    return wrap("isIn", {other: other._expr});
  };
  const kurtosis = (obj?, bias=true) => {
    return wrap("kurtosis", {
      fisher: obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true),
      bias : obj?.["bias"] ?? bias,
    });
  };
  const hash = (obj: any=0, k1=1, k2=2, k3=3) => {
    if(typeof obj === "number" || typeof obj === "bigint") {
      return wrap("hash", {
        k0: obj,
        k1: k1,
        k2: k2,
        k3: k3
      });
    }

    return wrap("hash", obj);
  };
  const over = (...exprs) => {

    const partitionBy = selectionToExprList(exprs, false);

    return wrap("over", {partitionBy});
  };
  const shiftAndFill = (opt, fillValue?) => {
    if(opt?.periods !== undefined) {
      return shiftAndFill(opt.periods, opt.fillValue);
    }
    fillValue = exprToLitOrExpr(fillValue, true)._expr;

    return wrap("shiftAndFill", {periods: opt, fillValue});
  };
  const rollingSkew = (val, bias=true) => {
    if(typeof val === "number") {
      return wrap("rollingSkew", {windowSize: val, bias});
    }

    return rollingSkew(val.windowSize, val.bias);
  };
  const sort = (reverse: any = false, nullsLast=false) => {
    if(typeof reverse === "boolean") {
      return wrap("sortWith", {reverse, nullsLast});
    }

    return wrap("sortWith", reverse);
  };

  const take = (indices) => {
    if(Array.isArray(indices)) {
      indices = lit(Series(indices));
    }

    return wrap("take", {other: indices._expr});
  };

  return {
    _expr,
    [inspect]() { return pli.expr.as_str({_expr});},
    get str() { return ExprStringFunctions(_expr);},
    get lst() {return ExprListFunctions(_expr);},
    get date() {return ExprDateTimeFunctions(_expr);},
    abs: wrapNullArgs("abs"),
    aggGroups: wrapNullArgs("aggGroups"),
    alias: wrapUnary("alias", "name"),
    and: wrapExprArg("and"),
    argMax: wrapNullArgs("argMax"),
    argMin: wrapNullArgs("argMin"),
    argSort: wrapUnaryWithDefault("argSort", "reverse", false),
    argUnique: wrapNullArgs("argUnique"),
    as: wrapUnary("alias", "name"),
    backwardFill: wrapNullArgs("backwardFill"),
    cast: (dtype, strict=false) => wrap("cast", {dtype, strict}),
    count: wrapNullArgs("count"),
    cumCount: wrapUnaryWithDefault("cumCount", "reverse", false),
    cumMax: wrapUnaryWithDefault("cumMax", "reverse", false),
    cumMin: wrapUnaryWithDefault("cumMin", "reverse", false),
    cumProd: wrapUnaryWithDefault("cumProd", "reverse", false),
    cumSum: wrapUnaryWithDefault("cumSum", "reverse", false),
    diff: wrapBinary("diff", "n", "nullBehavior"),
    dot: wrapExprArg("dot"),
    eq: wrapExprArg("eq"),
    exclude,
    explode: wrapNullArgs("explode"),
    fillNan: wrapExprArg("fillNan", true),
    fillNull,
    fillNullWithStrategy: wrapUnary("fillNullWithStrategy", "strategy"),
    filter: wrapExprArg("filter"),
    first: wrapNullArgs("first"),
    flatten: wrapNullArgs("explode"),
    floor: wrapNullArgs("floor"),
    forwardFill: wrapNullArgs("forwardFill"),
    gt: wrapExprArg("gt"),
    gtEq: wrapExprArg("gtEq"),
    hash,
    head: wrapUnaryNumber("head", "length"),
    interpolate: wrapNullArgs("interpolate"),
    isDuplicated: wrapNullArgs("isDuplicated"),
    isFinite: wrapNullArgs("isFinite"),
    isFirst: wrapNullArgs("isFirst"),
    isIn,
    isInfinite: wrapNullArgs("isInfinite"),
    isNan: wrapNullArgs("isNan"),
    isNotNan: wrapNullArgs("isNotNan"),
    isNotNull: wrapNullArgs("isNotNull"),
    isNull: wrapNullArgs("isNull"),
    isUnique: wrapNullArgs("isUnique"),
    keepName: wrapNullArgs("keepName"),
    kurtosis,
    last: wrapNullArgs("last"),
    list: wrapNullArgs("list"),
    lowerBound: wrapNullArgs("lowerBound"),
    lt: wrapExprArg("lt"),
    ltEq: wrapExprArg("ltEq"),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    median: wrapNullArgs("median"),
    min: wrapNullArgs("min"),
    mode: wrapNullArgs("mode"),
    neq: wrapExprArg("neq"),
    not: wrapNullArgs("not"),
    nUnique: wrapNullArgs("nUnique"),
    or: wrapExprArg("or"),
    over,
    pow: wrapUnary("pow", "exponent"),
    prefix: wrapUnary("prefix", "prefix"),
    quantile: wrapUnary("quantile", "quantile"),
    rank: wrapUnary("rank", "method"),
    reinterpret: wrapUnaryWithDefault("reinterpret", "signed", true),
    repeatBy: wrapExprArg("repeatBy"),
    reshape: wrapUnary("reshape", "dims"),
    reverse: wrapNullArgs("reverse"),
    rollingMedian: wrapUnary("rollingMedian", "windowSize"),
    rollingQuantile: wrapBinary("rollingQuantile", "windowSize", "quantile"),
    rollingSkew,
    round: wrapUnaryNumber("round", "decimals"),
    shift: wrapUnary("shift", "periods"),
    shiftAndFill,
    skew: wrapUnaryWithDefault("skew", "bias", true),
    slice: wrapBinary("slice", "offset", "length"),
    sort,
    std: wrapNullArgs("std"),
    suffix: wrapUnary("suffix", "suffix"),
    sum: wrapNullArgs("sum"),
    tail: wrapUnaryNumber("tail", "length"),
    take,
    takeEvery: wrapUnary("takeEvery", "n"),
    unique: wrapNullArgs("unique"),
    upperBound: wrapNullArgs("upperBound"),
    where: wrapExprArg("filter"),
    var: wrapNullArgs("var")
  } as any;
};

export const exprToLitOrExpr = (expr: any, stringToLit = true): Expr  => {
  if(typeof expr === "string" && !stringToLit) {
    return col(expr);
  } else if (isExpr(expr)) {
    return expr;
  } else if (Array.isArray(expr)) {
    return expr.map(e => exprToLitOrExpr(e, stringToLit)) as any;
  } else {
    return lit(expr);
  }
};
