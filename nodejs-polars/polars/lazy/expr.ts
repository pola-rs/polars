import {DataType} from "../datatypes";
import pli from "../internals/polars_internal";
import {col, lit} from "./lazy_functions";
import {ColumnSelection, ExpressionSelection, FillNullStrategy, isExpr, RankMethod, selectionToExprList} from "../utils";
import {Series} from "../series";
const inspect = Symbol.for("nodejs.util.inspect.custom");

type JsExpr = any;
type ColumnsOrExpr = ColumnSelection | ExpressionSelection

export interface ExprListFunctions {
  lengths(): Expr;
  max(): Expr;
  mean(): Expr;
  min(): Expr;
  reverse(): Expr;
  sort(reverse?: boolean): Expr;
  sum(): Expr;
  unique(): Expr;
}

export interface ExprStringFunctions {
  concat(delimiter: string): Expr;
  contains(pat: RegExp): Expr;
  extract(pat: RegExp, groupIndex: number): Expr;
  jsonPathMatch(pat: string): Expr;
  lengths(): Expr;
  parseDate(fmt?: string): Expr;
  parseDateTime(fmt?: string): Expr;
  replace(pat: RegExp, val: string): Expr;
  replaceAll(pat: RegExp, val: string): Expr;
  toLowercase(): Expr;
  toUppercase(): Expr;
  slice(start: number, length?: number): Expr;
}

interface ExprDateTimeFunctions {
  day(): Expr;
  hour(): Expr;
  minute(): Expr;
  month(): Expr;
  nanosecond(): Expr;
  ordinalDay(): Expr;
  second(): Expr;
  strftime(fmt: string): Expr;
  timestamp(): Expr;
  week(): Expr;
  weekday(): Expr;
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
  alias({name}: {name: string}): Expr
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
  as({name}: {name: string}): Expr
  as(name: string): Expr
  /** Fill missing values with the next to be seen values */
  backwardFill(): Expr
  /** Cast between data types. */
  cast(dtype:DataType, strict?: boolean): Expr
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
  diff(n:number, nullBehavior: "ignore" | "drop"): Expr
  diff(o: {n:number, nullBehavior: "ignore" | "drop"}): Expr
  /**
   * Compute the dot/inner product between two Expressions
   * @param other Expression to compute dot product with
   */
  dot(other: any | string): Expr
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
  exclude(...columns: ColumnSelection[]): Expr
  exclude({columns}: {columns: string[]}): Expr
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
  hash({k0, k1, k2, k3}: {k0?:number, k1?: number, k2?: number, k3?:number}): Expr
  /** Take the first n values.  */
  head(length?: number): Expr
  head({length}: {length: number}): Expr
  /** Interpolate intermediate values. The interpolation method is linear. */
  interpolate(): Expr
  isDuplicated(): Expr
  isFinite(): Expr
  isFirst(): Expr
  isIn(other: any): Expr
  isInfinite(): Expr
  isNan(): Expr
  isNotNan(): Expr
  isNotNull(): Expr
  isNull(): Expr
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
   * ...   .groupby("a")
   * ...   .agg(pl.col("b").list())
   * ...   .sort(by="a")
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
  kurtosis(fisher:boolean, bias?:boolean): Expr
  kurtosis({fisher, bias}:{fisher?:boolean, bias?:boolean}): Expr
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
   * >>> df.lazy().select(
   * ...     pl.col("groups").sum("values").over("groups")
   * ... )
   * ... .collectSync()
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
  over(...partitionBy: ColumnsOrExpr[]): Expr

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
  slice(offset:number, length:number): Expr
  slice({offset, length}: {offset:number, length:number}): Expr

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
  xor(other: any): Expr
}

const ExprListFunctions = (_expr: JsExpr): ExprListFunctions => {
  const wrap = <U>(method, args?): Expr => {

    return Expr(pli.expr.lst[method]({_expr, ...args }));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);

  return {
    lengths: wrapNullArgs("lengths"),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    min: wrapNullArgs("min"),
    reverse: wrapNullArgs("reverse"),
    sort: (reverse=false) => wrap("sort", {reverse: reverse?.["reverse"] ?? reverse}),
    sum: wrapNullArgs("sum"),
    unique: wrapNullArgs("unique"),
  };
};

const ExprDateTimeFunctions = (_expr: JsExpr): ExprDateTimeFunctions => {
  const wrap = <U>(method, args?): Expr => {

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

export const Expr = (_expr: JsExpr): Expr => {

  const wrap = <U>(method, args?): Expr => {

    return Expr(pli.expr[method]({_expr, ...args }));
  };

  const wrapNullArgs = (method: string) => () => wrap(method);
  const wrapExprArg = (method: string) => (other: any) => wrap(method, {other: exprToLitOrExpr(other)._expr});
  const wrapUnary = (method: string, key: string) => (val) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapUnaryWithDefault = (method: string, key: string, otherwise) => (val=otherwise) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapBinary = (method: string, key0: string, key1: string) => (val0, val1) => wrap(
    method, {
      [key0]: val0?.[key0] ?? val0,
      [key1]: val1?.[key1] ?? val1
    }
  );
  const kurtosis = (obj?, bias=true) => {
    return wrap("kurtosis", {
      fisher: obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true),
      bias : obj?.["bias"] ?? bias,
    });
  };

  const hash = (obj?: object | number, k1=1, k2=2, k3=3) => {
    return wrap<bigint>("hash", {
      k0: obj?.["k0"] ?? (typeof obj === "number" ? obj : 0),
      k1: obj?.["k1"] ?? k1,
      k2: obj?.["k2"] ?? k2,
      k3: obj?.["k3"] ?? k3
    });
  };
  const over = (...exprs) => {

    const partitionBy = selectionToExprList(exprs);

    return wrap("over", {partitionBy});
  };
  const shiftAndFill = (opt, fillValue?) => {
    if(opt?.periods !== undefined) {
      return shiftAndFill(opt.periods, opt.fillValue);
    }
    fillValue = exprToLitOrExpr(fillValue, true)._expr;

    return wrap("shiftAndFill", {periods: opt, fillValue});
  };

  return {
    _expr,
    [inspect]() { return pli.expr.as_str({_expr});},
    get lst() {return ExprListFunctions(_expr);},
    get date() {return ExprDateTimeFunctions(_expr);},
    abs: wrapNullArgs("abs"),
    aggGroups: wrapNullArgs("aggGroups"),
    argMax: wrapNullArgs("argMax"),
    argMin: wrapNullArgs("argMin"),
    argUnique: wrapNullArgs("argUnique"),
    backwardFill: wrapNullArgs("backwardFill"),
    count: wrapNullArgs("count"),
    explode: wrapNullArgs("explode"),
    first: wrapNullArgs("first"),
    floor: wrapNullArgs("floor"),
    forwardFill: wrapNullArgs("forwardFill"),
    interpolate: wrapNullArgs("interpolate"),
    isDuplicated: wrapNullArgs("isDuplicated"),
    isFinite: wrapNullArgs("isFinite"),
    isFirst: wrapNullArgs("isFirst"),
    isInfinite: wrapNullArgs("isInfinite"),
    isNan: wrapNullArgs("isNan"),
    isNotNan: wrapNullArgs("isNotNan"),
    isNotNull: wrapNullArgs("isNotNull"),
    isNull: wrapNullArgs("isNull"),
    isUnique: wrapNullArgs("isUnique"),
    keepName: wrapNullArgs("keepName"),
    last: wrapNullArgs("last"),
    list: wrapNullArgs("list"),
    lowerBound: wrapNullArgs("lowerBound"),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    median: wrapNullArgs("median"),
    min: wrapNullArgs("min"),
    mode: wrapNullArgs("mode"),
    not: wrapNullArgs("not"),
    nUnique: wrapNullArgs("nUnique"),
    reverse: wrapNullArgs("reverse"),
    std: wrapNullArgs("std"),
    sum: wrapNullArgs("sum"),
    unique: wrapNullArgs("unique"),
    upperBound: wrapNullArgs("upperBound"),
    var: wrapNullArgs("var"),
    and: wrapExprArg("and"),
    dot: wrapExprArg("dot"),
    eq: wrapExprArg("eq"),
    fillNan: wrapExprArg("fillNan"),
    fillNull: wrapExprArg("fillNull"),
    filter: wrapExprArg("filter"),
    gt: wrapExprArg("gt"),
    gtEq: wrapExprArg("gtEq"),
    isIn: wrapExprArg("isIn"),
    lt: wrapExprArg("lt"),
    ltEq: wrapExprArg("ltEq"),
    neq: wrapExprArg("neq"),
    or: wrapExprArg("or"),
    repeatBy: wrapExprArg("repeatBy"),
    take: wrapExprArg("take"),
    xor: wrapExprArg("xor"),
    rollingMedian: wrapUnary("rollingMedian", "windowSize"),
    pow: wrapUnary("pow", "exponent"),
    quantile: wrapUnary("quantile", "quantile"),
    shift: wrapUnary("shift", "periods"),
    suffix: wrapUnary("suffix", "suffix"),
    alias: wrapUnary("alias", "name"),
    prefix: wrapUnary("prefix", "prefix"),
    rank: wrapUnary("rank", "method"),
    fillNullWithStrategy: wrapUnary("fillNullWithStrategy", "strategy"),
    takeEvery: wrapUnary("takeEvery", "n"),
    exclude: wrapUnary("exclude", "columns"),
    over,
    reshape: wrapUnary("reshape", "dims"),
    argSort: wrapUnaryWithDefault("argSort", "reverse", false),
    cumCount: wrapUnaryWithDefault("cumCount", "reverse", false),
    cumMax: wrapUnaryWithDefault("cumMax", "reverse", false),
    cumMin: wrapUnaryWithDefault("cumMin", "reverse", false),
    cumProd: wrapUnaryWithDefault("cumProd", "reverse", false),
    cumSum: wrapUnaryWithDefault("cumSum", "reverse", false),
    sort: wrapUnaryWithDefault("sort", "reverse", false),
    reinterpret: wrapUnaryWithDefault("reinterpret", "signed", true),
    diff: wrapBinary("diff", "n", "nullBehavior"),
    slice: wrapBinary("slice", "offset", "length"),
    rollingQuantile: wrapBinary("rollingQuantile", "windowSize", "quantile"),
    cast: (dtype, strict=false) => wrap("cast", {dtype, strict}),
    rollingSkew: (windowSize, bias=false) => wrap("rollingSkew", {windowSize, bias}),
    shiftAndFill,
    kurtosis,
    hash
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