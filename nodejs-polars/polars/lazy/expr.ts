import {DataType} from "../datatypes";
import pli from "../internals/polars_internal";
import {
  ExprOrString,
  FillNullStrategy,
  RankMethod,
  selectionToExprList,
  INSPECT_SYMBOL
} from "../utils";
import {Series} from "../series/series";

import * as expr from "./expr/";
import {Arithmetic, Comparison, Cumulative, Deserialize, Rolling, Round, Sample, Serialize} from "../shared_traits";

export interface Expr extends
  Rolling<Expr>,
  Arithmetic<Expr>,
  Comparison<Expr>,
  Cumulative<Expr>,
  Sample<Expr>,
  Round<Expr>,
  Serialize {
  /** @ignore */
  _expr: any;
  get date(): expr.Datetime;
  get str(): expr.String;
  get lst(): expr.List;
  get struct(): expr.Struct;
  [Symbol.toStringTag](): string;
  [INSPECT_SYMBOL](): string;
  toString(): string;
  /** compat with `JSON.stringify` */
  toJSON(): string;
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
  /**
   * Extend the Series with given number of values.
   * @param value The value to extend the Series with. This value may be null to fill with nulls.
   * @param n The number of values to extend.
   * @deprecated
   * @see {@link extendConstant}
   */
  extend(value: any, n: number): Expr
  extend(opt: {value: any, n: number}): Expr
  /**
   * Extend the Series with given number of values.
   * @param value The value to extend the Series with. This value may be null to fill with nulls.
   * @param n The number of values to extend.
   */
  extendConstant(value: any, n: number): Expr
  extendConstant(opt: {value: any, n: number}): Expr
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
  /** Fill missing values with the latest seen values */
  forwardFill(): Expr
  /** Hash the Series. */
  hash(k0?: number, k1?: number, k2?: number, k3?: number): Expr
  hash({k0, k1, k2, k3}: {k0?: number, k1?: number, k2?: number, k3?: number}): Expr
  /** Take the first n values.  */
  head(length?: number): Expr
  head({length}: {length: number}): Expr
  inner(): any
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
  slice(offset: number | Expr, length: number | Expr): Expr
  slice({offset, length}: {offset: number | Expr, length: number | Expr}): Expr
  /**
   * Sort this column. In projection/ selection context the whole column is sorted.
   * @param reverse
   * * false -> order from small to large.
   * * true -> order from large to small.
   * @param nullsLast If true nulls are considered to be larger than any valid value
   */
  sort(reverse?: boolean, nullsLast?: boolean): Expr
  sort({reverse, nullsLast}: {reverse?: boolean, nullsLast?: boolean}): Expr
  /**
   * Sort this column by the ordering of another column, or multiple other columns.
      In projection/ selection context the whole column is sorted.
      If used in a groupby context, the groups are sorted.

      Parameters
      ----------
      @param by
          The column(s) used for sorting.
      @param reverse
          false -> order from small to large.
          true -> order from large to small.
   */
  sortBy(by: ExprOrString[] | ExprOrString, reverse?: boolean | boolean[]): Expr
  sortBy(options: {by: ExprOrString[] | ExprOrString, reverse?: boolean | boolean[]}): Expr
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
  take(index: Expr | number[] | Series): Expr
  take({index}: {index: Expr | number[] | Series}): Expr
  /** Take every nth value in the Series and return as a new Series. */
  takeEvery(n: number): Expr
  /**
   * Get the unique values of this expression;
   * @param maintainOrder Maintain order of data. This requires more work.
   */
  unique(maintainOrder?: boolean | {maintainOrder: boolean}): Expr
  /** Returns a unit Series with the highest value possible for the dtype of this expression. */
  upperBound(): Expr
  /** Get variance. */
  var(): Expr
  /** Alais for filter: @see {@link filter} */
  where(predicate: Expr): Expr
}


export const _Expr = (_expr: any): Expr => {

  const unwrap = (method: string, ...args: any[]) => {
    return _expr[method as any](...args);
  };
  const wrap = (method, ...args): Expr => {
    return _Expr(unwrap(method, ...args));
  };

  const wrapExprArg = (method: string, lit=false) => (other: any) => {

    const expr = exprToLitOrExpr(other, lit).inner();

    return wrap(method, expr);
  };

  const rolling = (method: string) =>  (opts, weights?, minPeriods?, center?): Expr => {
    const windowSize = opts?.["windowSize"] ?? (typeof opts === "number" ? opts : null);
    if(windowSize === null) {
      throw new Error("window size is required");
    }
    const callOpts = {
      windowSize: opts?.["windowSize"] ?? (typeof opts === "number"? opts : null),
      weights: opts?.["weights"] ?? weights,
      minPeriods: opts?.["minPeriods"] ?? minPeriods ?? windowSize,
      center : opts?.["center"] ?? center ?? false,
    };

    return wrap(method, callOpts);
  };

  return {
    _expr,
    [Symbol.toStringTag]() {
      return "Expr";
    },
    [INSPECT_SYMBOL]() {
      return _expr.toString();
    },
    serialize(format) {
      return _expr.serialize(format);
    },
    toString() {
      return _expr.toString();
    },
    toJSON(...args: any[]) {
      // this is passed by `JSON.stringify` when calling `toJSON()`
      if(args[0] === "") {
        return _expr.toJs();
      }

      return _expr.serialize("json").toString();
    },
    get str() {
      return expr.StringFunctions(_expr);
    },
    get lst() {
      return expr.ListFunctions(_expr);
    },
    get date() {
      return expr.DateTimeFunctions(_expr);
    },
    get struct() {
      return expr.StructFunctions(_expr);
    },
    abs() {
      return _Expr(_expr.abs());
    },
    aggGroups() {
      return _Expr(_expr.aggGroups());
    },
    alias(name) {
      return _Expr(_expr.alias(name));
    },
    inner() {
      return _expr;
    },
    and(other) {
      const expr = (exprToLitOrExpr(other, false) as any).inner();

      return _Expr(_expr.and(expr));
    },
    argMax() {
      return _Expr(_expr.argMax());
    },
    argMin() {
      return _Expr(_expr.argMin());
    },
    argSort(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.argSort(reverse));
    },
    argUnique() {
      return _Expr(_expr.argUnique());
    },
    as(name) {
      return _Expr(_expr.alias(name));
    },
    backwardFill() {
      return _Expr(_expr.backwardFill());
    },
    cast(dtype, strict=false) {
      return _Expr(_expr.cast(dtype, strict));
    },
    ceil() {
      return _Expr(_expr.ceil());
    },
    clip(arg, max?){
      if(typeof arg === "number") {
        return _Expr(_expr.clip(arg, max));
      } else {
        return _Expr(_expr.clip(arg.min, arg.max));
      }
    },
    count() {
      return _Expr(_expr.count());
    },
    cumCount(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.cumcount(reverse?.reverse ?? reverse));
    },
    cumMax(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.cummax(reverse));
    },
    cumMin(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.cummin(reverse));
    },
    cumProd(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.cumprod(reverse));
    },
    cumSum(reverse: any = false) {
      reverse = reverse?.reverse ?? reverse;

      return _Expr(_expr.cumsum(reverse));
    },
    diff(n, nullBehavior = "ignore") {

      if(typeof n === "number") {

        return _Expr(_expr.diff(n, nullBehavior));
      }
      else {
        return _Expr(_expr.diff(n.n, n.nullBehavior));
      }
    },
    dot(other) {
      const expr = (exprToLitOrExpr(other, false) as any).inner();

      return _Expr(_expr.dot(expr));
    },
    exclude(...columns) {
      return _Expr(_expr.exclude(columns.flat(2)));
    },
    explode() {
      return _Expr(_expr.explode());
    },
    extend(o, n?) {
      if(n !== null && typeof n === "number") {
        return _Expr(_expr.extendConstant(o, n));
      }

      return _Expr(_expr.extendConstant(o.value, o.n));

    },
    extendConstant(o, n?) {
      if(n !== null && typeof n === "number") {
        return _Expr(_expr.extendConstant(o, n));
      }

      return _Expr(_expr.extendConstant(o.value, o.n));
    },
    fillNan(other) {
      const expr = (exprToLitOrExpr(other, true) as any).inner();

      return _Expr(_expr.fillNan(expr));
    },
    fillNull(fillValue)  {
      if(["backward", "forward", "mean", "min", "max", "zero", "one"].includes(fillValue)) {
        return _Expr(_expr.fillNullWithStrategy(fillValue));
      }

      const expr = exprToLitOrExpr(fillValue).inner();

      return _Expr(_expr.fillNull(expr));
    },
    filter(predicate) {
      const expr = exprToLitOrExpr(predicate).inner();

      return _Expr(_expr.filter(expr));
    },
    first() {
      return _Expr(_expr.first());
    },
    flatten() {
      return _Expr(_expr.explode());
    },
    floor() {
      return _Expr(_expr.floor());
    },
    forwardFill() {
      return _Expr(_expr.forwardFill());
    },
    hash(obj: any=0, k1=1, k2=2, k3=3) {
      if (typeof obj === "number" || typeof obj === "bigint") {
        return wrap("hash", BigInt(obj), BigInt(k1), BigInt(k2), BigInt(k3));
      }
      const o = { k0: obj, k1: k1, k2: k2, k3: k3, ...obj};

      return wrap(
        "hash",
        BigInt(o.k0),
        BigInt(o.k1),
        BigInt(o.k2),
        BigInt(o.k3)
      );
    },
    head(length) {
      if(typeof length === "number") {
        return wrap("head", length);
      }

      return wrap("head", length.length);
    },
    interpolate() {
      return _Expr(_expr.interpolate());
    },
    isDuplicated() {
      return _Expr(_expr.isDuplicated());
    },
    isFinite() {
      return _Expr(_expr.isFinite());
    },
    isInfinite() {
      return _Expr(_expr.isInfinite());
    },
    isFirst() {
      return _Expr(_expr.isFirst());
    },
    isNan() {
      return _Expr(_expr.isNan());
    },
    isNotNan() {
      return _Expr(_expr.isNotNan());
    },
    isNotNull() {
      return _Expr(_expr.isNotNull());
    },
    isNull() {
      return _Expr(_expr.isNull());
    },
    isUnique() {
      return _Expr(_expr.isUnique());
    },
    isIn(other)  {
      if(Array.isArray(other)) {
        other = pli.lit(Series(other).inner());
      } else {
        other = exprToLitOrExpr(other, false).inner();
      }

      return wrap("isIn", other);
    },
    keepName() {
      return _Expr(_expr.keepName());
    },
    kurtosis(obj?, bias=true) {
      const fisher = obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true);
      bias = obj?.["bias"] ?? bias;

      return _Expr(_expr.kurtosis(fisher, bias));

    },
    last() {
      return _Expr(_expr.last());
    },
    list() {
      return _Expr(_expr.list());
    },
    lowerBound() {
      return _Expr(_expr.lowerBound());
    },
    max() {
      return _Expr(_expr.max());
    },
    mean() {
      return _Expr(_expr.mean());
    },
    median() {
      return _Expr(_expr.median());
    },
    min() {
      return _Expr(_expr.min());
    },
    mode() {
      return _Expr(_expr.mode());
    },
    not() {
      return _Expr(_expr.not());
    },
    nUnique() {
      return _Expr(_expr.nUnique());
    },
    or(other) {
      const expr = exprToLitOrExpr(other).inner();

      return _Expr(_expr.or(expr));
    },
    over(...exprs) {
      const partitionBy = selectionToExprList(exprs, false);

      return wrap("over", partitionBy);
    },
    pow(exponent) {
      return _Expr(_expr.pow(exponent?.exponent ?? exponent));
    },
    prefix(prefix) {

      return _Expr(_expr.prefix(prefix));
    },
    quantile(quantile, interpolation = "nearest") {
      return _Expr(_expr.quantile(quantile, interpolation));
    },
    rank(method, reverse=false) {
      return _Expr(_expr.rank(method?.method ?? method, method?.reverse ?? reverse));
    },
    reinterpret(signed: any = true) {
      signed = signed?.signed ?? signed;

      return _Expr(_expr.reinterpret(signed));
    },
    repeatBy(expr) {
      const e = exprToLitOrExpr(expr, false)._expr;

      return _Expr(_expr.repeatBy(e));
    },
    reverse() {
      return _Expr(_expr.reverse());
    },
    rollingMax: rolling("rollingMax"),
    rollingMean: rolling("rollingMean"),
    rollingMin: rolling("rollingMin"),
    rollingSum: rolling("rollingSum"),
    rollingStd: rolling("rollingStd"),
    rollingVar: rolling("rollingVar"),
    rollingMedian: rolling("rollingMedian"),
    rollingQuantile(val, interpolation?, windowSize?, weights?, minPeriods?, center?) {
      if(typeof val === "number") {

        return wrap("rollingQuantile",
          val,
          interpolation ?? "nearest",
          {
            windowSize,
            weights,
            minPeriods,
            center
          });
      }
      windowSize = val?.["windowSize"] ?? (typeof val === "number" ? val : null);
      if(windowSize === null) {
        throw new Error("window size is required");
      }
      const options = {
        windowSize: val?.["windowSize"] ?? (typeof val === "number"? val : null),
        weights: val?.["weights"] ?? weights,
        minPeriods: val?.["minPeriods"] ?? minPeriods ?? windowSize,
        center : val?.["center"] ?? center ?? false,
      };

      return wrap("rollingQuantile",
        val.quantile,
        val.interpolation ?? "nearest",
        options
      );
    },
    rollingSkew(val, bias=true)  {
      if(typeof val === "number") {
        return wrap("rollingSkew", val, bias);
      }

      return wrap("rollingSkew", val.windowSize, val.bias ?? bias);
    },
    round(decimals) {
      return _Expr(_expr.round(decimals?.decimals ?? decimals));
    },
    sample(opts?, frac?, withReplacement = false, seed?) {
      if(opts?.n  !== undefined || opts?.frac  !== undefined) {

        return this.sample(opts.n, opts.frac, opts.withReplacement, seed);
      }
      if (typeof opts === "number") {
        throw new Error("sample_n is not yet supported for expr");
      }
      if(typeof frac === "number") {
        return wrap("sampleFrac",
          frac,
          withReplacement,
          false,
          seed
        );
      }
      else {
        throw new TypeError("must specify either 'frac' or 'n'");
      }
    },
    shift(periods) {
      return _Expr(_expr.shift(periods));
    },
    shiftAndFill(optOrPeriods, fillValue?) {
      if(typeof optOrPeriods === "number") {
        fillValue = exprToLitOrExpr(fillValue).inner();

        return wrap("shiftAndFill", optOrPeriods, fillValue);

      }
      else {
        fillValue = exprToLitOrExpr(optOrPeriods.fillValue).inner();
        const periods = optOrPeriods.periods;

        return wrap("shiftAndFill", periods, fillValue);
      }
    },
    skew(bias) {
      return wrap("skew", bias?.bias ?? bias ?? true);
    },
    slice(arg, len?) {
      if(typeof arg === "number") {
        return wrap("slice", pli.lit(arg), pli.lit(len));
      }

      return wrap("slice", pli.lit(arg.offset), pli.lit(arg.length));
    },
    sort(reverse: any = false, nullsLast=false) {
      if(typeof reverse === "boolean") {
        return wrap("sortWith", reverse, nullsLast);
      }

      return wrap("sortWith", reverse?.reverse ?? false, reverse?.nullsLast ?? nullsLast);
    },
    sortBy(arg, reverse=false) {
      if(arg?.by !== undefined) {
        return this.sortBy(arg.by, arg.reverse);
      }

      reverse = Array.isArray(reverse) ? reverse.flat() : [reverse] as any;
      const by = selectionToExprList(arg, false);

      return wrap("sortBy", by, reverse);
    },
    std() {
      return _Expr(_expr.std());
    },
    suffix(suffix) {

      return _Expr(_expr.suffix(suffix));
    },
    sum() {
      return _Expr(_expr.sum());
    },
    tail(length) {
      return _Expr(_expr.tail(length));
    },

    take(indices)  {
      if(Array.isArray(indices)) {
        indices = pli.lit(Series(indices).inner());
      } else {
        indices = indices.inner();
      }

      return wrap("take", indices);
    },
    takeEvery(n) {
      return _Expr(_expr.takeEvery(n));
    },
    unique(opt?) {
      if(opt) {
        return wrap("unique_stable");
      }

      return wrap("unique");
    },
    upperBound() {
      return _Expr(_expr.upperBound());
    },
    where(expr) {

      return this.filter(expr);
    },
    var() {
      return _Expr(_expr.var());
    },
    add: wrapExprArg("add"),
    sub: wrapExprArg("sub"),
    div: wrapExprArg("div"),
    mul: wrapExprArg("mul"),
    rem: wrapExprArg("rem"),

    plus: wrapExprArg("add"),
    minus: wrapExprArg("sub"),
    divideBy: wrapExprArg("div"),
    multiplyBy: wrapExprArg("mul"),
    modulo: wrapExprArg("rem"),

    eq: wrapExprArg("eq"),
    equals: wrapExprArg("eq"),
    gtEq: wrapExprArg("gtEq"),
    greaterThanEquals: wrapExprArg("gtEq"),
    gt: wrapExprArg("gt"),
    greaterThan: wrapExprArg("gt"),
    ltEq: wrapExprArg("ltEq"),
    lessThanEquals: wrapExprArg("ltEq"),
    lt: wrapExprArg("lt"),
    lessThan: wrapExprArg("lt"),
    neq: wrapExprArg("neq"),
    notEquals: wrapExprArg("neq"),
  };
};

export interface ExprConstructor extends Deserialize<Expr> {
  isExpr(arg: any): arg is Expr;
}

const isExpr = (anyVal: any): anyVal is Expr =>  {
  try {
    return anyVal?.[Symbol.toStringTag]?.() === "Expr";
  } catch (err) {
    return false;
  }
};


const deserialize = (buf, format) => {
  return _Expr(pli.JsExpr.deserialize(buf, format));
};

export const Expr: ExprConstructor = Object.assign(_Expr, {isExpr, deserialize});

/** @ignore */
export const exprToLitOrExpr = (expr: any, stringToLit = true): Expr  => {
  if(typeof expr === "string" && !stringToLit) {
    return _Expr(pli.col(expr));
  } else if (Expr.isExpr(expr)) {
    return expr;
  } else if (Series.isSeries(expr)) {
    return _Expr(pli.lit((expr as any)._s));
  } else {
    return _Expr(pli.lit(expr));
  }
};
