import {DataType} from "../datatypes";
import pli from "../internals/polars_internal";
import {col, lit} from "./functions";
import {
  ExprOrString,
  FillNullStrategy,
  RankMethod,
  selectionToExprList,
  INSPECT_SYMBOL
} from "../utils";
import {isExternal} from "util/types";
import {Series} from "../series/series";

import * as expr from "./expr/";
import {Arithmetic, Comparison, Cumulative, Rolling, Round} from "../shared_traits";

export interface Expr extends
  Rolling<Expr>,
  Arithmetic<Expr>,
  Comparison<Expr>,
  Cumulative<Expr>,
  Round<Expr> {
  /** @ignore */
  _expr: any;
  get date(): expr.Datetime;
  get str(): expr.String;
  get lst(): expr.List;
  [INSPECT_SYMBOL](): string;
  toString(): string;
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
  /**
   * Get the unique/distinct values in the list
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


const _Expr = (_expr: any): Expr => {

  const wrap = (method, args?): Expr => {

    return Expr(pli.expr[method]({_expr, ...args }));
  };

  const wrapNullArgs = (method: string) => () => wrap(method);
  const wrapExprArg = (method: string, lit=false) => (other: any) => {

    const expr = exprToLitOrExpr(other, lit)._expr;

    return wrap(method, {other: expr});
  };
  type anyfunc = (...args: any[]) => any
  const wrapUnary = (method: string, key: string): anyfunc => (val) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapUnaryWithDefault = (method: string, key: string, otherwise): anyfunc => (val=otherwise) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapBinary = (method: string, key0: string, key1: string): anyfunc => (val0, val1) => {
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
  const rolling = (method: string) =>  (opts, weights?, minPeriods?, center?): Expr => {
    const windowSize = opts?.["windowSize"] ?? (typeof opts === "number" ? opts : null);
    if(windowSize === null) {
      throw new Error("window size is required");
    }
    const callOpts = {
      window_size: opts?.["windowSize"] ?? (typeof opts === "number"? opts : null),
      weights: opts?.["weights"] ?? weights,
      min_periods: opts?.["minPeriods"] ?? minPeriods ?? windowSize,
      center : opts?.["center"] ?? center ?? false,
    };

    return wrap(method, callOpts);
  };

  return {
    _expr,
    [INSPECT_SYMBOL]() {
      return pli.expr.as_str({_expr});
    },
    toString() {
      return pli.expr.as_str({_expr});
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
    cast(dtype, strict=false) {
      return wrap("cast", {dtype, strict});
    },
    ceil: wrapNullArgs("ceil"),
    clip(arg, max?){
      if(typeof arg === "number") {
        return wrap("clip", {min: arg, max});
      } else {
        return wrap("clip", arg);
      }
    },
    count: wrapNullArgs("count"),
    cumCount: wrapUnaryWithDefault("cumCount", "reverse", false),
    cumMax: wrapUnaryWithDefault("cumMax", "reverse", false),
    cumMin: wrapUnaryWithDefault("cumMin", "reverse", false),
    cumProd: wrapUnaryWithDefault("cumProd", "reverse", false),
    cumSum: wrapUnaryWithDefault("cumSum", "reverse", false),
    diff: wrapBinary("diff", "n", "nullBehavior"),
    dot: wrapExprArg("dot"),
    exclude(...columns) {
      return wrap("exclude", {columns: columns.flat(2)});
    },
    explode: wrapNullArgs("explode"),
    extend(o, n?) {
      if(n !== null && typeof n === "number") {
        return wrap("extendConstant", {value: o, n});
      }

      return wrap("extendConstant", o);
    },
    extendConstant(o, n?) {
      if(n !== null && typeof n === "number") {
        return wrap("extendConstant", {value: o, n});
      }

      return wrap("extendConstant", o);
    },
    fillNan: wrapExprArg("fillNan", true),
    fillNull(fillValue)  {
      if(["backward", "forward", "mean", "min", "max", "zero", "one"].includes(fillValue)) {
        return wrap("fillNullWithStrategy", {strategy: fillValue});
      }

      const expr = exprToLitOrExpr(fillValue)._expr;

      return wrap("fillNull", {other: expr});
    },
    filter: wrapExprArg("filter"),
    first: wrapNullArgs("first"),
    flatten: wrapNullArgs("explode"),
    floor: wrapNullArgs("floor"),
    forwardFill: wrapNullArgs("forwardFill"),
    hash(obj: any=0, k1=1, k2=2, k3=3) {
      if(typeof obj === "number" || typeof obj === "bigint") {
        return wrap("hash", { k0: obj, k1: k1, k2: k2, k3: k3 });
      }

      return wrap("hash", { k0: 0, k1, k2, k3, ...obj });
    },
    head: wrapUnaryNumber("head", "length"),
    interpolate: wrapNullArgs("interpolate"),
    isDuplicated: wrapNullArgs("isDuplicated"),
    isFinite: wrapNullArgs("isFinite"),
    isFirst: wrapNullArgs("isFirst"),
    isIn(other)  {
      if(Array.isArray(other)) {
        other = lit(Series(other));
      } else {
        other = exprToLitOrExpr(other, false);
      }

      return wrap("isIn", {other: other._expr});
    },
    isInfinite: wrapNullArgs("isInfinite"),
    isNan: wrapNullArgs("isNan"),
    isNotNan: wrapNullArgs("isNotNan"),
    isNotNull: wrapNullArgs("isNotNull"),
    isNull: wrapNullArgs("isNull"),
    isUnique: wrapNullArgs("isUnique"),
    keepName: wrapNullArgs("keepName"),
    kurtosis(obj?, bias=true) {
      return wrap("kurtosis", {
        fisher: obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true),
        bias : obj?.["bias"] ?? bias,
      });
    },
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
    or: wrapExprArg("or"),
    over(...exprs) {

      const partitionBy = selectionToExprList(exprs, false);

      return wrap("over", {partitionBy});
    },
    pow: wrapUnary("pow", "exponent"),
    prefix: wrapUnary("prefix", "prefix"),
    quantile: wrapUnary("quantile", "quantile"),
    rank: wrapUnary("rank", "method"),
    reinterpret: wrapUnaryWithDefault("reinterpret", "signed", true),
    repeatBy: wrapExprArg("repeatBy"),
    reverse: wrapNullArgs("reverse"),
    rollingMax: rolling("rollingMax"),
    rollingMean: rolling("rollingMean"),
    rollingMin: rolling("rollingMin"),
    rollingSum: rolling("rollingSum"),
    rollingStd: rolling("rollingStd"),
    rollingVar: rolling("rollingVar"),
    rollingMedian: rolling("rollingMedian"),
    rollingQuantile(val, interpolation?, windowSize?, weights?, minPeriods?, center?) {
      if(typeof val === "number") {
        return wrap("rollingQuantile", {
          quantile: val,
          interpolation,
          windowSize,
          weights,
          minPeriods,
          center
        });
      }

      return wrap("rollingQuantile", val);
    },
    rollingSkew(val, bias=true)  {
      if(typeof val === "number") {
        return wrap("rollingSkew", {windowSize: val, bias});
      }

      return wrap("rollingSkew", {bias:true, ...val});
    },
    round: wrapUnaryNumber("round", "decimals"),
    shift: wrapUnary("shift", "periods"),
    shiftAndFill(optOrPeriods, fillValue?) {
      if(typeof optOrPeriods === "number") {
        fillValue = exprToLitOrExpr(fillValue)._expr;

        return wrap("shiftAndFill", {periods: optOrPeriods, fillValue});

      }
      else {
        fillValue = exprToLitOrExpr(optOrPeriods.fillValue)._expr;
        const periods = optOrPeriods.periods;

        return wrap("shiftAndFill", {periods, fillValue});
      }
    },
    skew: wrapUnaryWithDefault("skew", "bias", true),
    slice: wrapBinary("slice", "offset", "length"),
    sort(reverse: any = false, nullsLast=false) {
      if(typeof reverse === "boolean") {
        return wrap("sortWith", {reverse, nullsLast});
      }

      return wrap("sortWith", reverse);
    },
    std: wrapNullArgs("std"),
    suffix: wrapUnary("suffix", "suffix"),
    sum: wrapNullArgs("sum"),
    tail: wrapUnaryNumber("tail", "length"),
    take(indices)  {
      if(Array.isArray(indices)) {
        indices = lit(Series(indices));
      }

      return wrap("take", {other: indices._expr});
    },
    takeEvery: wrapUnary("takeEvery", "n"),
    unique(opt?) {
      if(opt) {
        return wrap("unique_stable");
      }

      return wrap("unique");
    },
    upperBound: wrapNullArgs("upperBound"),
    where: wrapExprArg("filter"),
    var: wrapNullArgs("var"),

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

const isExpr = (anyVal: any): anyVal is Expr => isExternal(anyVal?._expr);
export const Expr = Object.assign(_Expr, {isExpr});

/** @ignore */
export const exprToLitOrExpr = (expr: any, stringToLit = true): Expr  => {
  if(typeof expr === "string" && !stringToLit) {
    return col(expr);
  } else if (Expr.isExpr(expr)) {
    return expr;
  } else {
    return lit(expr);
  }
};
