import {Expr} from "./lazy/expr";

export type RollingOptions = {
  windowSize: number,
  weights?: Array<number>,
  minPeriods?: number,
  center?: boolean
};

export type Interpolation = "nearest" | "higher" | "lower" | "midpoint" | "linear"

export interface Arithmetic<T> {
  add(rhs: any): T
  sub(rhs: any): T
  div(rhs: any): T
  mul(rhs: any): T
  rem(rhs: any): T
  plus(rhs: any): T
  minus(rhs: any): T
  divideBy(rhs: any): T
  multiplyBy(rhs: any): T
  modulo(rhs: any): T
}

export interface Comparison<T> {
  eq(rhs: any): T
  equals(rhs: any): T
  gtEq(rhs: any): T
  greaterThanEquals(rhs: any): T
  gt(rhs: any): T
  greaterThan(rhs: any): T
  ltEq(rhs: any): T
  lessThanEquals(rhs: any): T
  lt(rhs: any): T
  lessThan(rhs: any): T
  neq(rhs: any): T
  notEquals(rhs: any): T
}

export interface Cumulative<T> {
  /** Get an array with the cumulative count computed at every element. */
  cumCount(reverse?: boolean): T
  cumCount({reverse}: {reverse: boolean}): T
  /**
   * __Get an array with the cumulative max computes at every element.__
   * ___
   * @param reverse - reverse the operation
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.cumMax()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   * ```
   */
  cumMax(reverse?: boolean): T
  cumMax({reverse}: {reverse: boolean}): T
  /**
    * __Get an array with the cumulative min computed at every element.__
    * ___
    * @param reverse - reverse the operation
    * @example
    * ```
    * > const s = pl.Series("a", [1, 2, 3])
    * > s.cumMin()
    * shape: (3,)
    * Series: 'b' [i64]
    * [
    *         1
    *         1
    *         1
    * ]
    * ```
    */
  cumMin(reverse?: boolean): T
  cumMin({reverse}: {reverse: boolean}): T
  /**
   * __Get an array with the cumulative product computed at every element.__
   * ___
   * @param reverse - reverse the operation
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.cumProd()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         2
   *         6
   * ]
   * ```
   */
  cumProd(reverse?: boolean): T
  cumProd({reverse}: {reverse: boolean}): T
  /**
    * __Get an array with the cumulative sum computed at every element.__
    * ___
    * @param reverse - reverse the operation
    * @example
    * ```
    * > const s = pl.Series("a", [1, 2, 3])
    * > s.cumSum()
    * shape: (3,)
    * Series: 'b' [i64]
    * [
    *         1
    *         3
    *         6
    * ]
    * ```
    */
  cumSum(reverse?: boolean): T
  cumSum({reverse}: {reverse: boolean}): T
}

export interface Rolling<T> {
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
   */
  rollingMax(options: RollingOptions): T
  rollingMax(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
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
   */
  rollingMean(options: RollingOptions): T
  rollingMean(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
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
   */
  rollingMin(options: RollingOptions): T
  rollingMin(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
  /**
   * Compute a rolling std dev
   *
   * A window of length `window_size` will traverse the array. The values that fill this window
   * will (optionally) be multiplied with the weights given by the `weight` vector. The resulting
   * values will be aggregated to their sum.
   * ___
   * @param windowSize - The length of the window.
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   */
  rollingStd(options: RollingOptions): T
  rollingStd(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
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
  rollingSum(options: RollingOptions): T
  rollingSum(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
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
   */
  rollingVar(options: RollingOptions): T
  rollingVar(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
  /** Compute a rolling median */
  rollingMedian(options: RollingOptions): T
  rollingMedian(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?: boolean): T
  /**
   * Compute a rolling quantile
   * @param quantile quantile to compute
   * @param interpolation interpolation type
   * @param windowSize Size of the rolling window
   * @param weights - An optional slice with the same length as the window that will be multiplied
   * elementwise with the values in the window.
   * @param minPeriods The number of values in the window that should be non-null before computing a result.
   * If undefined, it will be set equal to window size.
   * @param center - Set the labels at the center of the window
   */
  rollingQuantile(options: RollingOptions & {quantile: number, interpolation?: Interpolation}): T
  rollingQuantile(
    quantile: number,
    interpolation?: Interpolation,
    windowSize?: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center?: boolean
  ): T
  /**
   * Compute a rolling skew
   * @param windowSize Size of the rolling window
   * @param bias If false, then the calculations are corrected for statistical bias.
   */
  rollingSkew(windowSize: number, bias?: boolean): T
  rollingSkew({windowSize, bias}: {windowSize: number, bias?: boolean}): T
}


export interface Round<T> {
  /**
   * Round underlying floating point data by `decimals` digits.
   *
   * Similar functionality to javascript `toFixed`
   * @param decimals number of decimals to round by.
   */
  round(decimals: number): T
  round(options: {decimals: number}): T
  /**
   * Floor underlying floating point array to the lowest integers smaller or equal to the float value.
   * Only works on floating point Series
   */
  floor(): T;
  /**
   * Ceil underlying floating point array to the highest integers smaller or equal to the float value.
   * Only works on floating point Series
   */
  ceil(): T;

  /**
   * Clip (limit) the values in an array to any value that fits in 64 floating point range.
   * Only works for the following dtypes: {Int32, Int64, Float32, Float64, UInt32}.
   * If you want to clip other dtypes, consider writing a when -> then -> otherwise expression
   * @param min Minimum value
   * @param max Maximum value
   */
  clip(min: number, max: number): T
  clip(options: {min: number, max: number})
}

export interface Sample<T> {
  /**
   * Sample from this DataFrame by setting either `n` or `frac`.
   * @param n - Number of samples < self.len() .
   * @param frac - Fraction between 0.0 and 1.0 .
   * @param withReplacement - Sample with replacement.
   * @param seed - Seed initialization. If not provided, a random seed will be used
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.sample({n: 2})
   * shape: (2, 3)
   * ╭─────┬─────┬─────╮
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" │
   * ╰─────┴─────┴─────╯
   * ```
   */

  sample(opts?: {n: number, withReplacement?: boolean, seed?: number | bigint}): T
  sample(opts?: {frac: number, withReplacement?: boolean, seed?: number | bigint}): T
  sample(n?: number, frac?: number, withReplacement?: boolean, seed?: number | bigint): T
}

export interface Bincode<T> {
  (bincode: Uint8Array): T;
  getState(T): Uint8Array;
}


export interface ListFunctions<T> {
  argMin(): T;
  argMax(): T;
  /**
   * Get the value by index in the sublists.
   * So index `0` would return the first item of every sublist
   * and index `-1` would return the last item of every sublist
   * if an index is out of bounds, it will return a `null`.
   */
  get(index: number): T
  /**
      Run any polars expression against the lists' elements
      Parameters
      ----------
      @param expr
          Expression to run. Note that you can select an element with `pl.first()`, or `pl.col()`
      @param parallel
          Run all expression parallel. Don't activate this blindly.
          Parallelism is worth it if there is enough work to do per thread.
          This likely should not be use in the groupby context, because we already parallel execution per group
      @example
      --------
      >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
      >>> df.withColumn(
      ...   pl.concatList(["a", "b"]).lst.eval(pl.first().rank()).alias("rank")
      ... )
      shape: (3, 3)
      ┌─────┬─────┬────────────┐
      │ a   ┆ b   ┆ rank       │
      │ --- ┆ --- ┆ ---        │
      │ i64 ┆ i64 ┆ list [f32] │
      ╞═════╪═════╪════════════╡
      │ 1   ┆ 4   ┆ [1.0, 2.0] │
      ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
      │ 8   ┆ 5   ┆ [2.0, 1.0] │
      ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
      │ 3   ┆ 2   ┆ [2.0, 1.0] │
      └─────┴─────┴────────────┘
   */
  eval(expr: Expr, parallel: boolean): T
  /** Get the first value of the sublists. */
  first(): T
  /**
   * Join all string items in a sublist and place a separator between them.
   * This errors if inner type of list `!= Utf8`.
   * @param separator A string used to separate one element of the list from the next in the resulting string.
   * If omitted, the list elements are separated with a comma.
   */
  join(separator?: string): T
  /** Get the last value of the sublists. */
  last(): T
  lengths(): T;
  max(): T;
  mean(): T;
  min(): T;
  reverse(): T;
  shift(periods: number): T;
  slice(offset: number, length: number): T;
  sort(reverse?: boolean): T;
  sort(opt: {reverse: boolean}): T;
  sum(): T;
  unique(): T;
}

export interface DateFunctions<T> {
  /**
   * Extract day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of month starting from 1.
   * The return value ranges from 1 to 31. (The last day of month differs by months.)
   * @returns day as pl.UInt32
   */
  day(): T;
  /**
   * Extract hour from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the hour number from 0 to 23.
   * @returns Hour as UInt32
   */
  hour(): T;
  /**
   * Extract minutes from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the minute number from 0 to 59.
   * @returns minute as UInt32
   */
  minute(): T;
  /**
   * Extract month from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the month number starting from 1.
   * The return value ranges from 1 to 12.
   * @returns Month as UInt32
   */
  month(): T;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the number of nanoseconds since the whole non-leap second.
   * The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
   * @returns Nanosecond as UInt32
   */
  nanosecond(): T;
  /**
   * Extract ordinal day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of year starting from 1.
   * The return value ranges from 1 to 366. (The last day of year differs by years.)
   * @returns Day as UInt32
   */
  ordinalDay(): T;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the second number from 0 to 59.
   * @returns Second as UInt32
   */
  second(): T;
  /**
   * Format Date/datetime with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
   */
  strftime(fmt: string): T;
  /** Return timestamp in ms as Int64 type. */
  timestamp(): T;
  /**
   * Extract the week from the underlying Date representation.
   * Can be performed on Date and Datetime
   *
   * Returns the ISO week number starting from 1.
   * The return value ranges from 1 to 53. (The last week of year differs by years.)
   * @returns Week number as UInt32
   */
  week(): T;
  /**
   * Extract the week day from the underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the weekday number where monday = 0 and sunday = 6
   * @returns Week day as UInt32
   */
  weekday(): T;
  /**
   * Extract year from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the year number in the calendar date.
   * @returns Year as Int32
   */
  year(): T;
}

export interface Serialize {
  /**
   * Serializes object to desired format via [serde](https://serde.rs/)
   *
   * @param format [json](https://github.com/serde-rs/json) | [bincode](https://github.com/bincode-org/bincode)
   *
   */
  serialize(format: "json" | "bincode"): Buffer
}
export interface Deserialize<T> {
  /**
  * De-serializes buffer via [serde](https://serde.rs/)
  * @param buf buffer to deserialize
  * @param format [json](https://github.com/serde-rs/json) | [bincode](https://github.com/bincode-org/bincode)
  *
  */
  deserialize(buf: Buffer, format: "json" | "bincode"): T
}
