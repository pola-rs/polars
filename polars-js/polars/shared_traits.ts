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
   * Ceil underlying floating point array to the heighest integers smaller or equal to the float value.
   * Only works on floating point Series
   */
  ceil(): T;

  /**
   * Clip (limit) the values in an array to any value that fits in 64 floating poitns range.
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
