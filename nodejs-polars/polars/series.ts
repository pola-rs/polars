import pli from "./internals/polars_internal";
import { arrayToJsSeries } from "./internals/construction";
import { DataType, DtypeToPrimitive, DTYPE_TO_FFINAME, Optional } from "./datatypes";
import {DataFrame, dfWrapper} from "./dataframe";
import {todo} from "./internals/utils";
import {StringFunctions} from "./series/string";
import {ListFunctions} from "./series/list";
import {InvalidOperationError} from "./error";
import {isSeries, RankMethod} from "./utils";
import {col} from "./lazy/lazy_functions";

const inspect = Symbol.for("nodejs.util.inspect.custom");

type ValueOrNever<V> = V extends ArrayLike<infer U> ? Series<U> : never;
type DataTypeOrValue<T, U> = U extends true ? DtypeToPrimitive<T> : DtypeToPrimitive<T> | null;
type ArrayLikeDataType<T> = ArrayLike<DtypeToPrimitive<T>>
type ArrayLikeOrDataType<T, U> = ArrayLike<DataTypeOrValue<T, U>>
export type JsSeries = any;
type RollingOptions = {
  windowSize: number,
  weights?: Array<number>,
  minPeriods?: number,
  center?:boolean
};


export interface Series<T> {
  [n: number]: T
  name: string
  dtype: DataType
  length: number
  _series: JsSeries;
  str: StringFunctions
  arr: ListFunctions
  [inspect](): string;
  [Symbol.iterator](): Generator<T, void, any>;
  inner(): JsSeries
  eq(field: Series<T> | number | bigint | string): Series<boolean>
  equals(field: Series<T> | number | bigint | string): Series<boolean>
  gt_eq(field: Series<T> | number | bigint | string): Series<boolean>
  greaterThanEquals(field: Series<T> | number | bigint | string): Series<boolean>
  gt(field: Series<T> | number | bigint | string): Series<boolean>
  greaterThan(field: Series<T> | number | bigint | string): Series<boolean>
  lt_eq(field: Series<T> | number | bigint | string): Series<boolean>
  lessThanEquals(field: Series<T> | number | bigint | string): Series<boolean>
  lt(field: Series<T> | number | bigint | string): Series<boolean>
  lessThan(field: Series<T> | number | bigint | string): Series<boolean>
  neq(field: Series<T> | number | bigint | string): Series<boolean>
  notEquals(field: Series<T> | number | bigint | string): Series<boolean>
  add(field: Series<T> | number | bigint): Series<T>
  sub(field: Series<T> | number | bigint): Series<T>
  div(field: Series<T> | number | bigint): Series<T>
  mul(field: Series<T> | number | bigint): Series<T>
  rem(field: Series<T> | number | bigint | string): Series<T>
  plus(field: Series<T> | number | bigint): Series<T>
  minus(field: Series<T> | number | bigint): Series<T>
  divide(field: Series<T> | number | bigint): Series<T>
  times(field: Series<T> | number | bigint): Series<T>
  remainder(field: Series<T> | number | bigint | string): Series<T>
  bitand(other: Series<any>): Series<T>
  bitor(other: Series<any>): Series<T>
  bitxor(other: Series<any>): Series<T>

  /**
   * Take absolute values
   */
  abs(): Series<T>
  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @see {@link rename}
   *
   */
   alias(name: string): Series<T>
  /**
   * __Append a Series to this one.__
   * ___
   * @param {Series} other - Series to append.
   * @example
   * > const s = pl.Series("a", [1, 2, 3])
   * > const s2 = pl.Series("b", [4, 5, 6])
   * > s.append(s2)
   * shape: (6,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   *         5
   *         6
   * ]
   */
  append(other: Series<T>): void
  /**
   * __Apply a function over elements in this Series and return a new Series.__
   *
   * If the function returns another datatype, the returnType arg should be set, otherwise the method will fail.
   * ___
   * @param {CallableFunction} func - function or lambda.
   * @param {DataType} returnType - Output datatype. If none is given, the same datatype as this Series will be used.
   * @returns {SeriesType} `Series<T> | Series<returnType>`
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.apply(x => x + 10)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         11
   *         12
   *         13
   * ]
   * ```
   */
  apply<U>(func: (s: T) => U): Series<U>
  /**
   * Get the index of the maximal value.
   */
  argMax(): Optional<number>
    /**
   * Get the index of the minimal value.
   */
  argMin(): Optional<number>
  /**
   * Get index values where Boolean Series evaluate True.
   *
   */
  argTrue(): Series<number>
  /**
   * Get unique index as Series.
   */
  argUnique(): Series<number>
  /**
   * Index location of the sorted variant of this Series.
   * ___
   * @param reverse
   * @return {SeriesType} indexes - Indexes that can be used to sort this array.
   */
  argSort(): Series<T>
  argSort(reverse: boolean): Series<T>
  argSort({reverse}: {reverse: boolean}): Series<T>
  /**
   * Cast between data types.
   */
  cast<D extends DataType>(dtype: D): Series<DtypeToPrimitive<D>>
  cast<D extends DataType>(dtype: D, strict:boolean): Series<DtypeToPrimitive<D>>
  cast<D extends DataType>(dtype: D, opt: {strict: boolean}): Series<DtypeToPrimitive<D>>
  /**
   * Get the length of each individual chunk
   */
  chunkLengths(): Array<number>

  /**
   * Cheap deep clones.
   */
  clone(): Series<T>
  concat(other: Series<T>): Series<T>
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
  cumMax(): Series<T>
  cumMax(reverse:boolean): Series<T>
  cumMax({reverse}: {reverse: boolean}): Series<T>
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
  cumMin(): Series<T>
  cumMin(reverse:boolean): Series<T>
  cumMin({reverse}: {reverse: boolean}): Series<T>
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
  cumProd(): Series<T>
  cumProd(reverse:boolean): Series<T>
  cumProd({reverse}: {reverse: boolean}): Series<T>
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
  cumSum(): Series<T>
  cumSum(reverse:boolean): Series<T>
  cumSum({reverse}: {reverse: boolean}): Series<T>
   /**
   * __Quick summary statistics of a series. __
   *
   * Series with mixed datatypes will return summary statistics for the datatype of the first value.
   * ___
   * @example
   * ```
   * > const seriesNum = pl.Series([1,2,3,4,5])
   * > series_num.describe()
   *
   * shape: (6, 2)
   * ┌──────────────┬────────────────────┐
   * │ statistic    ┆ value              │
   * │ ---          ┆ ---                │
   * │ str          ┆ f64                │
   * ╞══════════════╪════════════════════╡
   * │ "min"        ┆ 1                  │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "max"        ┆ 5                  │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "null_count" ┆ 0.0                │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "mean"       ┆ 3                  │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "std"        ┆ 1.5811388300841898 │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "count"      ┆ 5                  │
   * └──────────────┴────────────────────┘
   *
   * > series_str = pl.Series(["a", "a", None, "b", "c"])
   * > series_str.describe()
   *
   * shape: (3, 2)
   * ┌──────────────┬───────┐
   * │ statistic    ┆ value │
   * │ ---          ┆ ---   │
   * │ str          ┆ i64   │
   * ╞══════════════╪═══════╡
   * │ "unique"     ┆ 4     │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
   * │ "null_count" ┆ 1     │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
   * │ "count"      ┆ 5     │
   * └──────────────┴───────┘
   * ```
   */
  describe(): DataFrame
  /**
   * Calculates the n-th discrete difference.
   * @param n - number of slots to shift
   * @param nullBehavior - `'ignore' | 'drop'`
   */
  diff(n: number, nullBehavior: "ignore" | "drop"): Series<T>
  diff({n, nullBehavior}: {n: number, nullBehavior: "ignore" | "drop"}): Series<T>
   /**
   * Compute the dot/inner product between two Series
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > const s2 = pl.Series("b", [4.0, 5.0, 6.0])
   * > s.dot(s2)
   * 32.0
   * ```
   */
  dot(other: Series<any>): Optional<number>
  /**
   * Create a new Series that copies data from this Series without null values.
   */
  dropNulls(): Series<T>
  /**
   * __Explode a list or utf8 Series.__
   *
   * This means that every item is expanded to a new row.
   * ___
   * @example
   * ```
   * > const s = pl.Series('a', [[1, 2], [3, 4], [9, 10]])
   * > s.explode()
   * shape: (6,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   *         9
   *         10
   * ]
   * ```
   */
  explode(): any
  /**
   * __Fill null values with a filling strategy.__
   * ___
   * @param strategy - Filling Strategy
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3, None])
   * > s.fill_null('forward'))
   * shape: (4,)
   * Series: '' [i64]
   * [
   *         1
   *         2
   *         3
   *         3
   * ]
   * > s.fill_null('min'))
   * shape: (4,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         1
   * ]
   * ```
   */
  fillNull(strategy: "backward" | "forward" | "min" | "max" | "mean" | "one" | "zero"): Series<T>
  fillNull({strategy}: {strategy: "backward" | "forward" | "min" | "max" | "mean" | "one" | "zero"}): Series<T>
  /**
   * __Filter elements by a boolean mask.__
   * @param {SeriesType} predicate - Boolean mask
   *
   */
  filter(predicate: Series<boolean>): Series<T>
  filter({predicate}: {predicate: Series<boolean>}): Series<T>
  floor(): Series<T>
  get(index: number): T

  getIndex(n: number): T
  /**
   * Returns True if the Series has a validity bitmask.
   * If there is none, it means that there are no null values.
   */
  hasValidity(): boolean
  /**
   * Hash the Series
   * The hash value is of type `UInt64`
   * ___
   * @param k0 - seed parameter
   * @param k1 - seed parameter
   * @param k2 - seed parameter
   * @param k3 - seed parameter
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.hash(42)
   * shape: (3,)
   * Series: 'a' [u64]
   * [
   *   7499844439152382372
   *   821952831504499201
   *   6685218033491627602
   * ]
   * ```
   */
  hash(k0?:number, k1?: number, k2?: number, k3?:number): Series<bigint>
  hash({k0, k1, k2, k3}: {k0?:number, k1?: number, k2?: number, k3?:number}): Series<bigint>
  /**
   * __Get first N elements as Series.__
   * ___
   * @param length  Length of the head
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.head(2)
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   * ]
   * ```
   */
  head(length?: number): Series<T>
  /**
   * __Interpolate intermediate values.__
   *
   * The interpolation method is linear.
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, None, None, 5])
   * > s.interpolate()
   * shape: (5,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   *         5
   * ]
   * ```
   */
  interpolate(): Series<T>
  /**
   * Check if this Series is a Boolean.
   */
  isBoolean(): boolean
  /**
   * Check if this Series is a DataTime.
   */
  isDateTime(): boolean
  /**
   * __Get mask of all duplicated values.__
   *
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 2, 3])
   * > s.isDuplicated()
   *
   * shape: (4,)
   * Series: 'a' [bool]
   * [
   *         false
   *         true
   *         true
   *         false
   * ]
   * ```
   */
  isDuplicated(): Series<boolean>
  /**
   * Get mask of finite values if Series dtype is Float.
   */
  isFinite(): T extends number ? Series<boolean> : never
  /**
   * Get a mask of the first unique value.
   */
  isFirst(): Series<boolean>
  /**
   * Check if this Series is a Float.
   */
  isFloat(): boolean
  /**
   * Check if elements of this Series are in the right Series, or List values of the right Series.
   */
  isIn<U>(other: Series<U> | U[]): Series<boolean>
  /**
   * __Get mask of infinite values if Series dtype is Float.__
   * @example
   * ```
   * > const s = pl.Series("a", [1.0, 2.0, 3.0])
   * > s.isInfinite()
   *
   * shape: (3,)
   * Series: 'a' [bool]
   * [
   *         false
   *         false
   *         false
   * ]
   * ```
   */
  isInfinite(): T extends number ? Series<boolean> : never
  /**
   * __Get mask of non null values.__
   *
   * *`undefined` values are treated as null*
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1.0, undefined, 2.0, 3.0, null])
   * > s.isNotNull()
   * shape: (5,)
   * Series: 'a' [bool]
   * [
   *         true
   *         false
   *         true
   *         true
   *         false
   * ]
   * ```
   */
  isNotNull(): Series<boolean>
  /**
   * __Get mask of null values.__
   *
   * `undefined` values are treated as null
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1.0, undefined, 2.0, 3.0, null])
   * > s.isNull()
   * shape: (5,)
   * Series: 'a' [bool]
   * [
   *         false
   *         true
   *         false
   *         false
   *         true
   * ]
   * ```
   */
  isNull(): Series<boolean>
  /**
   * Check if this Series datatype is numeric.
   */
  isNumeric(): boolean
  /**
   * __Get mask of unique values.__
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 2, 3])
   * > s.isUnique()
   * shape: (4,)
   * Series: 'a' [bool]
   * [
   *         true
   *         false
   *         false
   *         true
   * ]
   * ```
   */
  isUnique(): Series<boolean>
  /**
   * Checks if this Series datatype is a Utf8.
   */
  isUtf8(): boolean
  /**
   * __Compute the kurtosis (Fisher or Pearson) of a dataset.__
   *
   * Kurtosis is the fourth central moment divided by the square of the
   * variance. If Fisher's definition is used, then 3.0 is subtracted from
   * the result to give 0.0 for a normal distribution.
   * If bias is False then the kurtosis is calculated using k statistics to
   * eliminate bias coming from biased moment estimators
   * ___
   * @param fisher -
   * - If True, Fisher's definition is used (normal ==> 0.0).
   * - If False, Pearson's definition is used (normal ==> 3.0)
   */
  kurtosis(): Optional<number>
  kurtosis({fisher, bias}:{fisher?:boolean, bias?:boolean}): Optional<number>
  kurtosis(fisher:boolean, bias?:boolean): Optional<number>
  /**
   * __Length of this Series.__
   * ___
   * @example
   * ```
   * > const s = pl.Series("a", [1, 2, 3])
   * > s.len()
   * 3
   * ```
   */
  len(): number
  /**
   * __Take `n` elements from this Series.__
   * ___
   * @param n - Amount of elements to take.
   * @see {@link head}
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s.limit(2)
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   * ]
   * ```
   */
  limit(n?:number): Series<T>
  /**
   * @see {@link Series.apply}
   */
  map<U>(func: (s: T) => U): Series<U>

  /**
   * Get the maximum value in this Series.
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.max()
   * 3
   * ```
   */
  max(): number
  /**
   * Reduce this Series to the mean value.
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.mean()
   * 2
   * ```
   */
  mean(): number
  /**
   * Get the median of this Series
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.median()
   * 2
   * ```
   */
  median(): number
  /**
   * Get the minimal value in this Series.
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.min()
   * 1
   * ```
   */
  min(): number
  /**
   * __Compute the most occurring value(s). Can return multiple Values__
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 2, 3])
   * s.mode()
   * shape: (1,)
   * Series: 'a' [i64]
   * [
   *         2
   * ]
   *
   * s = pl.Series("a", ['a', 'b', 'c', 'c', 'b'])
   * s.mode()
   * shape: (1,)
   * Series: 'a' [str]
   * [
   *         'b'
   *         'c'
   * ]
   * ```
   */
  mode(): Series<T>
  /**
   * Get the number of chunks that this Series contains.
   */
  nChunks(): number
  /**
   * __Count the number of unique values in this Series.__
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 2, 3])
   * s.nUnique()
   * 3
   * ```
   */
  nUnique(): number
  /**
   * Count the null values in this Series. --
   * _`undefined` values are treated as null_
   *
   */
  nullCount(): number
  /**
   * Get a boolean mask of the local maximum peaks.
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3, 4, 5])
   * s.peakMax()
   * shape: (5,)
   * Series: '' [bool]
   * [
   *         false
   *         false
   *         false
   *         false
   *         true
   * ]
   * ```
   */
  peakMax(): Series<boolean>
  /**
   * Get a boolean mask of the local minimum peaks.
   * ___
   * @example
   * ```
   * s = pl.Series("a", [4, 1, 3, 2, 5])
   * s.peakMin()
   * shape: (5,)
   * Series: '' [bool]
   * [
   *         false
   *         true
   *         false
   *         true
   *         false
   * ]
   * ```
   */
  peakMin(): Series<boolean>
  /**
   * Get the quantile value of this Series.
   * ___
   * @param quantile
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s.quantile(0.5)
   * 2
   * ```
   */
  quantile(quantile: number): number
  /**
   * Assign ranks to data, dealing with ties appropriately.
   * @param method
   * The method used to assign ranks to tied elements.
   * The following methods are available: _default is 'average'_
   *
   *  *   __'average'__: The average of the ranks that would have been assigned to
   *    all the tied values is assigned to each value.
   *  * __'min'__: The minimum of the ranks that would have been assigned to all
   *    the tied values is assigned to each value.  _This is also
   *    referred to as "competition" ranking._
   *  * __'max'__: The maximum of the ranks that would have been assigned to all
   *    the tied values is assigned to each value.
   *  * __'dense'__: Like 'min', but the rank of the next highest element is
   *    assigned the rank immediately after those assigned to the tied
   *    elements.
   *  * __'ordinal'__: All values are given a distinct rank, corresponding to
   *    the order that the values occur in `a`.
   *  * __'random'__: Like 'ordinal', but the rank for ties is not dependent
   *    on the order that the values occur in `a`.
   */
  rank(method?: RankMethod): Series<number>
  rechunk(): Series<T>
  rechunk(inPlace: boolean): void
  rechunk({inPlace}: {inPlace: true}): void
  rechunk({inPlace}: {inPlace: false}): Series<T>
  /**
   * __Reinterpret the underlying bits as a signed/unsigned integer.__
   *
   * This operation is only allowed for 64bit integers. For lower bits integers,
   * you can safely use that cast operation.
   * ___
   * @param signed signed or unsigned
   *
   * - True -> pl.Int64
   * - False -> pl.UInt64
   * @see {@link cast}
   *
   */
  reinterpret(signed?:boolean): T extends number ? Series<boolean> : never
  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @param inPlace - Modify the Series in-place.
   * @see {@link alias}
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s.rename('b')
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   * ```
   */
  rename(name: string): Series<T>;
  rename(name: string, inPlace: boolean): void
  rename({name, inPlace}: {name: string, inPlace?: boolean}): void
  // rename(name: string, inPlace?: boolean): void | Series<T>
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
   * ___
   * @example
   * ```
   * s = pl.Series("a", [100, 200, 300, 400, 500])
   * s.rollingMax(2)
   * shape: (5,)
   * Series: '' [i64]
   * [
   *         null
   *         null
   *         300
   *         400
   *         500
   * ]
   * ```
   * @see {@link rollingMean}, {@link rollingMin}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMax(options: RollingOptions): Series<T>
  rollingMax(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?:boolean): Series<T>
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
   * ___
   * @example
   * ```
   * s = pl.Series("a", [100, 200, 300, 400, 500])
   * s.rollingMean(2)
   * shape: (5,)
   * Series: '' [i64]
   * [
   *         null
   *         150
   *         250
   *         350
   *         450
   * ]
   * ```
   * @see {@link rollingMax}, {@link rollingMin}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMean(options: RollingOptions): Series<T>
  rollingMean(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?:boolean): Series<T>
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
   * ___
   * @example
   * ```
   * s = pl.Series("a", [100, 200, 300, 400, 500])
   * s.rollingMin(2)
   * shape: (5,)
   * Series: '' [i64]
   * [
   *         null
   *         null
   *         100
   *         200
   *         300
   * ]
   * ```
   * @see {@link rollingMax}, {@link rollingMean}, {@link rollingSum}, {@link rollingVar}
   */
  rollingMin(options: RollingOptions): Series<T>
  rollingMin(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?:boolean): Series<T>
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
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3, 4, 5])
   * s.rollingSum(2)
   * shape: (5,)
   * Series: '' [i64]
   * [
   *         null
   *         3
   *         5
   *         7
   *         9
   * ]
   * ```
   * @see {@link rollingMax}, {@link rollingMin}, {@link rollingMean}, {@link rollingVar}
   */
  rollingSum(options: RollingOptions): Series<T>
  rollingSum(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?:boolean): Series<T>
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
  rollingVar(options: RollingOptions): Series<T>
  rollingVar(windowSize: number, weights?: Array<number>, minPeriods?: Array<number>, center?:boolean): Series<T>
  /**
   * Round underlying floating point data by `decimals` digits.
   *
   * Similar functionality to javascript `toFixed`
   * @param decimals number of decimals to round by.
   *
   * */
  round<T>(decimals: number): T extends number ? Series<number> : never
  round(opt: {decimals: number}): T extends number ? Series<number> : never
  sample(opts: {n?: number, frac?: number, withReplacement?:boolean}): Series<T>
  sample(n?: number, frac?: number, withReplacement?:boolean): Series<T>
  /**
   * __Check if series is equal with another Series.__
   * @param other - Series to compare with.
   * @param nullEqual - Consider null values as equal. _('undefined' is treated as null)_
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s2 = pl.Series("b", [4, 5, 6])
   * s.series_equal(s)
   * true
   * s.series_equal(s2)
   * false
   * ```
   */
  seriesEqual<U>(other: Series<U>, nullEqual?:boolean): boolean
  seriesEqual<U>(other: Series<U>, opt: {nullEqual?:boolean}): boolean
  /**
   * __Set masked values__
   * @param filter Boolean mask
   * @param value value to replace masked values with
   */
  set(filter: Series<boolean>, value: T): Series<T>
  setAtIdx(indices: number[] | Series<number>,  value:T): Series<T>
  /**
   * __Shift the values by a given period__
   *
   * the parts that will be empty due to this operation will be filled with `null`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s.shift(1)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         null
   *         1
   *         2
   * ]
   * s.shift(-1)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         2
   *         3
   *         null
   * ]
   * ```
   */
  shift(opts: {periods?:number}): Series<T>
  shift(periods?:number): Series<T>
  /**
   * Shift the values by a given period
   *
   * the parts that will be empty due to this operation will be filled with `fillValue`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @param fillValue - Fill null & undefined values with the result of this expression.
   */
  shiftAndFill(opt: {periods: number, fillValue: any}): Series<T>
  shiftAndFill(periods: number, fillValue: any): Series<T>
  /**
   * __Shrink memory usage of this Series to fit the exact capacity needed to hold the data.__
   * @param inPlace - Modify the Series in-place.
   */
  shrinkToFit(inPlace?:boolean): Series<T> | void
  shrinkToFit(opt: {inPlace:boolean}): Series<T> | void
  /**
   * __Compute the sample skewness of a data set.__
   *
   * For normally distributed data, the skewness should be about zero. For
   * unimodal continuous distributions, a skewness value greater than zero means
   * that there is more weight in the right tail of the distribution. The
   * function `skewtest` can be used to determine if the skewness value
   * is close enough to zero, statistically speaking.
   * ___
   * @param bias - If false, then the calculations are corrected for statistical bias.
   */
  skew(bias?:boolean): number | undefined
  skew(opt: {bias?:boolean}): number | undefined
  /**
   * Create subslices of the Series.
   *
   * @param start - Start of the slice (negative indexing may be used).
   * @param length - length of the slice.
   */
  slice(opt: {start: number, length: number}): Series<T>
  slice(start: number, length?: number): Series<T>
  /**
   * __Sort this Series.__
   * @param reverse - Reverse sort
   * @example
   * ```
   * s = pl.Series("a", [1, 3, 4, 2])
   * s.sort()
   * shape: (4,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   * ]
   * s.sort(true)
   * shape: (4,)
   * Series: 'a' [i64]
   * [
   *         4
   *         3
   *         2
   *         1
   * ]
   * ```
   */
  sort(): Series<T>
  sort(reverse?:boolean): Series<T>
  sort(options: {reverse:boolean}): Series<T>

  /**
   * Reduce this Series to the sum value.
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.sum()
   * 6
   * ```
   */
  sum(): number
  /**
   * __Get last N elements as Series.__
   *
   * ___
   * @param length - Length of the tail
   * @see {@link head}
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3])
   * s.tail(2)
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         2
   *         3
   * ]
   * ```
   */
  tail(length?: number): Series<T>
  /**
   * Take every nth value in the Series and return as new Series.
   * @param n - nth value to take
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3, 4])
   * s.takeEvery(2))
   * shape: (2,)
   * Series: '' [i64]
   * [
   *         1
   *         3
   * ]
   * ```
   */
  takeEvery(n: number): Series<T>
  /**
   * Take values by index.
   * ___
   * @param indices - Index location used for the selection
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 3, 4])
   * s.take([1, 3])
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         2
   *         4
   * ]
   * ```
   */
  take(indices: Array<number>): Series<T>
  /**
   * __Convert this Series to a Javascript Array.__
   *
   * This operation clones data.
   * ___
   * @example
   * ```
   * const s = pl.Series("a", [1, 2, 3])
   * const arr = s.toArray()
   * [1, 2, 3]
   * Array.isArray(arr)
   * true
   * ```
   */
  toArray(): Array<T>
  /**
   * __Get unique elements in series.__
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 2, 3])
   * s.unique()
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   * ```
   */
  unique(): Series<T>
  /**
   * __Count the unique values in a Series.__
   * ___
   * @example
   * ```
   * s = pl.Series("a", [1, 2, 2, 3])
   * s.valueCounts()
   * shape: (3, 2)
   * ╭─────┬────────╮
   * │ a   ┆ counts │
   * │ --- ┆ ---    │
   * │ i64 ┆ u32    │
   * ╞═════╪════════╡
   * │ 2   ┆ 2      │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   * │ 1   ┆ 1      │
   * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   * │ 3   ┆ 1      │
   * ╰─────┴────────╯
   * ```
   */
  valueCounts(): DataFrame
  /**
   * Where mask evaluates true, take values from self.
   *
   * Where mask evaluates false, take values from other.
   * ___
   * @param mask - Boolean Series
   * @param other - Series of same type
   *
   */
  zipWith<U>(mask: Series<boolean>, other: Series<T>): Series<U>
  /**
   * _Returns a Javascript representation of Series
   *
   * @example
   * ```
   * const s = pl.Series("foo", [1,2,3])
   * s.toJS()
   * {
   *   name: "foo",
   *   datatype: "Float64",
   *   values: [1,2,3]
   * }
   * ```
   */
  toJS(): object
  toFrame(): DataFrame
}

export const seriesWrapper = <T>(_s:JsSeries): Series<T> => {
  const unwrap = <U>(method: string, args?: object, _series = _s): U => {

    return pli.series[method]({_series, ...args });
  };
  const wrap = <U>(method, args?, _series = _s): Series<U> => {
    return seriesWrapper(unwrap(method, args, _series));
  };
  const noArgWrap = <U>(method: string) => () => wrap<U>(method);
  const noArgUnwrap = <U>(method: string) => () => unwrap<U>(method);
  const dtypeAccessor = (fn) => (method, args: {field, key}, _series = _s) => {
    const dtype = unwrap<DataType>("dtype");
    if (args.field?._series) {

      return fn(method, { [args.key]: args.field._series }, _series);
    } else {
      const dt = (DTYPE_TO_FFINAME as any)[DataType[dtype]];
      const internalMethod = `${method}_${dt}`;

      return fn(internalMethod, { [args.key]: args.field }, _series);
    }
  };
  const clone = <U>() => wrap<U>("clone");
  const alias = (name: string) => {
    const s = clone<T>();
    unwrap("rename", { name }, s._series);

    return s;
  };
  const concat = (other: Series<T>): Series<T> => {
    const s = clone<T>();

    unwrap("append", { other: other._series }, s._series);

    return s;

  };
  const diff = (opt?: any, nullBehavior="ignore"): any => {
    return wrap("diff", {
      n: opt?.n ?? (typeof opt === "number" ? opt : 1),
      null_behavior: opt?.nullBehavior ?? nullBehavior
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
  const isFinite = () => {
    const dtype = unwrap<keyof DataType>("dtype");

    if (![DataType.Float32, DataType.Float64].includes(DataType[dtype])) {
      throw new InvalidOperationError("isFinite", dtype);
    } else {
      return wrap("is_finite") as any;
    }
  };
  const isInfinite = () => {
    const dtype = unwrap<keyof DataType>("dtype");

    if (![DataType.Float32, DataType.Float64].includes(DataType[dtype])) {
      throw new InvalidOperationError("isInfinite", dtype);
    } else {
      return wrap("is_infinite") as any;
    }
  };
  const kurtosis = (obj?, bias=true) => {
    return unwrap<Optional<number>>("kurtosis", {
      fisher: obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true),
      bias : obj?.["bias"] ?? bias,
    });
  };
  const inPlaceOptional = (method: string) =>  (obj?: {inPlace:boolean} | boolean): any  => {
    if(obj === true || obj?.["inPlace"] === true) {
      unwrap(method, {inPlace: true});
    } else {
      return wrap(method);
    }
  };
  const reinterpret = (signed=true) => {
    const dtype = unwrap<string>("dtype");
    if ([DataType.UInt64, DataType.Int64].includes(DataType[dtype])) {
      return wrap("reinterpret", { signed }) as any;
    } else {
      throw new InvalidOperationError("reinterpret", dtype);
    }
  };
  const rename = (obj: any, inPlace = false): any => {
    if (obj?.inPlace ?? inPlace) {
      unwrap("rename", { name: obj?.name ?? obj });
    } else {
      return alias(obj?.name ?? obj);
    }
  };
  const rolling = (method: string) =>  (opts, weights?, minPeriods?, center?): Series<T> => {
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
  const sample = (opts?, frac?, withReplacement = false): Series<T> => {
    if (opts?.n || typeof opts === "number") {
      return wrap("sample_n", {
        n: opts?.n ?? opts,
        with_replacement: opts?.withReplacement ?? withReplacement
      });
    }
    if(opts?.frac ?? frac) {
      return wrap("sample_frac", {
        frac: opts?.frac ?? frac,
        with_replacement: withReplacement,
      });
    }
    else {
      throw new Error("must specify either 'frac' or 'n'");
    }
  };
  const seriesEqual = (other, opt: any = true) => unwrap<boolean>(
    "series_equal", {
      other: other._series,
      null_equal: opt?.nullEqual ?? opt
    }
  );
  const skew = (opt:any = true) => {
    return unwrap<number | undefined>("skew", {
      bias: opt?.bias ?? (typeof opt === "boolean" ? opt : true)
    });
  };

  const isNumeric = () => [
    DataType.Int8,
    DataType.Int16,
    DataType.Int32,
    DataType.Int64,
    DataType.UInt8,
    DataType.UInt16,
    DataType.UInt32,
    DataType.UInt64,
    DataType.Float32,
    DataType.Float64
  ].includes(DataType[unwrap<keyof DataType>("dtype")]);

  const shiftAndFill = <T>(opt, fillValue?) => {
    if(opt.periods) {
      return shiftAndFill(opt.periods, opt.fillValue);
    }
    const s = seriesWrapper<T>(_s);
    const name = s.name;

    return s.toFrame().select(
      col(name).shiftAndFill(opt, fillValue)
    )
      .getColumn(name);

  };
  const set = (filter: Series<boolean>, value:T): Series<T> => {
    const dtype = unwrap<string>("dtype");
    const dt = DTYPE_TO_FFINAME[DataType[dtype]];
    if(!dt) {
      throw todo();
    }

    return wrap(`set_with_mask_${dt}`, {filter: filter._series, value});
  };
  const setAtIdx = (indices: number[] | Series<number>, value:T): Series<T> => {
    const dtype = unwrap<string>("dtype");
    const dt = DTYPE_TO_FFINAME[DataType[dtype]];
    if(!dt) {
      throw todo();
    }
    indices = isSeries(indices) ? indices.cast(DataType.UInt32).toArray() : indices;

    return wrap(`set_at_idx_${dt}`, {indices, value});
  };

  const propOrVal = (obj: any, key: string) => ({[key]: obj?.[key] ?? obj});
  const propOrElse = (obj: any, key: string, otherwise: boolean) => ({[key]: obj?.[key] ?? obj ?? otherwise});

  const out = {
    _series: _s,
    [inspect](): string { return unwrap<string>("get_fmt");},
    *[Symbol.iterator]() {
      let start = 0;
      let len = unwrap<number>("len");
      while (start < len) {
        const v = this.get(start);
        start++;
        yield v;
      }
    },
    toString: () => unwrap<string>("get_fmt"),
    get [Symbol.toStringTag]() { return "Series";},
    get dtype(): DataType { return unwrap("dtype");},
    get name(): string { return unwrap("name");},
    get length(): number { return unwrap("len");},
    get str(): StringFunctions {return StringFunctions(_s);},
    get arr(): ListFunctions {return ListFunctions(_s);},
    eq: (field) => dtypeAccessor(wrap)("eq", {field, key: "rhs"}),
    equals: (field) => dtypeAccessor(wrap)("eq", {field, key: "rhs"}),
    gt_eq: (field) => dtypeAccessor(wrap)("gt_eq", {field, key: "rhs"}),
    greaterThanEquals: (field) => dtypeAccessor(wrap)("gt_eq", {field, key: "rhs"}),
    gt: (field) => dtypeAccessor(wrap)("gt", {field, key: "rhs"}),
    greaterThan: (field) => dtypeAccessor(wrap)("gt", {field, key: "rhs"}),
    lt_eq: (field) => dtypeAccessor(wrap)("lt_eq", {field, key: "rhs"}),
    lessThanEquals: (field) => dtypeAccessor(wrap)("lt_eq", {field, key: "rhs"}),
    lt: (field) => dtypeAccessor(wrap)("lt", {field, key: "rhs"}),
    lessThan: (field) => dtypeAccessor(wrap)("lt", {field, key: "rhs"}),
    neq: (field) => dtypeAccessor(wrap)("neq", {field, key: "rhs"}),
    notEquals: (field) => dtypeAccessor(wrap)("neq", {field, key: "rhs"}),
    add: (field) => dtypeAccessor(wrap)("add", {field, key: "other"}),
    sub: (field) => dtypeAccessor(wrap)("sub", {field, key: "other"}),
    div: (field) => dtypeAccessor(wrap)("div", {field, key: "other"}),
    mul: (field) => dtypeAccessor(wrap)("mul", {field, key: "other"}),
    rem: (field) => dtypeAccessor(wrap)("rem", {field, key: "other"}),
    plus: (field) => dtypeAccessor(wrap)("add", {field, key: "other"}),
    minus: (field) => dtypeAccessor(wrap)("sub", {field, key: "other"}),
    divide: (field) => dtypeAccessor(wrap)("div", {field, key: "other"}),
    times: (field) => dtypeAccessor(wrap)("mul", {field, key: "other"}),
    remainder: (field) => dtypeAccessor(wrap)("rem", {field, key: "other"}),
    abs: noArgWrap("abs"),
    alias,
    append: (other: Series<T>) => wrap("append", { other: other._series }),
    argMax: noArgUnwrap("arg_max"),
    argMin: noArgUnwrap("arg_min"),
    argTrue: noArgWrap("arg_true"),
    argUnique: noArgWrap("arg_unique"),
    argSort: (opt?: any) => wrap<T>("argsort", propOrElse(opt, "reverse", false)),
    bitand: (other) => wrap("bitand", { other: other._series }),
    bitor: (other) => wrap("bitor", { other: other._series }),
    bitxor: (other) => wrap("bitxor", { other: other._series }),
    cast: <D extends DataType>(dtype: D, opt?: any) => wrap<DtypeToPrimitive<D>>("cast", { dtype, ...propOrElse(opt, "strict", false)}),
    chunkLengths: noArgUnwrap("chunk_lengths"),
    clone,
    concat,
    cumMax: (opt?: any) => wrap<T>("cummax", propOrElse(opt, "reverse", false)),
    cumMin: (opt?: any) => wrap<T>("cummin", propOrElse(opt, "reverse", false)),
    cumProd: (opt?: any) => wrap<T>("cumprod", propOrElse(opt, "reverse", false)),
    cumSum: (opt?: any) => wrap<T>("cumsum", propOrElse(opt, "reverse", false)),
    diff,
    dot: (other) => unwrap("dot", { other: other._series }),
    dropNulls: noArgWrap("drop_nulls"),
    explode: noArgWrap("explode"),
    fillNull: (opt: any) =>  wrap<T>("fill_null", propOrVal(opt, "strategy")),
    filter: (opt: any) => wrap<T>("filter", propOrVal(opt, "predicate")),
    floor: noArgWrap("floor"),
    get: (field: any) => dtypeAccessor(unwrap)("get", {field, key: "index"}),
    getIndex: (idx) => unwrap("get_idx", {idx}),
    hasValidity: noArgUnwrap("has_validity"),
    hash,
    head: (length=5) => wrap("head", {length}),
    inner: () => _s,
    interpolate: noArgWrap("interpolate"),
    isBoolean: () => DataType[unwrap<keyof DataType>("dtype")] === DataType.Bool,
    isDateTime: () => [DataType.Date, DataType.Datetime].includes(DataType[unwrap<keyof DataType>("dtype")]),
    isDuplicated: noArgWrap("is_duplicated"),
    isFinite,
    isFirst: noArgWrap("is_first"),
    isFloat: () => [DataType.Float32, DataType.Float64].includes(DataType[unwrap<keyof DataType>("dtype")]),
    isIn: (other) => wrap("is_in", {other: (<any>other)?._series ?? Series(other)._series}),
    isInfinite,
    isNotNull: noArgWrap("is_not_null"),
    isNull: noArgWrap("is_null"),
    isNumeric,
    isUnique: noArgWrap("is_unique"),
    isUtf8: () => DataType[unwrap<keyof DataType>("dtype")] === DataType.Utf8,
    kurtosis,
    len: noArgUnwrap("len"),
    limit: (n=10) => wrap("limit", { num_elements: n }),
    max: noArgUnwrap("max"),
    mean: noArgUnwrap("mean"),
    median: noArgUnwrap("median"),
    min: noArgUnwrap("min"),
    mode: noArgWrap("mode"),
    nChunks: noArgUnwrap<number>("n_chunks"),
    nUnique: noArgUnwrap<number>("n_unique"),
    nullCount: noArgUnwrap<number>("null_count"),
    peakMax: noArgWrap("peak_max"),
    peakMin: noArgWrap("peak_min"),
    quantile: (quantile) => unwrap<number>("quantile", {quantile}),
    rank: (method="average") => wrap("rank", { method}),
    rechunk: inPlaceOptional("rechunk"),
    reinterpret,
    rename,
    rollingMax: rolling("rolling_max"),
    rollingMean: rolling("rolling_mean"),
    rollingMin: rolling("rolling_min"),
    rollingSum: rolling("rolling_sum"),
    rollingVar: rolling("rolling_var"),
    round: (o) => wrap("round", {decimals: o?.decimals ?? o}) as any,
    sample,
    seriesEqual,
    set,
    setAtIdx,
    shift: (opt: any = 1) => wrap<T>("shift", {periods: opt?.periods ?? opt}),
    shrinkToFit: inPlaceOptional("shrink_to_fit"),
    skew,
    slice: (opt:any, length?:any) => wrap<T>("slice", {offset: opt?.start ?? opt, length: opt?.length ?? length}),
    sort: (opt?) => wrap<T>("sort", propOrElse(opt, "reverse", false)),
    sum: noArgUnwrap("sum"),
    tail: (length=5) => wrap("tail", {length}),
    takeEvery: (n) => wrap("take_every", {n}),
    take: (indices) => wrap("take", {indices}),
    toArray: (): Array<T> => [...seriesWrapper(_s)] as any,
    unique: noArgWrap("unique"),
    zipWith: (mask, other) => wrap("zip_with", {mask: mask._series, other: other._series}),
    toJS: noArgUnwrap("to_js"),
    toFrame: () => dfWrapper(pli.df.read_columns({columns: [_s]})),
    apply: <U>(func: (s: T) => U): Series<U> => {throw todo();},
    map: <U>(func: (s: T) => U): Series<U> => wrap("map", {func}),
    describe: (): DataFrame => {throw todo();},
    shiftAndFill,
    valueCounts: () => dfWrapper(unwrap("value_counts")),

  } as Series<T>;

  return new Proxy(out, {
    get: function(target, prop, receiver) {
      if(typeof prop !== "symbol" && !Number.isNaN(Number(prop))) {
        return target.get(Number(prop));
      } else {
        return Reflect.get(target, prop, receiver);
      }
    }
  });
};

export function Series<V extends ArrayLike<any>>(values: V): ValueOrNever<V>
export function Series<V extends ArrayLike<any>>(name: string, values: V): ValueOrNever<V>
export function Series<T extends DataType, U extends ArrayLikeDataType<T>>(name: string, values: U, dtype: T): Series<DtypeToPrimitive<T>>
export function Series<T extends DataType, U extends boolean, V extends ArrayLikeOrDataType<T, U>>(name: string, values: V, dtype?: T, strict?: U): Series<DataTypeOrValue<T, U>>
export function Series(arg0: any, arg1?: any, dtype?: any, strict?: any) {
  if(typeof arg0 !== "string") {
    return Series("", arg0);
  }
  const _s = arrayToJsSeries(arg0, arg1, dtype, strict);

  return seriesWrapper(_s);
}
