import pli from "../internals/polars_internal";
import {arrayToJsSeries} from "../internals/construction";
import {DataType, DTYPE_TO_FFINAME, Optional} from "../datatypes";
import {DataFrame} from "../dataframe";
import {StringFunctions} from "./string";
// import {ListFunctions} from "./list";
// import {DateTimeFunctions} from "./datetime";
import {InvalidOperationError, todo} from "../error";
import {RankMethod} from "../utils";
// import {col} from "../lazy/functions";
// import {isExternal, isTypedArray} from "util/types";
import {Arithmetic, Comparison, Cumulative, Rolling, Round, Sample} from "../shared_traits";

const inspect = Symbol.for("nodejs.util.inspect.custom");
// export interface Foo {}
export interface Series extends
  ArrayLike<any>,
  Rolling<Series>,
  Arithmetic<Series>,
  Comparison<Series>,
  Cumulative<Series>,
  Round<Series>,
  Sample<Series> {
  inner(): pli.PySeries
  name: string
  dtype: DataType
  str: StringFunctions
  [inspect](): string;
  [Symbol.iterator](): IterableIterator<any>;
  // inner(): JsSeries
  bitand(other: Series): Series
  bitor(other: Series): Series
  bitxor(other: Series): Series
  /**
   * Take absolute values
   */
  abs(): Series
  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @see {@link rename}
   *
   */
  alias(name: string): Series
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
  append(other: Series): void
  // TODO!
  // /**
  //  * __Apply a function over elements in this Series and return a new _Series.__
  //  *
  //  * If the function returns another datatype, the returnType arg should be set, otherwise the method will fail.
  //  * ___
  //  * @param {CallableFunction} func - function or lambda.
  //  * @param {DataType} returnType - Output datatype. If none is given, the same datatype as this Series will be used.
  //  * @returns {SeriesType} `Series<T> | Series<returnType>`
  //  * @example
  //  * ```
  //  * > const s = pl.Series("a", [1, 2, 3])
  //  * > s.apply(x => x + 10)
  //  * shape: (3,)
  //  * Series: 'a' [i64]
  //  * [
  //  *         11
  //  *         12
  //  *         13
  //  * ]
  //  * ```
  //  */
  // apply<U>(func: (s: T) => U): Series<U>
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
  argTrue(): Series
  /**
   * Get unique index as Series.
   */
  argUnique(): Series
  /**
   * Index location of the sorted variant of this Series.
   * ___
   * @param reverse
   * @return {SeriesType} indexes - Indexes that can be used to sort this array.
   */
  argSort(): Series
  argSort(reverse: boolean): Series
  argSort({reverse}: {reverse: boolean}): Series
  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @see {@link rename} {@link alias}
   *
   */
  as(name: string): Series
  /**
   * Cast between data types.
   */
  cast(dtype: DataType, strict?: boolean): Series
  /**
   * Get the length of each individual chunk
   */
  chunkLengths(): Array<any>
  /**
   * Cheap deep clones.
   */
  clone(): Series
  concat(other: Series): Series

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
  diff(n: number, nullBehavior: "ignore" | "drop"): Series
  diff({n, nullBehavior}: {n: number, nullBehavior: "ignore" | "drop"}): Series
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
  dot(other: Series): number | undefined | null
  /**
   * Create a new _Series that copies data from this Series without null values.
   */
  dropNulls(): Series
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
  * Extend the Series with given number of values.
  * @param value The value to extend the Series with. This value may be null to fill with nulls.
  * @param n The number of values to extend.
  * @deprecated
  * @see {@link extendConstant}
  */
  extend(value: any, n: number): Series
  /**
   * Extend the Series with given number of values.
   * @param value The value to extend the Series with. This value may be null to fill with nulls.
   * @param n The number of values to extend.
   */
  extendConstant(value: any, n: number): Series
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
  fillNull(strategy: "backward" | "forward" | "min" | "max" | "mean" | "one" | "zero"): Series
  fillNull({strategy}: {strategy: "backward" | "forward" | "min" | "max" | "mean" | "one" | "zero"}): Series
  /**
   * __Filter elements by a boolean mask.__
   * @param {SeriesType} predicate - Boolean mask
   *
   */
  filter(predicate: Series): Series
  filter({predicate}: {predicate: Series}): Series
  get(index: number): any
  getIndex(n: number): any
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
  hash(k0?: number, k1?: number, k2?: number, k3?: number): Series
  hash({k0, k1, k2, k3}: {k0?: number, k1?: number, k2?: number, k3?: number}): Series
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
  head(length?: number): Series
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
  interpolate(): Series
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
  isDuplicated(): Series
  /**
   * Get mask of finite values if Series dtype is Float.
   */
  isFinite(): Series
  /**
   * Get a mask of the first unique value.
   */
  isFirst(): Series
  /**
   * Check if this Series is a Float.
   */
  isFloat(): boolean
  /**
   * Check if elements of this Series are in the right Series, or List values of the right Series.
   */
  isIn<U>(other: Series | U[]): Series
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
  isInfinite(): Series
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
  isNotNull(): Series
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
  isNull(): Series
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
  isUnique(): Series
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
  kurtosis(fisher: boolean, bias?: boolean): Optional<number>
  kurtosis({fisher, bias}: {fisher?: boolean, bias?: boolean}): Optional<number>
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
  limit(n?: number): Series
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
  mode(): Series
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
  peakMax(): Series
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
  peakMin(): Series
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
  quantile(quantile: number, interpolation?: string): number
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
  rank(method?: RankMethod): Series
  rechunk(): Series
  rechunk(inPlace: true): Series
  rechunk(inPlace: false): void
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
  reinterpret(signed?: boolean): Series
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
  rename(name: string): Series;
  rename(name: string, inPlace: boolean): void
  rename({name, inPlace}: {name: string, inPlace?: boolean}): void
  rename({name, inPlace}: {name: string, inPlace: true}): void

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
  seriesEqual<U>(other: Series, nullEqual?: boolean, strict?: boolean): boolean
  /**
   * __Set masked values__
   * @param filter Boolean mask
   * @param value value to replace masked values with
   */
  set(filter: Series, value: any): Series
  setAtIdx(indices: number[] | Series, value: any): Series
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
  shift(periods: number): Series
  /**
   * Shift the values by a given period
   *
   * the parts that will be empty due to this operation will be filled with `fillValue`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @param fillValue - Fill null & undefined values with the result of this expression.
   */
  shiftAndFill(periods: number, fillValue: any): Series
  shiftAndFill(args: {periods: number, fillValue: any}): Series

  /**
   * __Shrink memory usage of this Series to fit the exact capacity needed to hold the data.__
   * @param inPlace - Modify the Series in-place.
   */
  shrinkToFit(): Series
  shrinkToFit(inPlace: true): void
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
  skew(bias?: boolean): number | undefined
  /**
   * Create subslices of the Series.
   *
   * @param offset - Start of the slice (negative indexing may be used).
   * @param length - length of the slice.
   */
  slice(start: number, length?: number): Series
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
  sort(): Series
  sort(reverse?: boolean): Series
  sort(options: {reverse: boolean}): Series
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
  tail(length?: number): Series
  /**
   * Take every nth value in the Series and return as new _Series.
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
  takeEvery(n: number): Series
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
  take(indices: Array<number>): Series

  /**
   * __Get unique elements in series.__
   * ___
   * @param maintainOrder Maintain order of data. This requires more work.
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
  unique(maintainOrder?: boolean | {maintainOrder: boolean}): Series
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
  zipWith(mask: Series, other: Series): Series

  /**
   * __Convert this Series to a Javascript Array.__
   *
   * This operation clones data, and is very slow, but maintains greater precision for all dtypes.
   * Often times `series.toObject().values` is faster, but less precise
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
  toArray(): Array<any>
  /**
   * Converts series to a javascript typedArray.
   *
   * __Warning:__
   * This will throw an error if you have nulls, or are using non numeric data types
   */
  toTypedArray(): any

  /**
   * _Returns a Javascript object representation of Series_
   * Often this is much faster than the iterator, or `values` method
   *
   * @example
   * ```
   * const s = pl.Series("foo", [1,2,3])
   * s.toObject()
   * {
   *   name: "foo",
   *   datatype: "Float64",
   *   values: [1,2,3]
   * }
   * ```
   */
  toObject(): {name: string, datatype: string, values: any[]}
  toFrame(): DataFrame
  /** serializes the Series to a [bincode buffer](https://docs.rs/bincode/latest/bincode/index.html)
   * @example
   * pl.Series.fromBinary(series.toBincode())
   */
  toBinary(): Buffer
  toJSON(): string
  /** Returns an iterator over the values */
  values(): IterableIterator<any>
}

export interface RollingOptions {
  windowSize: number
  weights?: Array<number>
  minPeriods: number
  center: boolean
}

export class _Series {
  static isSeries = (anyVal: any): anyVal is Series => anyVal[Symbol.toStringTag]() === "Series";
  static from = (name, values?: ArrayLike<any>): Series => {
    if(Array.isArray(name) ){
      return SeriesConstructor("", values);

    } else {
      return SeriesConstructor(name, values);
    }
  };
  static of = (...values: any[]): Series => {
    return Series.from(values);
  };
  #s: pli.PySeries;
  #wrap(s) {
    return new _Series(s) as any;
  }
  #w(method: keyof pli.PySeries, ...args: any[]) {
    return new _Series(this.#s[method as any](...args)) as any;
  }
  #dtypeWrap(method: string, ...args: any[]) {
    const dtype = this.dtype;

    const dt = (DTYPE_TO_FFINAME as any)[dtype];
    const internalMethod = `series${method}${dt}`;

    return this.#wrap(pli[internalMethod](this.#s, ...args));
  }
  #dtypeUnwrap(method: string, ...args: any[]) {
    const dtype = this.dtype;

    const dt = (DTYPE_TO_FFINAME as any)[dtype];
    const internalMethod = `series${method}${dt}`;

    return pli[internalMethod](this.#s, ...args);
  }

  #rolling (method: string, opts, weights?, minPeriods?, center?) {
    const windowSize = opts?.["windowSize"] ?? (typeof opts === "number" ? opts : null);
    if (windowSize === null) {
      throw new Error("window size is required");
    }
    const callOpts = {
      windowSize: opts?.["windowSize"] ?? (typeof opts === "number" ? opts : null),
      weights: opts?.["weights"] ?? weights,
      minPeriods: opts?.["minPeriods"] ?? minPeriods ?? windowSize,
      center: opts?.["center"] ?? center ?? false,
    };

    return this.#w(method as any, callOpts);
  }
  constructor(s: pli.PySeries) {
    this.#s = s;
  }
  [inspect]() {
    return this.#s.toString();
  }
  *[Symbol.iterator]() {
    let start = 0;
    let len = this.#s.len();
    while (start < len) {
      // const v = this.#s.get(start);
      start++;
      yield null as any;
    }
  }
  toString() {
    return this.#s.toString();
  }
  get [Symbol.toStringTag]() {
    return "Series";
  }
  get dtype() {
    return this.#s.dtype;
  }
  get name() {
    return this.#s.name;
  }
  get length() {
    return this.#s.len();
  }
  get str() {
    return StringFunctions(this.#s);
  }
  get lst() {
    return null as any;
  }
  get date() {
    return null as any;
  }
  abs() {
    return this.#w("abs");
  }
  add(field) {
    return this.#dtypeWrap("Add", field);
  }
  alias(name: string) {
    const s = this.#s.clone();
    s.rename(name);

    return this.#wrap(s);
  }
  append(other: Series) {
    this.#s.append(other.inner());
  }
  argMax() {
    return this.#s.argMax();
  }
  argMin() {
    return this.#s.argMin();
  }
  argSort(reverse: any = false, nullsLast=true) {
    if(typeof reverse === "boolean") {
      return this.#wrap(this.#s.argsort(reverse, nullsLast));
    }

    return this.#wrap(this.#s.argsort(
      reverse.reverse,
      reverse.nullsLast ?? nullsLast
    ));
  }
  argTrue() {
    return this.#wrap(this.#s.argTrue());
  }
  argUnique() {
    return this.#wrap(this.#s.argUnique());
  }
  as(name) {
    return this.alias(name);
  }
  bitand(other) {
    return this.#wrap(this.#s.bitand(other.#s));
  }
  bitor(other) {
    return this.#wrap(this.#s.bitor(other.#s));
  }
  bitxor(other) {
    return this.#wrap(this.#s.bitxor(other.#s));
  }
  cast(dtype, strict = false) {
    return this.#wrap(this.#s.cast(dtype, strict));
  }
  chunkLengths() {
    return this.#s.chunkLengths();
  }
  clone() {
    return this.#wrap(this.#s.clone());
  }
  concat(other) {
    const s = this.#s.clone();
    s.append(other.inner());

    return this.#wrap(s);
  }
  cumSum(reverse?) {
    return this.#wrap(this.#s.cumsum(reverse));
  }
  cumMax(reverse?) {
    return this.#wrap(this.#s.cummax(reverse));
  }
  cumMin(reverse?) {
    return this.#wrap(this.#s.cummin(reverse));
  }
  cumProd(reverse?) {
    return this.#wrap(this.#s.cumprod(reverse));
  }
  diff(n: any = 1, nullBehavior = "ignore") {
    return typeof n === "number" ?
      this.#wrap(this.#s.diff(n, nullBehavior)) :
      this.#wrap(this.#s.diff(n?.n ?? 1, n.nullBehavior ?? nullBehavior));
  }
  div(field: Series) {
    return this.#dtypeWrap("Div", field);
  }
  divideBy(field: Series) {
    return this.div(field);
  }
  dot(other: Series) {
    return this.#w("dot", (other as any).#s) as any;
  }
  dropNulls() {
    return this.#w("dropNulls");
  }
  eq(field) {
    return this.#dtypeWrap("Eq", field);
  }
  equals(field: Series) {
    return this.eq(field);
  }
  explode() {
    return this.#w("explode");
  }
  extend(value, n) {
    return this.#w("extendConstant", value, n);
  }
  extendConstant(value, n) {
    return this.#w("extendConstant", value, n);
  }
  fillNull(strategy) {
    return typeof strategy === "string" ?
      this.#w("fillNull", strategy) :
      this.#w("fillNull", strategy.strategy);
  }
  filter(predicate) {
    return Series.isSeries(predicate) ?
      this.#w("filter", (predicate as any).#s) :
      this.#w("filter", (SeriesConstructor("", predicate) as any).#s);
  }
  get(field) {
    return this.#dtypeUnwrap("Get", field);
  }
  getIndex(idx) {
    return this.#s.getIdx(idx);
  }
  gt(field) {
    return this.#dtypeWrap("Gt", field);
  }
  greaterThan(field) {
    return this.gt(field);
  }
  gtEq(field) {
    return this.#dtypeWrap("GtEq", field);
  }
  greaterThanEquals(field) {
    return this.gtEq(field);
  }
  hash(obj: any = 0n, k1 = 1n, k2 = 2n, k3 = 3n) {
    if (typeof obj === "number" || typeof obj === "bigint") {
      return this.#w("hash", BigInt(obj), BigInt(k1), BigInt(k2), BigInt(k3));
    }
    const o = { k0: obj, k1: k1, k2: k2, k3: k3, ...obj};

    return this.#w(
      "hash",
      BigInt(o.k0),
      BigInt(o.k1),
      BigInt(o.k2),
      BigInt(o.k3)
    );
  }
  hasValidity() {
    return this.#s.hasValidity();
  }
  head(length = 5) {
    return this.#w("head", length);
  }
  inner() {
    return this.#s;
  }
  interpolate() {
    return this.#w("interpolate");
  }
  isBoolean() {
    const dtype = this.dtype;

    return dtype === pli.DataType.Bool;
  }
  isDateTime() {
    const dtype = this.dtype;

    return [pli.DataType.Date, pli.DataType.Datetime].includes(dtype);
  }
  isDuplicated() {
    return this.#w("isDuplicated");
  }
  isFinite() {
    const dtype = this.dtype;

    if (![pli.DataType.Float32, pli.DataType.Float64].includes(dtype)) {
      throw new InvalidOperationError("isFinite", dtype);
    } else {
      return this.#w("isFinite");
    }
  }
  isFirst() {
    return this.#w("isFirst");
  }
  isFloat() {
    const dtype = this.dtype;

    return [pli.DataType.Float32, pli.DataType.Float64].includes(dtype);
  }
  isIn(other) {
    return Series.isSeries(other) ?
      this.#w("isIn", (other as any).#s) :
      this.#w("isIn", (_Series.from("", other) as any).#s);
  }
  isInfinite() {
    const dtype = this.dtype;

    if (![pli.DataType.Float32, pli.DataType.Float64].includes(dtype)) {
      throw new InvalidOperationError("isFinite", dtype);
    } else {
      return this.#w("isInfinite");
    }
  }
  isNotNull() {
    return this.#w("isNotNull");
  }
  isNull() {
    return this.#w("isNull");
  }
  isNaN() {
    return this.#w("isNan");
  }
  isNotNaN() {
    return this.#w("isNotNan");
  }
  isNumeric() {
    const dtype = this.dtype;

    const numericTypes = [
      pli.DataType.Int8,
      pli.DataType.Int16,
      pli.DataType.Int32,
      pli.DataType.Int64,
      pli.DataType.UInt8,
      pli.DataType.UInt16,
      pli.DataType.UInt32,
      pli.DataType.UInt64,
      pli.DataType.Float32,
      pli.DataType.Float64
    ];

    return numericTypes.includes(dtype);
  }
  isUnique() {
    return this.#w("isUnique");
  }
  isUtf8() {
    return this.dtype === pli.DataType.Utf8;
  }
  kurtosis(fisher: any = true, bias = true) {
    if (typeof fisher === "boolean") {
      return this.#s.kurtosis(fisher, bias);
    }
    const d =  {
      fisher: true,
      bias,
      ...fisher
    };

    return this.#s.kurtosis(d.fisher, d.bias);
  }
  len() {
    return this.length;
  }
  lt(field) {
    return this.#dtypeWrap("Lt", field);
  }
  lessThan(field) {
    return this.#dtypeWrap("Lt", field);
  }
  ltEq(field) {
    return this.#dtypeWrap("LtEq", field);
  }
  lessThanEquals(field) {
    return this.#dtypeWrap("LtEq", field);
  }
  limit(n=10) {
    return this.#w("limit", n);
  }
  max() {
    return this.#s.max();
  }
  mean() {
    return this.#s.mean();
  }
  median() {
    return this.#s.median();
  }
  min() {
    return this.#s.min();
  }
  mode() {
    return this.#w("mode");
  }
  minus(other) {
    return this.#dtypeWrap("Sub", other);
  }
  mul(other) {
    return this.#dtypeWrap("Mul", other);
  }
  nChunks() {
    return this.#s.nChunks();
  }
  neq(other) {
    return this.#dtypeWrap("Neq", other);
  }
  notEquals(other) {
    return this.neq(other);
  }
  nullCount() {
    return this.#s.nullCount();
  }
  nUnique() {
    return this.#s.nUnique();
  }
  peakMax() {
    return this.#w("peakMax");
  }
  peakMin() {
    return this.#w("peakMin");
  }
  plus(other) {
    return this.#dtypeWrap("Add", other);
  }
  quantile(quantile, interpolation = "nearest") {
    return this.#s.quantile(quantile, interpolation);
  }
  rank(method = "average", reverse = false) {
    return this.#w("rank", method, reverse);
  }
  rechunk(inPlace = false) {
    return this.#w("rechunk", inPlace);
  }
  reinterpret(signed = true) {
    const dtype = this.dtype;
    if ([pli.DataType.UInt64, pli.DataType.Int64].includes(dtype)) {
      return this.#w("reinterpret", signed);
    } else {
      throw new InvalidOperationError("reinterpret", dtype);
    }
  }
  rem(field) {
    return this.#dtypeWrap("Rem", field);
  }
  modulo(field) {
    return this.rem(field);
  }
  rename(obj: any, inPlace = false) {
    if (obj?.inPlace ?? inPlace) {
      this.#s.rename(obj?.name ?? obj);
    } else {
      return this.alias(obj?.name ?? obj);
    }
  }
  rollingMax(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingMax", windowSize, weights, minPeriods, center);
  }
  rollingMean(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingMean", windowSize, weights, minPeriods, center);
  }
  rollingMin(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingMin", windowSize, weights, minPeriods, center);
  }
  rollingSum(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingSum", windowSize, weights, minPeriods, center);
  }
  rollingStd(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingStd", windowSize, weights, minPeriods, center);
  }
  rollingVar(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingVar", windowSize, weights, minPeriods, center);
  }
  rollingMedian(windowSize, weights?, minPeriods?, center? ) {
    return this.#rolling("rollingMedian", windowSize, weights, minPeriods, center);
  }
  rollingQuantile(val, interpolation = "nearest", windowSize?, weights?, minPeriods?, center?) {
    if(typeof val === "number") {
      return this.#w("rollingQuantile", val, interpolation, {
        windowSize,
        weights,
        minPeriods,
        center
      });
    }

    return this.#w("rollingQuantile", val.quantile, val.interpolation, val);
  }
  rollingSkew(windowSize, bias?) {
    return null as any;

    // return this
    //   .toFrame()
    //   .select(col(this.name).rollingSkew(windowSize, bias))
    //   .getColumn(this.name);
  }
  floor() {
    return this.#w("floor");
  }
  ceil() {
    return this.#w("ceil");
  }
  round(opt): any {
    if (this.isNumeric()) {
      if (typeof opt === "number") {
        return this.#w("round", opt);
      } else {
        return this.#w("round", opt.decimals);
      }

    } else {
      throw new InvalidOperationError("round", this.dtype);
    }
  }
  clip(arg, max?) {
    return null as any;
  }
  setAtIdx(indices, value) {
    indices = Series.isSeries(indices) ? indices.cast(DataType.UInt32).toArray() : indices;

    return this.#dtypeWrap("SetAtIdx", indices, value);
  }
  set(mask, value) {
    mask = Series.isSeries(mask) ? mask : Series.from(mask);

    return this.#dtypeWrap("SetWithMask", mask.inner(), value);
  }
  sample(opts?, frac?, withReplacement = false, seed?) {
    if(arguments.length === 0) {
      return this.#w("sampleN",
        1,
        withReplacement,
        seed
      );
    }
    if(opts?.n  !== undefined || opts?.frac  !== undefined) {
      return this.sample(opts.n, opts.frac, opts.withReplacement, seed);
    }
    if (typeof opts === "number") {
      return this.#w("sampleN",
        opts,
        withReplacement,
        seed
      );
    }
    if(typeof frac === "number") {
      return this.#w("sampleFrac",
        frac,
        withReplacement,
        seed
      );
    }
    else {
      throw new TypeError("must specify either 'frac' or 'n'");
    }

  }
  seriesEqual(other, nullEqual: any = true, strict=false) {
    return this.#s.seriesEqual(other.#s, nullEqual, strict);

  }
  shift(periods=1) {
    return this.#w("shift", periods);
  }
  shiftAndFill(periods, fillValue) {
    return this
      .toFrame()
      .select(
        pli.col(this.name).shiftAndFill(periods, fillValue)
      )
      .getColumn(this.name);
  }
  shrinkToFit(inPlace?: boolean) {
    if(inPlace) {
      this.#s.shrinkToFit();
    } else {
      const s = this.clone();
      s.shrinkToFit();

      return s;
    }

  }
  skew(bias: any = true) {
    if (typeof bias === "boolean") {
      return this.#s.skew(bias);
    }
    this.#s.skew(bias?.bias ?? true);
  }
  slice(offset, length?) {
    if (typeof offset === "number") {
      return this.#w("slice", offset, length);
    }

    return this.#w("slice", offset.offset, offset.length);
  }
  sort(reverse?) {
    if (typeof reverse === "boolean") {
      return this.#w("sort", reverse);
    }

    return this.#w("sort", reverse?.reverse ?? false);

  }
  sub(field) {
    return this.#dtypeWrap("Sub", field);
  }
  sum() {
    return this.#s.sum();
  }
  tail(length=5) {
    return this.#w("tail", length);
  }
  take(indices) {
    return this.#w("take", indices);
  }
  takeEvery(n) {
    return this.#w("takeEvery", n);
  }
  multiplyBy(field) {
    return this.mul(field);
  }
  toArray() {
    return this.#s.toArray();
  }
  toTypedArray() {
    if(!this.hasValidity()) {
      return this.#s.toTypedArray();
    } else {
      throw new Error("data contains nulls, unable to convert to TypedArray");
    }

  }
  toFrame() {
    return null as any;
  }
  toBinary() {
    return this.#s.toBinary();
  }
  toJSON() {
    return this.#s.toJson();
  }
  toObject() {
    return JSON.parse(this.#s.toJson().toString());
  }
  unique(maintainOrder?) {
    if(maintainOrder) {
      return this.#w("uniqueStable");
    } else {
      return this.#w("unique");
    }
  }
  valueCounts() {
    return null as any;

  }
  values() {
    return this[Symbol.iterator]();
  }
  zipWith(mask, other) {
    return this.#w("zipWith", mask.#s, other.#s);
  }
}

export interface SeriesConstructor {
  (values: any): Series
  (name: string, values: any[], dtype?): Series

  /**
   * Creates an array from an array-like object.
   * @param arrayLike — An array-like object to convert to an array.
   */
  from<T>(arrayLike: ArrayLike<T>): Series
  from<T>(name: string, arrayLike: ArrayLike<T>): Series
   /**
   * Returns a new _Series from a set of elements.
   * @param items — A set of elements to include in the new _series object.
   */
  of<T>(...items: T[]): Series
  isSeries(arg: any): arg is Series;
  /**
   * @param binary used to serialize/deserialize series. This will only work with the output from series.toBinary().
  */
  // fromBinary(binary: Buffer): Series
}

let SeriesConstructor = function(arg0: any, arg1?: any, dtype?: any, strict?: any): Series {
  if (typeof arg0 === "string") {
    const _s = arrayToJsSeries(arg0, arg1, dtype, strict);

    return new _Series(_s) as any;
  }

  return SeriesConstructor("", arg0);
};
const isSeries = (anyVal: any): anyVal is Series => anyVal?.[Symbol.toStringTag] === "Series";
const from = (name, values?: ArrayLike<any>): Series => {
  if(Array.isArray(name) ){
    return SeriesConstructor("", values);

  } else {
    return SeriesConstructor(name, values);
  }
};
const of = (...values: any[]): Series => {
  return Series.from(values);
};

export const Series: SeriesConstructor = Object.assign(SeriesConstructor, {
  isSeries,
  from,
  of
});
