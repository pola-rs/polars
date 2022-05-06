import pli from "../internals/polars_internal";
import {arrayToJsSeries} from "../internals/construction";
import {DataType, DTYPE_TO_FFINAME, Optional} from "../datatypes";
import {DataFrame, _DataFrame} from "../dataframe";
import {StringFunctions} from "./string";
import {SeriesListFunctions} from "./list";
import {SeriesDateFunctions} from "./datetime";
import {SeriesStructFunctions} from "./struct";
import {InvalidOperationError} from "../error";
import {RankMethod} from "../utils";
import {Arithmetic, Comparison, Cumulative, Deserialize, Rolling, Round, Sample, Serialize} from "../shared_traits";
import {col} from "../lazy/functions";

const inspect = Symbol.for("nodejs.util.inspect.custom");
export interface Series extends
  ArrayLike<any>,
  Rolling<Series>,
  Arithmetic<Series>,
  Comparison<Series>,
  Cumulative<Series>,
  Round<Series>,
  Sample<Series>,
  Serialize {
  inner(): any
  name: string
  dtype: DataType
  str: StringFunctions
  lst: SeriesListFunctions,
  struct: SeriesStructFunctions,
  date: SeriesDateFunctions
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
  //  * __Apply a function over elements in this Series and return a new Series.__
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
   * Create a new Series that copies data from this Series without null values.
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
  hash(k0?: number | bigint, k1?: number | bigint, k2?: number | bigint, k3?: number | bigint): Series
  hash({k0, k1, k2, k3}: {k0?: number | bigint, k1?: number | bigint, k2?: number | bigint, k3?: number | bigint}): Series
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
  /** compat with `JSON.stringify */
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
export function _Series(_s: any): Series {
  const unwrap = (method: keyof any, ...args: any[]) => {
    return _s[method as any](...args);
  };
  const wrap = (method, ...args): Series => {
    return _Series(unwrap(method, ...args));
  };
  const  dtypeWrap = (method: string, ...args: any[])  => {
    const dtype = _s.dtype;

    const dt = (DTYPE_TO_FFINAME as any)[dtype];
    const internalMethod = `series${method}${dt}`;

    return _Series(pli[internalMethod](_s, ...args));
  };

  const dtypeUnwrap = (method: string, ...args: any[]) =>  {
    const dtype = _s.dtype;

    const dt = (DTYPE_TO_FFINAME as any)[dtype];
    const internalMethod = `series${method}${dt}`;

    return pli[internalMethod](_s, ...args);
  };

  const rolling =  (method: string, opts, weights?, minPeriods?, center?)  => {
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

    return _Series(_s[method](callOpts));
  };
  const series = {
    _s,
    [inspect]() {
      return _s.toString();
    },
    *[Symbol.iterator]() {
      let start = 0;
      let len = _s.len();
      while (start < len) {
        const v = _s.getIdx(start);
        start++;
        yield v as any;
      }
    },
    toString() {
      return _s.toString();
    },
    serialize(format) {
      return _s.serialize(format);
    },
    [Symbol.toStringTag]() {
      return "Series";
    },
    get dtype() {
      return _s.dtype as any;
    },
    get name() {
      return _s.name;
    },
    get length() {
      return _s.len();
    },
    get str() {
      return StringFunctions(_s);
    },
    get lst() {
      return SeriesListFunctions(_s);
    },
    get date() {
      return SeriesDateFunctions(_s);
    },
    get struct() {
      return SeriesStructFunctions(_s);
    },
    abs() {
      return wrap("abs");
    },
    add(field) {
      return dtypeWrap("Add", field);
    },
    alias(name: string) {
      const s = _s.clone();
      s.rename(name);

      return _Series(s);
    },
    append(other: Series) {
      _s.append(other.inner());
    },
    argMax() {
      return _s.argMax();
    },
    argMin() {
      return _s.argMin();
    },
    argSort(reverse: any = false, nullsLast=true) {
      if(typeof reverse === "boolean") {
        return _Series(_s.argsort(reverse, nullsLast));
      }

      return _Series(_s.argsort(
        reverse.reverse,
        reverse.nullsLast ?? nullsLast
      ));
    },
    argTrue() {
      return _Series(_s.argTrue());
    },
    argUnique() {
      return _Series(_s.argUnique());
    },
    as(name) {
      return this.alias(name);
    },
    bitand(other) {
      return _Series(_s.bitand(other._s));
    },
    bitor(other) {
      return _Series(_s.bitor(other._s));
    },
    bitxor(other) {
      return _Series(_s.bitxor(other._s));
    },
    cast(dtype, strict = false) {
      return _Series(_s.cast(dtype, strict));
    },
    chunkLengths() {
      return _s.chunkLengths();
    },
    clone() {
      return _Series(_s.clone());
    },
    concat(other) {
      const s = _s.clone();
      s.append(other.inner());

      return _Series(s);
    },
    cumCount(reverse?) {
      return this
        .toFrame()
        .select(col(this.name).cumCount(reverse))
        .getColumn(this.name);
    },
    cumSum(reverse?) {
      return _Series(_s.cumsum(reverse));
    },
    cumMax(reverse?) {
      return _Series(_s.cummax(reverse));
    },
    cumMin(reverse?) {
      return _Series(_s.cummin(reverse));
    },
    cumProd(reverse?) {
      return _Series(_s.cumprod(reverse));
    },
    describe() {
      let s = this.clone();
      let stats = {};
      if (!this.length) {
        throw new RangeError("Series must contain at least one value");
      }
      if (this.isNumeric()) {
        s = s.cast(DataType.Float64);
        stats = {
          "min": s.min(),
          "max": s.max(),
          "null_count": s.nullCount(),
          "mean": s.mean(),
          "count": s.len(),
        };
      } else if (s.isBoolean()) {
        stats = {
          "sum": s.sum(),
          "null_count": s.nullCount(),
          "count": s.len(),
        };
      } else if (s.isUtf8()) {
        stats = {
          "unique": s.nUnique(),
          "null_count": s.nullCount(),
          "count": s.len(),
        };
      } else {
        throw new InvalidOperationError("describe", s.dtype);
      }

      return DataFrame({
        "statistic": Object.keys(stats),
        "value": Object.values(stats)
      });
    },
    diff(n: any = 1, nullBehavior = "ignore") {
      return typeof n === "number" ?
        _Series(_s.diff(n, nullBehavior)) :
        _Series(_s.diff(n?.n ?? 1, n.nullBehavior ?? nullBehavior));
    },
    div(field: Series) {
      return dtypeWrap("Div", field);
    },
    divideBy(field: Series) {
      return this.div(field);
    },
    dot(other: Series) {
      return wrap("dot", (other as any)._s) as any;
    },
    dropNulls() {
      return wrap("dropNulls");
    },
    eq(field) {
      return dtypeWrap("Eq", field);
    },
    equals(field: Series) {
      return this.eq(field);
    },
    explode() {
      return wrap("explode");
    },
    extend(value, n) {
      return wrap("extendConstant", value, n);
    },
    extendConstant(value, n) {
      return wrap("extendConstant", value, n);
    },
    fillNull(strategy) {
      return typeof strategy === "string" ?
        wrap("fillNull", strategy) :
        wrap("fillNull", strategy.strategy);
    },
    filter(predicate) {
      return Series.isSeries(predicate) ?
        wrap("filter", (predicate as any)._s) :
        wrap("filter", (SeriesConstructor("", predicate) as any)._s);
    },
    get(field) {
      return dtypeUnwrap("Get", field);
    },
    getIndex(idx) {
      return _s.getIdx(idx);
    },
    gt(field) {
      return dtypeWrap("Gt", field);
    },
    greaterThan(field) {
      return this.gt(field);
    },
    gtEq(field) {
      return dtypeWrap("GtEq", field);
    },
    greaterThanEquals(field) {
      return this.gtEq(field);
    },
    hash(obj: any = 0n, k1 = 1n, k2 = 2n, k3 = 3n) {
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
    hasValidity() {
      return _s.hasValidity();
    },
    head(length = 5) {
      return wrap("head", length);
    },
    inner() {
      return _s;
    },
    interpolate() {
      return wrap("interpolate");
    },
    isBoolean() {
      const dtype = this.dtype;

      return dtype === pli.DataType.Bool;
    },
    isDateTime() {
      const dtype = this.dtype;

      return [pli.DataType.Date, pli.DataType.Datetime].includes(dtype);
    },
    isDuplicated() {
      return wrap("isDuplicated");
    },
    isFinite() {
      const dtype = this.dtype;

      if (![pli.DataType.Float32, pli.DataType.Float64].includes(dtype)) {
        throw new InvalidOperationError("isFinite", dtype);
      } else {
        return wrap("isFinite");
      }
    },
    isFirst() {
      return wrap("isFirst");
    },
    isFloat() {
      const dtype = this.dtype;

      return [pli.DataType.Float32, pli.DataType.Float64].includes(dtype);
    },
    isIn(other) {
      return Series.isSeries(other) ?
        wrap("isIn", (other as any)._s) :
        wrap("isIn", (Series("", other) as any)._s);
    },
    isInfinite() {
      const dtype = this.dtype;

      if (![pli.DataType.Float32, pli.DataType.Float64].includes(dtype)) {
        throw new InvalidOperationError("isFinite", dtype);
      } else {
        return wrap("isInfinite");
      }
    },
    isNotNull() {
      return wrap("isNotNull");
    },
    isNull() {
      return wrap("isNull");
    },
    isNaN() {
      return wrap("isNan");
    },
    isNotNaN() {
      return wrap("isNotNan");
    },
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
    },
    isUnique() {
      return wrap("isUnique");
    },
    isUtf8() {
      return this.dtype === pli.DataType.Utf8;
    },
    kurtosis(fisher: any = true, bias = true) {
      if (typeof fisher === "boolean") {
        return _s.kurtosis(fisher, bias);
      }
      const d =  {
        fisher: true,
        bias,
        ...fisher
      };

      return _s.kurtosis(d.fisher, d.bias);
    },
    len() {
      return this.length;
    },
    lt(field) {
      return dtypeWrap("Lt", field);
    },
    lessThan(field) {
      return dtypeWrap("Lt", field);
    },
    ltEq(field) {
      return dtypeWrap("LtEq", field);
    },
    lessThanEquals(field) {
      return dtypeWrap("LtEq", field);
    },
    limit(n=10) {
      return wrap("limit", n);
    },
    max() {
      return _s.max() as any;
    },
    mean() {
      return _s.mean() as any;
    },
    median() {
      return _s.median() as any;
    },
    min() {
      return _s.min() as any;
    },
    mode() {
      return wrap("mode");
    },
    minus(other) {
      return dtypeWrap("Sub", other);
    },
    mul(other) {
      return dtypeWrap("Mul", other);
    },
    nChunks() {
      return _s.nChunks();
    },
    neq(other) {
      return dtypeWrap("Neq", other);
    },
    notEquals(other) {
      return this.neq(other);
    },
    nullCount() {
      return _s.nullCount();
    },
    nUnique() {
      return _s.nUnique();
    },
    peakMax() {
      return wrap("peakMax");
    },
    peakMin() {
      return wrap("peakMin");
    },
    plus(other) {
      return dtypeWrap("Add", other);
    },
    quantile(quantile, interpolation = "nearest") {
      return _s.quantile(quantile, interpolation);
    },
    rank(method = "average", reverse = false) {
      return wrap("rank", method, reverse);
    },
    rechunk(inPlace = false) {
      return wrap("rechunk", inPlace);
    },
    reinterpret(signed = true) {
      const dtype = this.dtype;
      if ([pli.DataType.UInt64, pli.DataType.Int64].includes(dtype)) {
        return wrap("reinterpret", signed);
      } else {
        throw new InvalidOperationError("reinterpret", dtype);
      }
    },
    rem(field) {
      return dtypeWrap("Rem", field);
    },
    modulo(field) {
      return this.rem(field);
    },
    rename(obj: any, inPlace = false): any {
      if (obj?.inPlace ?? inPlace) {
        _s.rename(obj?.name ?? obj);
      } else {
        return this.alias(obj?.name ?? obj);
      }
    },
    rollingMax(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingMax", windowSize, weights, minPeriods, center);
    },
    rollingMean(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingMean", windowSize, weights, minPeriods, center);
    },
    rollingMin(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingMin", windowSize, weights, minPeriods, center);
    },
    rollingSum(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingSum", windowSize, weights, minPeriods, center);
    },
    rollingStd(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingStd", windowSize, weights, minPeriods, center);
    },
    rollingVar(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingVar", windowSize, weights, minPeriods, center);
    },
    rollingMedian(windowSize, weights?, minPeriods?, center? ) {
      return rolling("rollingMedian", windowSize, weights, minPeriods, center);
    },
    rollingQuantile(val, interpolation = "nearest", windowSize?, weights?, minPeriods?, center?) {
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
    rollingSkew(windowSize, bias?) {
      return this
        .toFrame()
        .select(col(this.name).rollingSkew(windowSize, bias))
        .getColumn(this.name);
    },
    floor() {
      return wrap("floor");
    },
    ceil() {
      return wrap("ceil");
    },
    round(opt): any {
      if (this.isNumeric()) {
        if (typeof opt === "number") {
          return wrap("round", opt);
        } else {
          return wrap("round", opt.decimals);
        }

      } else {
        throw new InvalidOperationError("round", this.dtype);
      }
    },
    clip(arg, max?) {
      return this
        .toFrame()
        .select(
          col(this.name).clip(arg, max)
        )
        .getColumn(this.name);
    },
    setAtIdx(indices, value) {
      indices = Series.isSeries(indices) ? indices.cast(DataType.UInt32).toArray() : indices;

      return dtypeWrap("SetAtIdx", indices, value);
    },
    set(mask, value) {
      mask = Series.isSeries(mask) ? mask : Series.from(mask);

      return dtypeWrap("SetWithMask", mask.inner(), value);
    },
    sample(opts?, frac?, withReplacement = false, seed?) {
      if(arguments.length === 0) {
        return wrap("sampleN",
          1,
          withReplacement,
          false,
          seed
        );
      }
      if(opts?.n  !== undefined || opts?.frac  !== undefined) {
        return this.sample(opts.n, opts.frac, opts.withReplacement, seed);
      }
      if (typeof opts === "number") {
        return wrap("sampleN",
          opts,
          withReplacement,
          false,
          seed
        );
      }
      if(typeof frac === "number") {
        return wrap("sampleFrac",
          frac,
          withReplacement,
          seed
        );
      }
      else {
        throw new TypeError("must specify either 'frac' or 'n'");
      }

    },
    seriesEqual(other, nullEqual: any = true, strict=false) {
      return _s.seriesEqual(other._s, nullEqual, strict);
    },
    shift(periods=1) {
      return wrap("shift", periods);
    },
    shiftAndFill(periods, fillValue?) {
      return this
        .toFrame()
        .select(
          col(this.name).shiftAndFill(periods, fillValue)
        )
        .getColumn(this.name);
    },
    shrinkToFit(inPlace?: boolean) {
      if(inPlace) {
        _s.shrinkToFit();
      } else {
        const s = this.clone();
        s.shrinkToFit();

        return s as any;
      }

    },
    skew(bias: any = true) {
      if (typeof bias === "boolean") {
        return _s.skew(bias) as any;
      }

      return _s.skew(bias?.bias ?? true) as any;
    },
    slice(offset, length?) {
      if (typeof offset === "number") {
        return wrap("slice", offset, length);
      }

      return wrap("slice", offset.offset, offset.length);
    },
    sort(reverse?) {
      if (typeof reverse === "boolean") {
        return wrap("sort", reverse);
      }

      return wrap("sort", reverse?.reverse ?? false);

    },
    sub(field) {
      return dtypeWrap("Sub", field);
    },
    sum() {
      return _s.sum() as any;
    },
    tail(length=5) {
      return wrap("tail", length);
    },
    take(indices) {
      return wrap("take", indices);
    },
    takeEvery(n) {
      return wrap("takeEvery", n);
    },
    multiplyBy(field) {
      return this.mul(field);
    },
    toArray() {
      return _s.toArray();
    },
    toTypedArray() {
      if(!this.hasValidity()) {
        return _s.toTypedArray();
      } else {
        throw new Error("data contains nulls, unable to convert to TypedArray");
      }
    },
    toFrame() {
      return _DataFrame(new pli.JsDataFrame([_s]));
    },
    toBinary() {
      return _s.toBinary();
    },
    toJSON(...args: any[]) {
      // this is passed by `JSON.stringify` when calling `toJSON()`
      if(args[0] === "") {
        return _s.toJs();
      }

      return _s.serialize("json").toString();
    },
    toObject() {
      return _s.toJs();
    },
    unique(maintainOrder?) {
      if(maintainOrder) {
        return wrap("uniqueStable");
      } else {
        return wrap("unique");
      }
    },
    valueCounts() {
      return null as any;
    },
    values() {
      return this[Symbol.iterator]();
    },
    zipWith(mask, other) {
      return wrap("zipWith", mask._s, other._s);
    }
  };

  return new Proxy(series, {
    get: function (target, prop, receiver) {
      if (typeof prop !== "symbol" && !Number.isNaN(Number(prop))) {
        return target.get(Number(prop));
      } else {
        return Reflect.get(target, prop, receiver);
      }
    },
    set: function (series, prop, input): any {
      if (typeof prop !== "symbol" && !Number.isNaN(Number(prop))) {
        series.setAtIdx([Number(prop)], input);

        return true;
      }
    }
  });
}

export interface SeriesConstructor extends Deserialize<Series> {
  (values: any): Series
  (name: string, values: any[], dtype?): Series

  /**
   * Creates an array from an array-like object.
   * @param arrayLike — An array-like object to convert to an array.
   */
  from<T>(arrayLike: ArrayLike<T>): Series
  from<T>(name: string, arrayLike: ArrayLike<T>): Series
   /**
   * Returns a new Series from a set of elements.
   * @param items — A set of elements to include in the new Series object.
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

    return _Series(_s) as any;
  }

  return SeriesConstructor("", arg0);
};
const isSeries = (anyVal: any): anyVal is Series =>  {
  try {
    return anyVal?.[Symbol.toStringTag]?.() === "Series";
  } catch (err) {
    return false;
  }
};

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
  of,
  deserialize: (buf, fmt) => _Series(pli.JsSeries.deserialize(buf, fmt))
});
