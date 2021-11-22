/* eslint-disable no-unused-vars */
import polarsInternal from './internals/polars_internal';
import { arrayToJsSeries } from './internals/construction';
import util from 'util';
import { DataType, DtypeToPrimitive, DTYPE_TO_FFINAME, Optional } from './datatypes';
// /**
//  * 'any' type representing the internal rust `JsSeries`
//  */
// Dtype.Float32 | Dtype.Float64 | Dtype.UInt8 | Dtype.UInt16 | Dtype.UInt32 
//  | Dtype.Int8 | Dtype.Int16 | Dtype.Int32 ? number
type JsSeries = any;


export class Series<T> {
  /**
   * @ignore
   */
  // eslint-disable-next-line no-undef
  [n:number]: T;
  /**
   * @ignore
   */
  [Symbol.isConcatSpreadable] = true;
  
  private _series: JsSeries;
  private internal = polarsInternal.series;

  /**
   * 
   * @ignore
   */
  constructor(series: JsSeries) {
    this._series = series;
    const len = this.internal.len({ _series: series });

    for (let index = 0; index < len; index++) {
      Object.defineProperty(this, `${index}`, {
        get() {
          return this.get(index);
        },
      });
    }
  }
  /**
   * Get the name of this Series.
   * @example
   * ```
   * > const s = pl.Series("a", [1,2,3])
   * > s.name
   * 'a'
   * ```
   */
  get name(): string {
    return this.unwrap('name');
  }
  /**
     * Get the data type of this Series.
     * ___
     * @example
     * ```
     * > const s = pl.Series("a", [1,2,3])
     * > s.dtype
     * 'Int64Type'
     * ```
     */
  get dtype(): DataType {
    return this.unwrap<DataType>('dtype');
  }
  /**
     * @see {@link len}
     */
  get length(): number {
    return this.unwrap<number>('len');
  }
  /**
     * A Series represents a single column in a polars DataFrame.
     * @param name - Name of the series. Will be used as a column name when used in a DataFrame.
     * @param {PolarsArrayLike} values - One-dimensional data in various forms. Supported are: Array, Series,
     * Set
     * @param {DataType} [dtype] - Polars dtype of the Series data. If not specified, the dtype is inferred.
     * @param [strict] - Throw error on numeric overflow
     *
     * @example
     * ```
     * > const s = pl.Series('a', [1,2,3]);
     * > s
     * shape: (3,)
     * Series: 'a' [i64]
     * [
     *         1
     *         2
     *         3
     * ]
     * // Notice that the dtype is automatically inferred as a polars Int64:
     * > s.dtype()
     * "Int64"
     *
     * // Constructing a Series with a specific dtype:
     * > const s2 = pl.Series('a', [1, 2, 3], dtype=pl.Float32);
     * > s2
     * shape: (3,)
     * Series: 'a' [f32]
     * [
     *         1
     *         2
     *         3
     * ]
     * ```
     */
  static of<V extends ArrayLike<any>>(
    name: string,
    values: V,
  ): V extends ArrayLike<infer U> ? Series<U> : never;
  static of<T extends DataType, U extends ArrayLike<DtypeToPrimitive<T>>>(
    name: string,
    values: U,
    dtype: T,
  ): Series<DtypeToPrimitive<T>>;
  static of<T extends DataType, U extends ArrayLike<S extends true ? DtypeToPrimitive<T> : DtypeToPrimitive<T> | null>, S extends boolean>(
    name: string,
    values: U,
    dtype?: T,
    strict?: S,
  ): Series<S extends true ? DtypeToPrimitive<T> : DtypeToPrimitive<T> | null> {
    const series = arrayToJsSeries(name, values as any, dtype, strict ?? false);

    return new Series(series) as any;
  }

  [util.inspect.custom](): string {
    return this.unwrap<any>('get_fmt');
  }
  *[Symbol.iterator]() {
    let start = 0;
    let len = this.unwrap<number>('len');

    while (start < len) {
      const v = this.get(start);
      start++;
      yield v;
    }
  }


  // javascript overrides for iteration, and console.log


  eq(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('eq', { field, key: 'rhs' });
  }
  gt_eq(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('gt_eq', { field, key: 'rhs' });
  }
  gt(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('gt', { field, key: 'rhs' });
  }
  lt_eq(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('lt_eq', { field, key: 'rhs' });
  }
  lt(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('lt', { field, key: 'rhs' });
  }
  neq(field: Series<T> | number | bigint | string): Series<boolean> {
    return this.wrapDtype('neq', { field, key: 'rhs' });
  }

  rem(field: Series<T> | number | bigint | string): Series<boolean>  {
    return this.wrapDtype('rem', { field, key: 'rhs' });
  }

  add(field: Series<T> | number | bigint): Series<T> {
    return this.wrapDtype('add', { field, key: 'other' });
  }
  sub(field: Series<T> | number | bigint): Series<T>  {
    return this.wrapDtype('sub', { field, key: 'other' });
  }
  sum(field: Series<T> | number | bigint): Series<T> {
    return this.wrapDtype('sum', { field, key: 'other' });
  }
  div(field: Series<T> | number | bigint): Series<T> {
    return this.wrapDtype('div', { field, key: 'other' });
  }
  mul(field: Series<T> | number | bigint): Series<T> {
    return this.wrapDtype('mul', { field, key: 'other' });
  }

  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @see {@link rename}
   *
   */
  alias(name: string): Series<T> {
    const s = this.clone();

    this.unwrap('rename', { name }, s._series);

    return s;
  }

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
  append(other: Series<T>): void {
    this.wrap('append', { other: other._series });
  }

  // extend(other: Series<T>): Series<T> {
  //   const s = this.clone();

  //   this.unwrap('append', { other: other._series }, s._series);

  //   return s;

  // }

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
  apply<U>(func: (s: T) => U): Series<U> {
    throw new Error('Unimplemented');
  }
  
  /**
   * Get the index of the maximal value.
   */
  argMax(): Optional<number> {
    return this.unwrap('arg_max');
  }
  /**
   * Get the index of the minimal value.
   */
  argMin(): Optional<number> {
    return this.unwrap('arg_min');
  }
  /**
   * Get index values where Boolean Series evaluate True.
   *
   */
  argTrue(): Series<number> {
    return this.wrap<number>('arg_true');
  }
  /**
   * Get unique index as Series.
   */
  argUnique(): Series<number> {
    return this.wrap('arg_unique');
  }
  /**
   * Index location of the sorted variant of this Series.
   * ___
   * @param reverse
   * @return {SeriesType} indexes - Indexes that can be used to sort this array.
   */
  argSort<U>(reverse: boolean): Series<U> {
    return this.wrap<U>('argsort', { reverse });
  }

  bitand(other: Series<any>): Series<T> {
    return this.wrap('bitand', { other: other._series });
  }

  bitor(other: Series<any>): Series<T> {
    return this.wrap('bitor', { other: other._series });
  }

  bitxor(other: Series<any>): Series<T> {
    return this.wrap('bitxor', { other: other._series });
  }
  /**
   * Cast between data types.
   */
  cast<D extends DataType>(dtype: D, strict = true): Series<DtypeToPrimitive<D>> {
    return this.wrap('cast', { dtype, strict });
  }
  /**
   * Get the length of each individual chunk
   */
  chunkLengths(): Array<number> {
    return this.unwrap('chunk_lengths');
  }
  /**
   * Cheap deep clones.
   */
  clone(): Series<T> {
    return this.wrap('clone');
  }
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
  cumMax(reverse = false): Series<T> {
    return this.wrap('cummax', { reverse });
  }
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
  cumMin(reverse = false): Series<T> {
    return this.wrap('cummin', { reverse });
  }
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
  cumProd(reverse = false): Series<T> {
    return this.wrap('cumprod', { reverse });
  }
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
  cumSum(reverse = false): Series<T> {
    return this.wrap('cumsum', { reverse });
  }
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
  describe(): 'DataFrame' {
    throw new Error('unimplemented');
  }
  /**
   * Calculates the n-th discrete difference.
   * @param n - number of slots to shift
   * @param nullBehavior - `'ignore' | 'drop'`
   */
  diff(n: number, nullBehavior: 'ignore' | 'drop'): Series<T> {
    return this.wrap<T>('diff', { n, null_behavior: nullBehavior });
  }
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
  dot(other: Series<any>): Optional<number> {
    return this.unwrap('dot', { other: other._series });
  }
  /**
   * Create a new Series that copies data from this Series without null values.
   */
  dropNulls(): Series<T> {
    return this.wrap<T>('drop_nulls');
  }
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
  explode() {
    return this.wrap<T extends ArrayLike<infer Item> ? Item : T>('explode');
  }
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
  fillNull(
    strategy: 'backward' | 'forward' | 'min' | 'max' | 'mean' | 'one' | 'zero',
  ): Series<T> {
    return this.wrap('fill_null', { strategy });
  }
  /**
   * __Filter elements by a boolean mask.__
   * @param {SeriesType} predicate - Boolean mask
   *
   */
  filter(predicate: Series<boolean>): Series<T> {
    return this.wrap('filter', { predicate: predicate._series });
  }
  get(index: number): T {
    return this.unwrapDtype('get', { field: index, key: 'index' });
  }
  set(n: number): T {
    throw new Error('unimplemented');
    // return this.unwrapDtype('set_at_idx', { field: n, key: 'n' });
  }
  getIndex(n: number): T {
    return this.unwrap('get_idx', { n });
  }
  /**
   * Returns True if the Series has a validity bitmask.
   * If there is none, it means that there are no null values.
   */ 
  hasValidity(): boolean {
    return this.unwrap('has_validity');
  }
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
   *         7499844439152382372
   *         821952831504499201
   *         6685218033491627602
   * ]
   * ```
   */
  hash(k0 = 0, k1 = 1, k2 = 2, k3 = 3): Series<bigint> {
    return this.wrap('hash', { k0, k1, k2, k3 });
  }
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
  head(length: number): Series<T> {
    return this.wrap('head', { length });
  }
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
  interpolate(): Series<T> {
    return this.wrap('interpolate');
  }
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
  isDuplicated(): Series<boolean> {
    return this.wrap('is_duplicated');
  }
  /**
   * Get mask of finite values if Series dtype is Float.
   */
  isFinite(): T extends number ? Series<boolean> : never {
    const dtype = this.dtype;

    if (![DataType.Float32, DataType.Float64].includes(dtype)) {
      throw new Error(
        `Invalid operation isFinite is not suppored for ${dtype}`,
      );
    } else {
      return this.wrap('is_finite') as any;
    }
  }
  /**
   * Get a mask of the first unique value.
   */
  isFirst(): Series<boolean> {
    return this.wrap('is_first');
  }
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
  isInfinite(): T extends number ? Series<boolean> : never {
    const dtype = this.dtype;

    if (![DataType.Float32, DataType.Float64].includes(dtype)) {
      throw new Error(
        `Invalid operation isInfinite is not suppored for ${dtype}`,
      );
    } else {
      return this.wrap('is_infinite') as any;
    }
  }
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
  isNotNull(): Series<boolean> {
    return this.wrap('is_not_null');
  }
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
  isNull(): Series<boolean> {
    return this.wrap('is_null');
  }
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
  isUnique(): Series<boolean> {
    return this.wrap('is_unique');
  }
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
  kurtosis(fisher = true, bias = true): Optional<number> {
    return this.unwrap('kurtosis', { fisher, bias });
  }
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
  len(): number {
    return this.unwrap('len');
  }
  /**
   * __Take `n` elements from this Series.__
   * ___
   * @param n - Amount of elements to take.
   * @see {@link head}, {@link take}
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.limit(2)
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   * ]
   * ```
   */
  limit(n = 10): Series<T> {
    return this.wrap('limit', { num_elements: n });
  }
  /**
   * Compute the max value of the arrays in the list
   */
  max(): Series<T> {
    return this.wrap('max');
  }
  /**
   * Compute the mean value of the arrays in the list
   */
  mean(): Series<T> {
    return this.wrap('mean');
  }
  /**
   * Compute the median value of the arrays in the list
   */
  median(): Series<T> {
    return this.wrap('median');
  }
  /**
   * Compute the min value of the arrays in the list
   */
  min(): Series<T> {
    return this.wrap('min');
  }
  /**
   * __Compute the most occurring value(s). Can return multiple Values__
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 2, 3])
   * >>> s.mode()
   * shape: (1,)
   * Series: 'a' [i64]
   * [
   *         2
   * ]
   *
   * >>> s = pl.Series("a", ['a', 'b', 'c', 'c', 'b'])
   * >>> s.mode()
   * shape: (1,)
   * Series: 'a' [str]
   * [
   *         'b'
   *         'c'
   * ]
   * ```
   */
  mode(): Series<T> {
    return this.wrap('mode');
  }
  /**
   * Get the number of chunks that this Series contains.
   */
  nChunks(): number {
    return this.unwrap('n_chunks');
  }
  /**
   * __Count the number of unique values in this Series.__
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 2, 3])
   * >>> s.nUnique()
   * 3
   * ```
   */
  nUnique(): number {
    return this.unwrap('n_unique');
  }
  /**
   * Count the null values in this Series. --
   * _`undefined` values are treated as null_
   *
   */
  nullCount(): number {
    return this.unwrap('null_count');
  }
  /**
   * Get a boolean mask of the local maximum peaks.
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3, 4, 5])
   * >>> s.peakMax()
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
  peakMax(): Series<boolean> {
    return this.wrap('peak_max');
  }
  /**
   * Get a boolean mask of the local minimum peaks.
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [4, 1, 3, 2, 5])
   * >>> s.peakMin()
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
  peakMin(): Series<boolean> {
    return this.wrap('peak_min');
  }
  /**
   * Get the quantile value of this Series.
   * ___
   * @param quantile
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.quantile(0.5)
   * 2
   * ```
   */
  quantile(): number {
    return this.unwrap('quantile');
  }
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
  rank(
    method?: 'average' | 'min' | 'max' | 'dense' | 'ordinal' | 'random',
  ): Series<number> {
    return this.wrap('rank', { method: method ?? 'average' });
  }
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
  reinterpret(signed = true): T extends number ? Series<boolean> : never {
    const dtype = this.dtype;

    if (![DataType.UInt64, DataType.Int64].includes(dtype)) {
      throw new Error(
        `Invalid operation reinterpret is not suppored for ${dtype}`,
      );
    } else {
      return this.wrap('reinterpret', { signed }) as any;
    }
  }
  /**
   * __Rename this Series.__
   *
   * @param name - new name
   * @param inPlace - Modify the Series in-place.
   * @see {@link alias}
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.rename('b')
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
  rename(name: string, inPlace = false): void | Series<T> {
    if (inPlace) {
      this.unwrap('rename', { name });
    } else {
      return this.alias(name);
    }
  }
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
   * >>> s = pl.Series("a", [100, 200, 300, 400, 500])
   * >>> s.rollingMax(2)
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
  rollingMax(
    windowSize: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center = false,
  ): Series<T> {
    return this.wrap('rolling_max', {
      window_size: windowSize,
      weights,
      min_periods: minPeriods ?? windowSize,
      center,
    });
  }
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
   * >>> s = pl.Series("a", [100, 200, 300, 400, 500])
   * >>> s.rollingMean(2)
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
  rollingMean(
    windowSize: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center = false,
  ): Series<T> {
    return this.wrap('rolling_mean', {
      window_size: windowSize,
      weights,
      min_periods: minPeriods ?? windowSize,
      center,
    });
  }
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
   * >>> s = pl.Series("a", [100, 200, 300, 400, 500])
   * >>> s.rollingMin(2)
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
  rollingMin(
    windowSize: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center = false,
  ): Series<T> {
    return this.wrap('rolling_min', {
      window_size: windowSize,
      weights,
      min_periods: minPeriods ?? windowSize,
      center,
    });
  }
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
   * >>> s = pl.Series("a", [1, 2, 3, 4, 5])
   * >>> s.rollingSum(2)
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
  rollingSum(
    windowSize: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center = false,
  ): Series<T> {
    return this.wrap('rolling_sum', {
      window_size: windowSize,
      weights,
      min_periods: minPeriods ?? windowSize,
      center,
    });
  }
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
  rollingVar(
    windowSize: number,
    weights?: Array<number>,
    minPeriods?: Array<number>,
    center = false,
  ): Series<T> {
    return this.wrap('rolling_var', {
      window_size: windowSize,
      weights,
      min_periods: minPeriods ?? windowSize,
      center,
    });
  }
  sample(n?: number, frac?: number, withReplacement = false): Series<T> {
    if (n) {
      return this.wrap('sample_n', { n, with_replacement: withReplacement });
    }

    return this.wrap('sample_frac', {
      frac,
      with_replacement: withReplacement,
    });
  }
  /**
   * __Check if series is equal with another Series.__
   * @param other - Series to compare with.
   * @param nullEqual - Consider null values as equal. _('undefined' is treated as null)_
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s2 = pl.Series("b", [4, 5, 6])
   * >>> s.series_equal(s)
   * true
   * >>> s.series_equal(s2)
   * false
   * ```
   */
  seriesEqual<U>(other: Series<U>, nullEqual = false): boolean {
    return this.unwrap('series_equal', {
      other: other._series,
      null_equal: nullEqual,
    });
  }
  /**
   * __Shift the values by a given period__
   *
   * the parts that will be empty due to this operation will be filled with `null`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.shift(1)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         null
   *         1
   *         2
   * ]
   * >>> s.shift(-1)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         2
   *         3
   *         null
   * ]
   * ```
   */
  shift(periods = 1): Series<T> {
    return this.wrap('shift', { periods });
  }
  /**
   * Shift the values by a given period
   *
   * the parts that will be empty due to this operation will be filled with `fillValue`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @param fillValue - Fill null & undefined values with the result of this expression.
   */
  shiftAndFill(periods: number, fillValue: any): Series<T> {
    throw new Error('unimplemented');
  }
  /**
   * __Shrink memory usage of this Series to fit the exact capacity needed to hold the data.__
   * @param inPlace - Modify the Series in-place.
   */
  shrinkToFit(inPlace = false): Series<T> | void {
    if (inPlace) {
      this.unwrap('shrink_to_fit');
    } else {
      const s = this.clone();
      this.unwrap('shrink_to_fit', {}, s);

      return s;
    }
  }
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
  skew(bias = true): number | undefined {
    return this.unwrap('skew', { bias });
  }
  /**
   * Create subslices of the string values of a Utf8 Series.
   *
   * @param start - Start of the slice (negative indexing may be used).
   * @param length - Optional length of the slice.
   */
  slice(start: number, length?: number): T extends string ? Series<T> : never {
    return this.wrap('slice', { start, length }) as any;
  }
  sortInPlace(): void {
    throw new Error('unimplemented');
  }
  /**
   * __Sort this Series.__
   * @param options
   * @param options.reverse - Reverse sort
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 3, 4, 2])
   * >>> s.sort()
   * shape: (4,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   * ]
   * >>> s.sort(true)
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
  sort(reverse = false): Series<T> {
    return this.wrap('sort', { reverse });
  }
  /**
   * __Get last N elements as Series.__
   *
   * ___
   * @param length - Length of the tail
   * @see {@link head}
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.tail(2)
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         2
   *         3
   * ]
   * ```
   */
  tail(length?: number): Series<T> {
    return this.wrap('tail', { length });
  }
  /**
   * Take every nth value in the Series and return as new Series.
   * @param n - nth value to take
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3, 4])
   * >>> s.takeEvery(2))
   * shape: (2,)
   * Series: '' [i64]
   * [
   *         1
   *         3
   * ]
   * ```
   */
  takeEvery(n: number): Series<T> {
    return this.wrap('take_every', { n });
  }
  /**
   * Take values by index.
   * ___
   * @param indices - Index location used for the selection
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 3, 4])
   * >>> s.take([1, 3])
   * shape: (2,)
   * Series: 'a' [i64]
   * [
   *         2
   *         4
   * ]
   * ```
   */
  take(indices: Array<number>): Series<T> {
    throw new Error('unimplemented');
  }
  /**
   * __Convert this Series to a Javascript Array.__
   *
   * This operation clones data.
   * ___
   * @example
   * ```
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> const arr = s.toArray()
   * [1, 2, 3]
   * >>> Array.isArray(arr)
   * true
   * ```
   */
  toArray(): Array<T> {
    return [...this];
  }
  /**
   * __Get unique elements in series.__
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 2, 3])
   * >>> s.unique()
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   * ```
   */
  unique(): Series<T> {
    return this.wrap('unique');
  }
  /**
   * __Count the unique values in a Series.__
   * ___
   * @example
   * ```
   * >>> s = pl.Series("a", [1, 2, 2, 3])
   * >>> s.valueCounts()
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
  valueCounts(): 'DataFrame' {
    throw new Error('unimplemented');
  }
  
  /**
   * Where mask evaluates true, take values from self.
   *
   * Where mask evaluates false, take values from other.
   * ___
   * @param mask - Boolean Series
   * @param other - Series of same type
   *
   */
  zipWith<U>(mask: Series<boolean>, other: Series<T>): Series<U> {
    return this.wrap('zip_with', { mask: mask._series, other: other._series });
  }
  
  /**
   * _Returns a Javascript representation of Series
   * 
   * @example
   * ```
   * >>> const s = pl.Series("foo", [1,2,3])
   * >>> s.toJS()
   * {
   *   name: "foo",
   *   datatype: "Float64",
   *   values: [1,2,3]
   * }
   * ```
   */
  toJS(): object {
    return this.unwrap('to_js');
  }


  private unwrap<T>(method: string, args?: object, _series = this._series): T {
    // console.log({method, args, _series});

    return this.internal[method]({ _series, ...args });
  }

  private unwrapDtype<T>(
    method: string,
    args: { field: any; key: string },
    _series = this._series,
  ): T {
    if (args.field instanceof Series) {
      return this.unwrap(method, { [args.key]: args.field._series }, _series);
    } else {
      const dt = (DTYPE_TO_FFINAME as any)[DataType[this.dtype]];
      const internalMethod = `${method}_${dt}`;

      return this.unwrap<T>(
        internalMethod,
        { [args.key]: args.field },
        _series,
      );
    }
  }

  private wrapDtype<U>(
    method: string,
    args: { field: any; key: string },
    _series = this._series,
  ): Series<U> {

    if (args.field instanceof Series) {
      return this.wrap(method, { [args.key]: args.field._series }, _series);
    } else {
      const dt = (DTYPE_TO_FFINAME as any)[DataType[this.dtype]];
      const internalMethod = `${method}_${dt}`;

      return this.wrap<U>(internalMethod, { [args.key]: args.field }, _series);
    }
  }

  /**
     * Wraps the internal `_series` into the `Series` class
     */
  private wrap<U>(
    method: string,
    args?: object,
    _series = this._series,
  ): Series<U> {
    // console.log({method,args, dtype: this.dtype});

    return new Series(this.internal[method]({ _series, ...args }));
  }
}

export default Series.of;