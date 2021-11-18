import polars_internal from "./polars_internal";
import {Dtype, DataType, DTYPE_TO_FFINAME} from './datatypes';
import {getInternalFunc} from "./internals/utils";
import {arrayToJsSeries} from "./internals/construction";
import util from "util"


/**
 * 'any' type representing the internal rust `JsSeries`
 */
export type JsSeries = any;
export type mapping<T, U> = (x: T, ...args: any[]) => U;

interface SeriesList {

  /**
   * Get the length of the arrays as UInt32.
   */
  lengths: () => Series<DataType[Dtype.UInt32]>;
}

interface DateTimeNameSpace {

  /**
   * Extract the day from the underlying date representation.
   * 
   * Can be performed on Date and Datetime.
   * 
   * The return value ranges from 1 to 31. (The last day of month differs by months.)
   *  @returns {Series} `Series<number>` -  the day of month starting from 1.
   */
  day: () => Series<DataType[Dtype.UInt32]>;
}
/**
 * A Series represents a single column in a polars DataFrame.
 * @param {string} name - Name of the series. Will be used as a column name when used in a DataFrame.
 * @param {PolarsArrayLike} values - One-dimensional data in various forms. Supported are: Array, Series,
 * Set
 * @param {Dtype} [dtype] - Polars dtype of the Series data. If not specified, the dtype is inferred.
 * @param {boolean} [strict] - Throw error on numeric overflow
 * 
 * @example
 * >>> const s = pl.Series('a', [1,2,3]);
 * >>> console.log(s);
 * shape: (3,)
 * Series: 'a' [i64]
 * [
 *         1
 *         2
 *         3
 * ]
 * // Notice that the dtype is automatically inferred as a polars Int64:
 * >>> s.dtype()
 * "Int64"
 * 
 * // Constructing a Series with a specific dtype:
 * >>> const s2 = pl.Series('a', [1, 2, 3], dtype=pl.Float32);
 * >>> console.log(s2);
 * shape: (3,)
 * Series: 'a' [f32]
 * [
 *         1
 *         2
 *         3
 * ]
 */
interface _Series<T extends any> extends ArrayLike<T> {


  add: (other: Series<T>) => Series<T>;

  /**
   * Append a Series to this one.
   *
   * Parameters
   * ----------
   * @param {Series} other - Series to append.
   * 
   * @example
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s2 = pl.Series("b", [4, 5, 6])
   * >>> s.append(s2)
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
  append: (other: Series<T>) => void;

  /**
   * Apply a function over elements in this Series and return a new Series.
   * 
   * If the function returns another datatype, the return_dtype arg should be set, otherwise the method will fail.
   * 
   * Parameters
   * ----------
   * @param {CallableFunction} func - function or lambda.
   * @param {Dtype} returnDtype - Output datatype. If none is given, the same datatype as this Series will be used.
   * 
   * Returns
   * -------
   * Series
   * 
   * @example
   * >>> s = pl.Series("a", [1, 2, 3])
   * >>> s.apply(lambda x: x + 10)
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         11
   *         12
   *         13
   * ]
   */
  // apply: <U extends any>(func: (s: Series<T>) => Series<U>) => ReturnType<typeof func>;

  /**
   * Get the index of the maximal value.
   */
  argMax: () => number | undefined;
  /**
   * Get the index of the minimal value.
   */
  argMin: () => number | undefined;
  /**
   * Get index values where Boolean Series evaluate True.
   * 
   */
  argTrue: () => Series<T>;
  /**
   * Get unique index as Series.
   */
  argUnique: () => Series<T>;
  /**
   * Index location of the sorted variant of this Series.
   * Returns
   * -------
   * @return indexes - Indexes that can be used to sort this array.
   */
  argSort: <U>(reverse: boolean) => Series<U>;


  bitand: (other: Series<any>) => Series<T>;
  bitor: (other: Series<any>) => Series<T>;
  bitxor: (other: Series<any>) => Series<T>;

  /**
   * Cast between data types.
   */

  // cast: (dtype: Dtype, strict?: boolean) => Series<plDtype[typeof dtype]>;
  /**
   * Get the length of each individual chunk
   */
  chunkLengths: () => Array<number>;
  /**
   * Cheap deep clones.
   */
  clone: () => Series<T>;

  /**
   * Get an array with the cumulative max computes at every element.
   * 
   * @param {boolean} reverse - reverse the operation
   * 
   * @example
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> s.cumMax()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   */
  cumMax: (reverse?: boolean) => Series<T>;

  /**
   * Get an array with the cumulative min computed at every element.
   * @param {boolean} reverse - reverse the operation
   *         
   * @example
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> s.cumMin()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         1
   *         1
   * ]
   */
  cumMin: (reverse?: boolean) => Series<T>;
  /**
   * Get an array with the cumulative product computed at every element.
   * @param {boolean} reverse - reverse the operation
   * @example
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> s.cumProd()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         2
   *         6
   * ]
   */
  cumProd: (reverse?: boolean) => Series<T>;
  /**
   * Get an array with the cumulative sum computed at every element.
   * @param {boolean} reverse - reverse the operation
   *         
   * @example
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> s.cumSum()
   * shape: (3,)
   * Series: 'b' [i64]
   * [
   *         1
   *         3
   *         6
   * ]
   */
  cumSum: () => void;

  /**
   * Calculates the n-th discrete difference.
   * @param {number} n - number of slots to shift
   * @param {string} nullBehavior - `'ignore' | 'drop'`
   */
  diff: (n: number, nullBehavior: 'ignore' | 'drop') => Series<T>;

  div: (other: Series<T>) => Series<T>; // math

  /**
   * Compute the dot/inner product between two Series
   * 
   * @example
   * >>> const s = pl.Series("a", [1, 2, 3])
   * >>> const s2 = pl.Series("b", [4.0, 5.0, 6.0])
   * >>> s.dot(s2)
   * 32.0
   */
  dot: (other: Series<any>) => number | undefined;

  /**
   * Create a new Series that copies data from this Series without null values.
   */
  drop_nulls: () => Series<T>;

  /**
  * Get the data type of this Series.
  * @example
  * >>> const s = pl.Series("a", [1,2,3]);
  * >>> console.log(s.dtype);
  * 'Int64'
  */
  get dtype(): Dtype;


  eq: () => void; // math

  /**
   * Explode a list or utf8 Series. This means that every item is expanded to a new row.
   * 
   * @returns {Series}
   * @example
   * >>> s = pl.Series('a', [[1, 2], [3, 4], [9, 10]])
   * >>> s.explode()
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
   */
  explode: () => Series<T extends ArrayLike<infer Item> ? Item : T>;
  fill_null: () => void;
  filter: () => void;
  get_fmt: () => void;
  get_idx: () => void;
  get_list: () => void;
  gt_eq: () => void;
  gt: () => void;
  has_validity: () => void;
  hash: () => void;
  head: () => void;
  hour: () => void;
  interpolate: () => void;
  is_duplicated: () => void;
  is_finite: () => void;
  is_first: () => void;
  is_infinite: () => void;
  is_nan: () => void;
  is_not_nan: () => void;
  is_not_null: () => void;
  is_null: () => void;
  is_unique: () => void;
  kurtosis: () => void;
  len: () => void;
  limit: () => void;
  lt_eq: () => void;
  lt: () => void;
  max: () => void;
  mean: () => void;
  median: () => void;
  min: () => void;
  minute: () => void;
  mode: () => void;
  month: () => void;
  mul: () => void;
  n_chunks: () => void;
  n_unique: () => void;
  name: () => void;
  nanosecond: () => void;
  neq: () => void;
  null_count: () => void;
  ordinal_day: () => void;
  peak_max: () => void;
  peak_min: () => void;
  quantile: () => void;
  rank: () => void;
  reinterpret: () => void;
  rem: () => void;
  rename: () => void;
  repeat: () => void;
  rolling_max: () => void;
  rolling_mean: () => void;
  rolling_min: () => void;
  rolling_sum: () => void;
  rolling_var: () => void;
  round: () => void;
  sample_frac: () => void;
  sample_n: () => void;
  second: () => void;
  series_equal: () => void;
  shift: () => void;
  shrink_to_fit: () => void;
  skew: () => void;
  slice: () => void;
  sort_in_place: () => void;
  sort: () => void;
  strftime: () => void;
  sub: () => void;
  sum: () => void;
  tail: () => void;
  take_every: () => void;
  take_with_series: () => void;
  take: () => void;
  timestamp: () => void;
  to_array: () => void;
  to_dummies: () => void;

  /**
   * String representation of Series
   */
  toString: () => string;

  unique: () => void;
  value_counts: () => void;
  week: () => void;
  weekday: () => void;
  year: () => void;
  zip_with: () => void;
  /**
  * Get the data type of this Series.
  * @example
  * >>> const s = pl.Series("a", [1,2,3]);
  * >>> console.log(s.dtype);
  * 'Int64'
  */
  readonly _dtype: Dtype;
  /**
   * Quick summary statistics of a series. 
   * Series with mixed datatypes will return summary statistics for the datatype of the first value.
   * @returns Dataframe with summary statistics of a Series.
   * 
   * @example
   * >>> const seriesNum = pl.Series([1,2,3,4,5])
   * >>> series_num.describe()
   * 
   * shape: (6, 2)
    ┌──────────────┬────────────────────┐
    │ statistic    ┆ value              │
    │ ---          ┆ ---                │
    │ str          ┆ f64                │
    ╞══════════════╪════════════════════╡
    │ "min"        ┆ 1                  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ "max"        ┆ 5                  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ "null_count" ┆ 0.0                │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ "mean"       ┆ 3                  │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ "std"        ┆ 1.5811388300841898 │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
    │ "count"      ┆ 5                  │
    └──────────────┴────────────────────┘
   *
   * >>> series_str = pl.Series(["a", "a", None, "b", "c"])
   * >>> series_str.describe()
   * 
   * shape: (3, 2)
    ┌──────────────┬───────┐
    │ statistic    ┆ value │
    │ ---          ┆ ---   │
    │ str          ┆ i64   │
    ╞══════════════╪═══════╡
    │ "unique"     ┆ 4     │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
    │ "null_count" ┆ 1     │
    ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
    │ "count"      ┆ 5     │
    └──────────────┴───────┘

   */
  describe: () => Record<string, number>;

}



class Series<T extends any> implements _Series<T>, ArrayLike<T>  {
  [n: number]: T;
  private _series: JsSeries;
  private internal = polars_internal.series;

  /**
   * Unwraps the internal `_series` into another type based on the internal method
   */
  private unwrap<T>(method: string, args?: object, _series = this._series): T {
    return this.internal[method]({_series, ...args})
  }

  /**
   * Wraps the internal `_series` into the `Series` class
   */
  private wrap(method: string, args?: object, _series = this._series): Series<T> {
    console.log({method, args, _series})
    return new Series(this.internal[method]({_series, ...args}))
  }

  // Constructors
  /**
   * Might consider removing this, or setting up a configuration flag to enable/disable
   * 
   * the iteration is needed for Javascript bracket notation on the series
   * 
   * @example
   * >>> const s = pl.Series('a', ['foo', 'bar' 'baz'])
   * >>> s[1]
   * 'bar'
   */
  constructor(series: JsSeries) {
    this._series = series
    const len = this.internal.len({_series: series});
    for (let index = 0; index < len; index++) {
      Object.defineProperty(this, `${index}`, {
        get() {
          return this.internal.get_idx({_series: series, idx: index});
        }
      })
    }
  }
  static of<V extends ArrayLike<any>>(
    name: string,
    values: V
  ): V extends ArrayLike<infer U> ? Series<U> : never;
  static of<T extends Dtype, U extends ArrayLike<DataType[T]>>(
    name: string,
    values: U,
    dtype: T,
  ): Series<DataType[T]>;
  static of<T extends Dtype, U extends ArrayLike<DataType[T]>>(
    name: string,
    values: U,
    dtype: T,
    strict?: boolean
  ): Series<DataType[T]>;
  static of<T extends Dtype, U extends ArrayLike<DataType[T]>>(
    name: string,
    values: U,
    dtype?: T,
    strict: boolean = true
  ): Series<DataType[T]> {
    const series = arrayToJsSeries(name, values as any, dtype, strict);
    return new Series(series) as any;
  }

  // javascript overrides for iteration, and console.log
  [util.inspect.custom]() {
    return this.unwrap('get_fmt')
  }
  [Symbol.isConcatSpreadable] = true;
  *[Symbol.iterator]() {
    let len = this.unwrap<number>("len");
    let s = this._series;
    while (len >= 0) {
      let v = this.wrap('head', {length: 1}, s);
      s = this.unwrap('slice', {offset: 1, length: len--}, s)
      yield v
    }
  }
  public get dtype() {
    return this.unwrap<Dtype>('dtype')
  }

  public get length() {
    return this.unwrap<number>("len");
  }

  add(other: Series<T>) {
    return this.wrap("add", {other: other._series})
  }

  alias(name: string): Series<T> {
    return this.wrap("rename", {name})
  }

  append(other: Series<T>) {
    return this.wrap("append", {other: other._series})
  }

  apply<U extends any>(func: (s: T) => U): Series<U> {
    throw new Error("Unimplemented")
  }

  argMax() {
    return this.unwrap<number | undefined>("arg_max")
  }

  argMin() {
    return this.unwrap<number | undefined>("arg_min")
  }

  argTrue() {
    return this.wrap("arg_true")
  }

  argUnique() {
    return this.wrap("arg_unique")
  }

  argSort<U>(reverse: boolean): Series<U> {
    return this.wrap('argsort', {reverse}) as Series<U>;
  }

  bitand(other: Series<any>): Series<T> {
    return this.wrap('bitand', {other: other._series})
  }

  bitor(other: Series<any>): Series<T> {
    return this.wrap('bitor', {other: other._series})
  }

  bitxor(other: Series<any>): Series<T> {
    return this.wrap('bitxor', {other: other._series})
  }
  chunkLengths(): Array<number> {
    return this.unwrap('chunk_lengths')
  }
  clone(): Series<T> {
    return this.wrap('clone');
  }

  cumMax(reverse = false): Series<T> {
    return this.wrap('cum_max', {reverse});
  }

  cumMin(reverse = false): Series<T> {
    return this.wrap('cum_min', {reverse});
  }

  cumProd(reverse = false): Series<T> {
    return this.wrap('cum_prod', {reverse});
  }
  cumSum(reverse = false): Series<T> {
    return this.wrap('cum_sum', {reverse});
  }
  diff(n: number, nullBehavior: 'ignore' | 'drop'): Series<T> {
    throw new Error("Unimplemented")
  }
  // rename(name: string): Series<T>;
  // rename(name: string, inPlace: true): void;
  // rename(name: string, inPlace: false): Series<T>;
  // rename(name: string, inPlace?: boolean): void | Series<T> {
  //   if (inPlace) {
  //     this._series = this.wrap('rename', {name})
  //   } else {
  //     return this.alias(name);
  //   }
  // };

  explode(): Series<T extends ArrayLike<infer Item> ? Item : T> {
    return this.wrap('explode') as any;
  }

}




const s = Series.of("1", [1, 22, 333, 111, 44, 5, 333, 2, 2])
const s2 = s.argMax();
console.log(s2)
// const cx = s.explode<number[]>()
// console.log(s[1])
// console.log(s.dtype)

