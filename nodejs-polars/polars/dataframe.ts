import pli from "./internals/polars_internal";
import { arrayToJsDataFrame } from "./internals/construction";
import {GroupBy} from "./groupby";
import {LazyDataFrame} from "./lazy/dataframe";
import {concat} from "./functions";
import {Expr} from "./lazy/expr";
import {Series, seriesWrapper} from "./series/series";
import {Stream, Writable} from "stream";
import {isExternal} from "util/types";

import {
  DataType,
  JoinBaseOptions,
  JsDataFrame
} from "./datatypes";

import {
  columnOrColumns,
  columnOrColumnsStrict,
  ColumnSelection,
  FillNullStrategy,
  isSeriesArray,
  ColumnsOrExpr,
  ValueOrArray,
  ExprOrString
} from "./utils";

import {Arithmetic} from "./shared_traits";
import {col} from "./lazy/functions";

const inspect = Symbol.for("nodejs.util.inspect.custom");

type WriteCsvOptions = {
  hasHeader?: boolean;
  sep?: string;
};

type WriteJsonOptions = {
  orient?: "row" | "col" | "dataframe";
  multiline?: boolean;
};

type WriteParquetOptions = {
  compression?: "uncompressed" | "snappy" | "gzip" | "lzo" | "brotli" | "lz4" | "zstd"
};

type WriteIPCOptions = {
  compression?: "uncompressed" | "lz4" | "zstd"
};


/**
 *
  A DataFrame is a two-dimensional data structure that represents data as a table
  with rows and columns.

  Parameters
  ----------
  @param data -  Object, Array, or Series
      Two-dimensional data in various forms. object must contain Arrays.
      Array may contain Series or other Arrays.
  @param columns - Array of str, default undefined
      Column labels to use for resulting DataFrame. If specified, overrides any
      labels already present in the data. Must match data dimensions.
  @param orient - 'col' | 'row' default undefined
      Whether to interpret two-dimensional data as columns or as rows. If None,
      the orientation is inferred by matching the columns and data dimensions. If
      this does not yield conclusive results, column orientation is used.
  Examples
  --------
  Constructing a DataFrame from an object :
  ```
  data = {'a': [1n, 2n], 'b': [3, 4]}
  df = pl.DataFrame(data)
  df
  shape: (2, 2)
  ╭─────┬─────╮
  │ a   ┆ b   │
  │ --- ┆ --- │
  │ u64 ┆ i64 │
  ╞═════╪═════╡
  │ 1   ┆ 3   │
  ├╌╌╌╌╌┼╌╌╌╌╌┤
  │ 2   ┆ 4   │
  ╰─────┴─────╯
  ```
  Notice that the dtype is automatically inferred as a polars Int64:
  ```
  df.dtypes
  ['UInt64', `Int64']
  ```
  In order to specify dtypes for your columns, initialize the DataFrame with a list
  of Series instead:
  ```
  data = [pl.Series('col1', [1, 2], pl.Float32),
  ...         pl.Series('col2', [3, 4], pl.Int64)]
  df2 = pl.DataFrame(series)
  df2
  shape: (2, 2)
  ╭──────┬──────╮
  │ col1 ┆ col2 │
  │ ---  ┆ ---  │
  │ f32  ┆ i64  │
  ╞══════╪══════╡
  │ 1    ┆ 3    │
  ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
  │ 2    ┆ 4    │
  ╰──────┴──────╯
  ```

  Constructing a DataFrame from a list of lists, row orientation inferred:
  ```
  data = [[1, 2, 3], [4, 5, 6]]
  df4 = pl.DataFrame(data, ['a', 'b', 'c'])
  df4
  shape: (2, 3)
  ╭─────┬─────┬─────╮
  │ a   ┆ b   ┆ c   │
  │ --- ┆ --- ┆ --- │
  │ i64 ┆ i64 ┆ i64 │
  ╞═════╪═════╪═════╡
  │ 1   ┆ 2   ┆ 3   │
  ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
  │ 4   ┆ 5   ┆ 6   │
  ╰─────┴─────┴─────╯
  ```
 */
export interface DataFrame extends Arithmetic<DataFrame> {
  /** @ignore */
  _df: JsDataFrame
  dtypes: DataType[]
  height: number
  shape: {height: number, width: number}
  width: number
  get columns(): string[]
  set columns(cols: string[])
  [inspect](): string;
  [Symbol.iterator](): Generator<any, void, any>;
  /**
   * Very cheap deep clone.
   */
  clone(): DataFrame
  /**
   * __Summary statistics for a DataFrame.__
   *
   * Only summarizes numeric datatypes at the moment and returns nulls for non numeric datatypes.
   * ___
   * Example
   * ```
   * > df = pl.DataFrame({
   * >     'a': [1.0, 2.8, 3.0],
   * >     'b': [4, 5, 6],
   * >     "c": [True, False, True]
   * >     })
   * > df.describe()
   * shape: (5, 4)
   * ╭──────────┬───────┬─────┬──────╮
   * │ describe ┆ a     ┆ b   ┆ c    │
   * │ ---      ┆ ---   ┆ --- ┆ ---  │
   * │ str      ┆ f64   ┆ f64 ┆ f64  │
   * ╞══════════╪═══════╪═════╪══════╡
   * │ "mean"   ┆ 2.267 ┆ 5   ┆ null │
   * ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ "std"    ┆ 1.102 ┆ 1   ┆ null │
   * ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ "min"    ┆ 1     ┆ 4   ┆ 0.0  │
   * ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ "max"    ┆ 3     ┆ 6   ┆ 1    │
   * ├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ "median" ┆ 2.8   ┆ 5   ┆ null │
   * ╰──────────┴───────┴─────┴──────╯
   * ```
   */
  describe(): DataFrame

  /**
   * Drop duplicate rows from this DataFrame.
   * Note that this fails if there is a column of type `List` in the DataFrame.
   * @param maintainOrder
   * @param subset - subset to drop duplicates for
   * @param keep "first" | "last"
   */
  distinct(maintainOrder?: boolean, subset?: ColumnSelection, keep?: "first"| "last"): DataFrame
  distinct(opts: {maintainOrder?: boolean, subset?: ColumnSelection, keep?: "first"| "last"}): DataFrame
  /**
   * __Remove column from DataFrame and return as new.__
   * ___
   * @param name
   * @example
   * ```
   * > df = pl.DataFrame({
   * >   "foo": [1, 2, 3],
   * >   "bar": [6.0, 7.0, 8.0],
   * >   "ham": ['a', 'b', 'c'],
   * >   "apple": ['a', 'b', 'c']
   * > })
   * > df.drop(['ham', 'apple'])
   * shape: (3, 2)
   * ╭─────┬─────╮
   * │ foo ┆ bar │
   * │ --- ┆ --- │
   * │ i64 ┆ f64 │
   * ╞═════╪═════╡
   * │ 1   ┆ 6   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   │
   * ╰─────┴─────╯
   *
   * ```
   *
   */
  drop(name: string): DataFrame
  drop(names: string[]): DataFrame
  drop(name: string, ...names: string[]): DataFrame
  /**
   * __Drop duplicate rows from this DataFrame.__
   *
   * Note that this fails if there is a column of type `List` in the DataFrame.
   * @param maintainOrder
   * @param subset - subset to drop duplicates for
   * @deprecated @since 0.2.1 @use {@link distinct}
   */
  dropDuplicates(maintainOrder?: boolean, subset?: ColumnSelection): DataFrame
  /** @deprecated @since 0.2.1 @use {@link distinct} ==*/
  dropDuplicates(opts: {maintainOrder?: boolean, subset?: ColumnSelection}): DataFrame
  /**
   * __Return a new DataFrame where the null values are dropped.__
   *
   * This method only drops nulls row-wise if any single value of the row is null.
   * ___
   * @example
   * ```
   * > df = pl.DataFrame({
   * >   "foo": [1, 2, 3],
   * >   "bar": [6, null, 8],
   * >   "ham": ['a', 'b', 'c']
   * > })
   * > df.dropNulls()
   * shape: (2, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" │
   * └─────┴─────┴─────┘
   * ```
   */
  dropNulls(column: string): DataFrame
  dropNulls(columns: string[]): DataFrame
  dropNulls(...columns: string[]): DataFrame
  /**
   * __Explode `DataFrame` to long format by exploding a column with Lists.__
   * ___
   * @param columns - column or columns to explode
   * @example
   * ```
   * > df = pl.DataFrame({
   * >   "letters": ["c", "c", "a", "c", "a", "b"],
   * >   "nrs": [[1, 2], [1, 3], [4, 3], [5, 5, 5], [6], [2, 1, 2]]
   * > })
   * > console.log(df)
   * shape: (6, 2)
   * ╭─────────┬────────────╮
   * │ letters ┆ nrs        │
   * │ ---     ┆ ---        │
   * │ str     ┆ list [i64] │
   * ╞═════════╪════════════╡
   * │ "c"     ┆ [1, 2]     │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "c"     ┆ [1, 3]     │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "a"     ┆ [4, 3]     │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "c"     ┆ [5, 5, 5]  │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "a"     ┆ [6]        │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ "b"     ┆ [2, 1, 2]  │
   * ╰─────────┴────────────╯
   * > df.explode("nrs")
   * shape: (13, 2)
   * ╭─────────┬─────╮
   * │ letters ┆ nrs │
   * │ ---     ┆ --- │
   * │ str     ┆ i64 │
   * ╞═════════╪═════╡
   * │ "c"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 2   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 3   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ ...     ┆ ... │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 5   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 6   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 2   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 2   │
   * ╰─────────┴─────╯
   * ```
   */
  explode(column: ExprOrString): DataFrame
  explode(columns: ExprOrString[]): DataFrame
  explode(column: ExprOrString, ...columns: ExprOrString[]): DataFrame
  /**
   *
   *
   * __Extend the memory backed by this `DataFrame` with the values from `other`.__
   * ___

    Different from `vstack` which adds the chunks from `other` to the chunks of this `DataFrame`
    `extent` appends the data from `other` to the underlying memory locations and thus may cause a reallocation.

    If this does not cause a reallocation, the resulting data structure will not have any extra chunks
    and thus will yield faster queries.

    Prefer `extend` over `vstack` when you want to do a query after a single append. For instance during
    online operations where you add `n` rows and rerun a query.

    Prefer `vstack` over `extend` when you want to append many times before doing a query. For instance
    when you read in multiple files and when to store them in a single `DataFrame`.
    In the latter case, finish the sequence of `vstack` operations with a `rechunk`.

   * @param other DataFrame to vertically add.
   */
  extend(other: DataFrame): DataFrame
  /**
   * Fill null/missing values by a filling strategy
   *
   * @param strategy - One of:
   *   - "backward"
   *   - "forward"
   *   - "mean"
   *   - "min'
   *   - "max"
   *   - "zero"
   *   - "one"
   * @returns DataFrame with None replaced with the filling strategy.
   */
  fillNull(strategy: FillNullStrategy): DataFrame
  /**
   * Filter the rows in the DataFrame based on a predicate expression.
   * ___
   * @param predicate - Expression that evaluates to a boolean Series.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> // Filter on one condition
   * >>> df.filter(pl.col("foo").lt(3))
   * shape: (2, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ a   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ b   │
   * └─────┴─────┴─────┘
   * >>>  // Filter on multiple conditions
   * >>> df.filter(
   *  pl.col("foo").lt(3)
   *    .and(pl.col("ham").eq("a"))
   * )
   * shape: (1, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ a   │
   * └─────┴─────┴─────┘
   * ```
   */
  filter(predicate: any): DataFrame
  /**
   * Find the index of a column by name.
   * ___
   * @param name -Name of the column to find.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.findIdxByName("ham"))
   * 2
   * ```
   */
  findIdxByName(name: string): number
  /**
   * __Apply a horizontal reduction on a DataFrame.__
   *
   * This can be used to effectively determine aggregations on a row level,
   * and can be applied to any DataType that can be supercasted (casted to a similar parent type).
   *
   * An example of the supercast rules when applying an arithmetic operation on two DataTypes are for instance:
   *  - Int8 + Utf8 = Utf8
   *  - Float32 + Int64 = Float32
   *  - Float32 + Float64 = Float64
   * ___
   * @param operation - function that takes two `Series` and returns a `Series`.
   * @returns Series
   * @example
   * ```
   * >>> // A horizontal sum operation
   * >>> df = pl.DataFrame({
   * >>>   "a": [2, 1, 3],
   * >>>   "b": [1, 2, 3],
   * >>>   "c": [1.0, 2.0, 3.0]
   * >>> })
   * >>> df.fold((s1, s2) => s1.plus(s2))
   * Series: 'a' [f64]
   * [
   *     4
   *     5
   *     9
   * ]
   * >>> // A horizontal minimum operation
   * >>> df = pl.DataFrame({
   * >>>   "a": [2, 1, 3],
   * >>>   "b": [1, 2, 3],
   * >>>   "c": [1.0, 2.0, 3.0]
   * >>> })
   * >>> df.fold((s1, s2) => s1.zipWith(s1.lt(s2), s2))
   * Series: 'a' [f64]
   * [
   *     1
   *     1
   *     3
   * ]
   * >>> // A horizontal string concattenation
   * >>> df = pl.DataFrame({
   * >>>   "a": ["foo", "bar", 2],
   * >>>   "b": [1, 2, 3],
   * >>>   "c": [1.0, 2.0, 3.0]
   * >>> })
   * >>> df.fold((s1, s2) => s.plus(s2))
   * Series: '' [f64]
   * [
   *     "foo11"
   *     "bar22
   *     "233"
   * ]
   * ```
   */
  fold<T, U, V>(operation: (s1: Series<T>, s2: Series<U>) => Series<V>): Series<V>
  /**
   * Check if DataFrame is equal to other.
   * ___
   * @param options
   * @param options.other - DataFrame to compare.
   * @param options.nullEqual Consider null values as equal.
   * @example
   * ```
   * >>> df1 = pl.DataFrame({
   * >>    "foo": [1, 2, 3],
   * >>    "bar": [6.0, 7.0, 8.0],
   * >>    "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df2 = pl.DataFrame({
   * >>>   "foo": [3, 2, 1],
   * >>>   "bar": [8.0, 7.0, 6.0],
   * >>>   "ham": ['c', 'b', 'a']
   * >>> })
   * >>> df1.frameEqual(df1)
   * true
   * >>> df1.frameEqual(df2)
   * false
   * ```
   */
  frameEqual(other: DataFrame): boolean
  frameEqual(other: DataFrame, nullEqual: boolean): boolean
  /**
   * Get a single column as Series by name.
   */
  getColumn(name: string): Series<any>
  /**
   * Get the DataFrame as an Array of Series.
   */
  getColumns(): Array<Series<any>>
  /**
   * Start a groupby operation.
   * ___
   * @param by - Column(s) to group by.
   */
  groupBy(...by: ColumnSelection[]): GroupBy
  /**
   * Hash and combine the rows in this DataFrame. _(Hash value is UInt64)_
   * @param k0 - seed parameter
   * @param k1 - seed parameter
   * @param k2 - seed parameter
   * @param k3 - seed parameter
   */
  hashRows(k0?: number, k1?: number, k2?: number, k3?: number): Series<bigint>
  hashRows(options: {k0?: number, k1?: number, k2?: number, k3?: number}): Series<bigint>
  /**
   * Get first N rows as DataFrame.
   * ___
   * @param length -  Length of the head.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3, 4, 5],
   * >>>   "bar": [6, 7, 8, 9, 10],
   * >>>   "ham": ['a', 'b', 'c', 'd','e']
   * >>> })
   * >>> df.head(3)
   * shape: (3, 3)
   * ╭─────┬─────┬─────╮
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ "b" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" │
   * ╰─────┴─────┴─────╯
   * ```
   */
  head(length?: number): DataFrame
  /**
   * Return a new DataFrame grown horizontally by stacking multiple Series to it.
   * @param columns - array of Series or DataFrame to stack
   * @param inPlace - Modify in place
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> x = pl.Series("apple", [10, 20, 30])
   * >>> df.hStack([x])
   * shape: (3, 4)
   * ╭─────┬─────┬─────┬───────╮
   * │ foo ┆ bar ┆ ham ┆ apple │
   * │ --- ┆ --- ┆ --- ┆ ---   │
   * │ i64 ┆ i64 ┆ str ┆ i64   │
   * ╞═════╪═════╪═════╪═══════╡
   * │ 1   ┆ 6   ┆ "a" ┆ 10    │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ "b" ┆ 20    │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" ┆ 30    │
   * ╰─────┴─────┴─────┴───────╯
   * ```
   */
  hstack(columns: Array<Series<any>> | DataFrame): DataFrame
  hstack(columns: Array<Series<any>> | DataFrame, inPlace?: boolean): void
  /**
   * Insert a Series at a certain column index. This operation is in place.
   * @param index - Column position to insert the new `Series` column.
   * @param series - `Series` to insert
   */
  insertAtIdx(index: number, series: Series<any>): void
  /**
   * Interpolate intermediate values. The interpolation method is linear.
   */
  interpolate(): DataFrame
  /**
   * Get a mask of all duplicated rows in this DataFrame.
   */
  isDuplicated(): Series<boolean>
  /**
   * Check if the dataframe is empty
   */
  isEmpty(): boolean
  /**
   * Get a mask of all unique rows in this DataFrame.
   */
  isUnique(): Series<boolean>
  /**
   *  __SQL like joins.__
   * @param df - DataFrame to join with.
   * @param options
   * @param options.leftOn - Name(s) of the left join column(s).
   * @param options.rightOn - Name(s) of the right join column(s).
   * @param options.on - Name(s) of the join columns in both DataFrames.
   * @param options.how - Join strategy
   * @param options.suffix - Suffix to append to columns with a duplicate name.
   * @see {@link JoinOptions}
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6.0, 7.0, 8.0],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> otherDF = pl.DataFrame({
   * >>>   "apple": ['x', 'y', 'z'],
   * >>>   "ham": ['a', 'b', 'd']
   * >>> })
   * >>> df.join(otherDF, {on: 'ham'})
   * shape: (2, 4)
   * ╭─────┬─────┬─────┬───────╮
   * │ foo ┆ bar ┆ ham ┆ apple │
   * │ --- ┆ --- ┆ --- ┆ ---   │
   * │ i64 ┆ f64 ┆ str ┆ str   │
   * ╞═════╪═════╪═════╪═══════╡
   * │ 1   ┆ 6   ┆ "a" ┆ "x"   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ "b" ┆ "y"   │
   * ╰─────┴─────┴─────┴───────╯
   * ```
   */
  join(df: DataFrame, options: {on: ValueOrArray<string>} & JoinBaseOptions): DataFrame
  join(df: DataFrame, options: {leftOn: ValueOrArray<string>, rightOn: ValueOrArray<string>} & JoinBaseOptions): DataFrame
  lazy(): LazyDataFrame
  /**
   * Get first N rows as DataFrame.
   * @see {@link head}
   */
  limit(length?: number): DataFrame
  map<T>(func: (...args: any[]) => T): T[]

  /**
   * Aggregate the columns of this DataFrame to their maximum value.
   * ___
   * @param axis - either 0 or 1
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.max()
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ i64 ┆ i64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 3   ┆ 8   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  max(): DataFrame
  max(axis: 0): DataFrame
  max(axis: 1): Series<any>
  /**
   * Aggregate the columns of this DataFrame to their mean value.
   * ___
   *
   * @param axis - either 0 or 1
   * @param nullStrategy - this argument is only used if axis == 1
   */
  mean(): DataFrame
  mean(axis: 0): DataFrame
  mean(axis: 1): Series<any>
  mean(axis: 1, nullStrategy?: "ignore" | "propagate"): Series<any>
  /**
   * Aggregate the columns of this DataFrame to their median value.
   * ___
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.median()
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ f64 ┆ f64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 2   ┆ 7   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  median(): DataFrame
  /**
   * Unpivot DataFrame to long format.
   * ___
   *
   * @param idVars - Columns to use as identifier variables.
   * @param valueVars - Values to use as identifier variables.
   */
  melt(idVars: ColumnSelection, valueVars: ColumnSelection): DataFrame
  /**
   * Aggregate the columns of this DataFrame to their minimum value.
   * ___
   * @param axis - either 0 or 1
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.min()
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ i64 ┆ i64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 1   ┆ 6   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  min(): DataFrame
  min(axis: 0): DataFrame
  min(axis: 1): Series<any>
  /**
   * Get number of chunks used by the ChunkedArrays of this DataFrame.
   */
  nChunks(): number
  /**
   * Create a new DataFrame that shows the null counts per column.
   * ___
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, null, 3],
   * >>>   "bar": [6, 7, null],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.nullCount()
   * shape: (1, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ u32 ┆ u32 ┆ u32 │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 1   ┆ 0   │
   * └─────┴─────┴─────┘
   * ```
   */
  nullCount(): DataFrame
  // TODO!
  // /**
  //  * Apply a function on Self.
  //  */
  // pipe<T>(func: (...args: any[]) => T, ...args: any[]): T
  /**
   * Aggregate the columns of this DataFrame to their quantile value.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.quantile(0.5)
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ i64 ┆ i64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 2   ┆ 7   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  quantile(quantile: number): DataFrame
  /**
   * __Rechunk the data in this DataFrame to a contiguous allocation.__
   *
   * This will make sure all subsequent operations have optimal and predictable performance.
   */
  rechunk(): DataFrame
  /**
   * __Rename column names.__
   * ___
   *
   * @param mapping - Key value pairs that map from old name to new name.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.rename({"foo": "apple"})
   * ╭───────┬─────┬─────╮
   * │ apple ┆ bar ┆ ham │
   * │ ---   ┆ --- ┆ --- │
   * │ i64   ┆ i64 ┆ str │
   * ╞═══════╪═════╪═════╡
   * │ 1     ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2     ┆ 7   ┆ "b" │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3     ┆ 8   ┆ "c" │
   * ╰───────┴─────┴─────╯
   * ```
   */
  rename(mapping: Record<string, string>): DataFrame
  /**
   * Replace a column at an index location.
   * ___
   * @param index - Column index
   * @param newColumn - New column to insert
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> x = pl.Series("apple", [10, 20, 30])
   * >>> df.replaceAtIdx(0, x)
   * shape: (3, 3)
   * ╭───────┬─────┬─────╮
   * │ apple ┆ bar ┆ ham │
   * │ ---   ┆ --- ┆ --- │
   * │ i64   ┆ i64 ┆ str │
   * ╞═══════╪═════╪═════╡
   * │ 10    ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 20    ┆ 7   ┆ "b" │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 30    ┆ 8   ┆ "c" │
   * ╰───────┴─────┴─────╯
   * ```
   */
  replaceAtIdx(index: number, newColumn: Series<any>): void
  /**
   * Get a row as Array
   * @param index - row index
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.row(2)
   * [3, 8, 'c']
   * ```
   */
  row(index: number): Array<any>
  /**
   * Convert columnar data to rows as arrays
   */
  rows(): Array<Array<any>>
  /**
   * Sample from this DataFrame by setting either `n` or `frac`.
   * @param n - Number of samples < self.len() .
   * @param frac - Fraction between 0.0 and 1.0 .
   * @param withReplacement - Sample with replacement.
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
  sample(opts: {n: number, withReplacement?: boolean}): DataFrame
  sample(opts: {frac: number, withReplacement?: boolean}): DataFrame
  sample(n?: number, frac?: number, withReplacement?: boolean): DataFrame
  schema(): Record<string, string>
  /**
   * Select columns from this DataFrame.
   * ___
   * @param columns - Column or columns to select.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>     "foo": [1, 2, 3],
   * >>>     "bar": [6, 7, 8],
   * >>>     "ham": ['a', 'b', 'c']
   * >>>     })
   * >>> df.select('foo')
   * shape: (3, 1)
   * ┌─────┐
   * │ foo │
   * │ --- │
   * │ i64 │
   * ╞═════╡
   * │ 1   │
   * ├╌╌╌╌╌┤
   * │ 2   │
   * ├╌╌╌╌╌┤
   * │ 3   │
   * └─────┘
   * ```
   */
  select(column: ExprOrString, ...columns: ExprOrString[]): DataFrame
  /**
   * Shift the values by a given period and fill the parts that will be empty due to this operation
   * with `Nones`.
   * ___
   * @param periods - Number of places to shift (may be negative).
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.shift(1)
   * shape: (3, 3)
   * ┌──────┬──────┬──────┐
   * │ foo  ┆ bar  ┆ ham  │
   * │ ---  ┆ ---  ┆ ---  │
   * │ i64  ┆ i64  ┆ str  │
   * ╞══════╪══════╪══════╡
   * │ null ┆ null ┆ null │
   * ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 1    ┆ 6    ┆ "a"  │
   * ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 2    ┆ 7    ┆ "b"  │
   * └──────┴──────┴──────┘
   * >>> df.shift(-1)
   * shape: (3, 3)
   * ┌──────┬──────┬──────┐
   * │ foo  ┆ bar  ┆ ham  │
   * │ ---  ┆ ---  ┆ ---  │
   * │ i64  ┆ i64  ┆ str  │
   * ╞══════╪══════╪══════╡
   * │ 2    ┆ 7    ┆ "b"  │
   * ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ 3    ┆ 8    ┆ "c"  │
   * ├╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
   * │ null ┆ null ┆ null │
   * └──────┴──────┴──────┘
   * ```
   */
  shift(periods: number): DataFrame
  shift({periods}: {periods: number}): DataFrame
  /**
   * Shift the values by a given period and fill the parts that will be empty due to this operation
   * with the result of the `fill_value` expression.
   * ___
   * @param opts
   * @param opts.periods - Number of places to shift (may be negative).
   * @param opts.fillValue - fill null values with this value.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.shiftAndFill({periods:1, fill_value:0})
   * shape: (3, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 0   ┆ 0   ┆ "0" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 1   ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ "b" │
   * └─────┴─────┴─────┘
   * ```
   */
  shiftAndFill(periods: number, fillValue: number | string): DataFrame
  shiftAndFill({periods, fillValue}: {periods: number, fillValue: number | string}): DataFrame
  /**
   * Shrink memory usage of this DataFrame to fit the exact capacity needed to hold the data.
   */
  shrinkToFit(): DataFrame
  shrinkToFit(inPlace: true): void
  shrinkToFit({inPlace}: {inPlace: true}): void
  /**
   * Slice this DataFrame over the rows direction.
   * ___
   * @param opts
   * @param opts.offset - Offset index.
   * @param opts.length - Length of the slice
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6.0, 7.0, 8.0],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.slice(1, 2) // Alternatively `df.slice({offset:1, length:2})`
   * shape: (2, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 2   ┆ 7   ┆ "b" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" │
   * └─────┴─────┴─────┘
   * ```
   */
  slice({offset, length}: {offset: number, length: number}): DataFrame
  slice(offset: number, length: number): DataFrame
  /**
   * Sort the DataFrame by column.
   * ___
   * @param by - By which columns to sort. Only accepts string.
   * @param reverse - Reverse/descending sort.
   */
  sort(by: ColumnsOrExpr, reverse?: boolean): DataFrame
  sort({by, reverse}: {by: ColumnsOrExpr, reverse?: boolean}): DataFrame
  /**
   * Aggregate the columns of this DataFrame to their standard deviation value.
   * ___
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.std()
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ f64 ┆ f64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 1   ┆ 1   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  std(): DataFrame
  /**
   * Aggregate the columns of this DataFrame to their mean value.
   * ___
   *
   * @param axis - either 0 or 1
   * @param nullStrategy - this argument is only used if axis == 1
   */
  sum(): DataFrame
  sum(axis: 0): DataFrame
  sum(axis: 1): Series<any>
  sum(axis: 1, nullStrategy?: "ignore" | "propagate"): Series<any>
  /**
   *
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "letters": ["c", "c", "a", "c", "a", "b"],
   * >>>   "nrs": [1, 2, 3, 4, 5, 6]
   * >>> })
   * >>> df
   * shape: (6, 2)
   * ╭─────────┬─────╮
   * │ letters ┆ nrs │
   * │ ---     ┆ --- │
   * │ str     ┆ i64 │
   * ╞═════════╪═════╡
   * │ "c"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 2   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 3   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 4   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 5   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 6   │
   * ╰─────────┴─────╯
   * >>> df.groupby("letters")
   * >>>   .tail(2)
   * >>>   .sort("letters")
   * >>>
   * shape: (5, 2)
   * ╭─────────┬─────╮
   * │ letters ┆ nrs │
   * │ ---     ┆ --- │
   * │ str     ┆ i64 │
   * ╞═════════╪═════╡
   * │ "a"     ┆ 3   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 5   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 6   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 2   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 4   │
   * ╰─────────┴─────╯
   * ```
   */
  tail(length?: number): DataFrame
  /**
   * __Write DataFrame to comma-separated values file (csv).__
   *
   * If no options are specified, it will return a new string containing the contents
   * ___
   * @param dest file or stream to write to
   * @param options
   * @param options.hasHeader - Whether or not to include header in the CSV output.
   * @param options.sep - Separate CSV fields with this symbol. _defaults to `,`_
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.toCSV()
   * foo,bar,ham
   * 1,6,a
   * 2,7,b
   * 3,8,c
   *
   * // using a file path
   * >>> df.head(1).toCSV("./foo.csv")
   * // foo.csv
   * foo,bar,ham
   * 1,6,a
   *
   * // using a write stream
   * >>> const writeStream = new Stream.Writable({
   * >>>   write(chunk, encoding, callback) {
   * >>>     console.log("writeStream: %O', chunk.toString());
   * >>>     callback(null);
   * >>>   }
   * >>> });
   * >>> df.head(1).toCSV(writeStream, {hasHeader: false})
   * writeStream: '1,6,a'
   * ```
   */
  toCSV(): string;
  toCSV(options: WriteCsvOptions): string;
  toCSV(dest: string | Writable, options?: WriteCsvOptions): void;
  /**
   * Converts dataframe object into javascript object
   * Same logic applies for `toJSON` except this will use js values instead of a json string
   * @param options
   * @param options.orient - col|row|dataframe
   *
   * @example
   * ```
   * >>> const df = pl.DataFrame({
   * >>>   foo: [1,2,3],
   * >>>   bar: ['a','b','c']
   * >>> })
   *
   * // defaults to 'dataframe' orientation
   * >>> df.toObject()
   * {
   *   "columns":[
   *     {
   *       "name":"foo",
   *       "datatype":"Float64",
   *       "values":[1,2,3]
   *     },
   *     {
   *       "name":"bar",
   *       "datatype":"Utf8",
   *       "values":["a","b","c"]
   *     }
   *   ]
   * }
   *
   * // row oriented
   * >>> df.toObject({orient:"row"})
   * [
   *   {"foo":1.0,"bar":"a"},
   *   {"foo":2.0,"bar":"b"},
   *   {"foo":3.0,"bar":"c"}
   * ]
   *
   * // column oriented
   * >>> df.toObject({orient: "col"})
   * {
   *   "foo":[1,2,3],
   *   "bar":["a","b","c"]
   * }
   * ```
   */
  toObject(): object
  toObject(options: {orient: "row" | "col" | "dataframe"}): object

  /**
   * Write Dataframe to JSON string, file, or write stream
   * @param destination file or write stream
   * @param options
   * @param options.orient - col|row|dataframe
   *  - col will write to a column oriented object
   *
   * @example
   * ```
   * >>> const df = pl.DataFrame({
   * >>>   foo: [1,2,3],
   * >>>   bar: ['a','b','c']
   * >>> })
   *
   * // defaults to 'dataframe' orientation
   * >>> df.toJSON()
   * `{"columns":[ {"name":"foo","datatype":"Float64","values":[1,2,3]}, {"name":"bar","datatype":"Utf8","values":["a","b","c"]}]}`
   *
   * // this will produce the same results as 'df.toJSON()'
   * >>> JSON.stringify(df)
   *
   * // row oriented
   * >>> df.toJSON({orient:"row"})
   * `[ {"foo":1.0,"bar":"a"}, {"foo":2.0,"bar":"b"}, {"foo":3.0,"bar":"c"}]`
   *
   * // column oriented
   * >>> df.toJSON({orient: "col"})
   * `{"foo":[1,2,3],"bar":["a","b","c"]}`
   *
   * // multiline (will always be row oriented)
   * >>> df.toJSON({multiline: true})
   * `{"foo":1.0,"bar":"a"}
   * {"foo":2.0,"bar":"b"}
   * {"foo":3.0,"bar":"c"}`
   *
   * // writing to a file
   * >>> df.toJSON("/path/to/file.json", {multiline:true})
   * ```
   */
  toJSON(options?: WriteJsonOptions): string
  toJSON(destination: string | Writable, options?: WriteJsonOptions): void

  /**
   * Write to Arrow IPC binary stream, or a feather file.
   * @param file File path to which the file should be written.
   * @param options.compression Compression method *defaults to "uncompressed"*
   * */
   toIPC(options?: WriteIPCOptions): Buffer
   toIPC(destination: string | Writable, options?: WriteIPCOptions): void

  /**
   * Write the DataFrame disk in parquet format.
   * @param file File path to which the file should be written.
   * @param options.compression Compression method *defaults to "uncompressed"*
   * */
  toParquet(options?: WriteParquetOptions): Buffer
  toParquet(destination: string | Writable, options?: WriteParquetOptions): void
  toSeries(index: number): Series<any>
  toString(): string
  /**
   * Transpose a DataFrame over the diagonal.
   *
   * @note This is a very expensive operation. Perhaps you can do it differently.
   * @param options
   * @param options.includeHeader If set, the column names will be added as first column.
   * @param options.headerName If `includeHeader` is set, this determines the name of the column that will be inserted
   * @param options.columnNames Optional generator/iterator that yields column names. Will be used to replace the columns in the DataFrame.
   *
   * @example
   * >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
   * >>> df.transpose({includeHeader:true})
   * shape: (2, 4)
   * ┌────────┬──────────┬──────────┬──────────┐
   * │ column ┆ column_0 ┆ column_1 ┆ column_2 │
   * │ ---    ┆ ---      ┆ ---      ┆ ---      │
   * │ str    ┆ i64      ┆ i64      ┆ i64      │
   * ╞════════╪══════════╪══════════╪══════════╡
   * │ a      ┆ 1        ┆ 2        ┆ 3        │
   * ├╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┤
   * │ b      ┆ 1        ┆ 2        ┆ 3        │
   * └────────┴──────────┴──────────┴──────────┘
   * // replace the auto generated column names with a list
   * >>> df.transpose({includeHeader:false, columnNames:["a", "b", "c"]})
   * shape: (2, 3)
   * ┌─────┬─────┬─────┐
   * │ a   ┆ b   ┆ c   │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ i64 │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 2   ┆ 3   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 1   ┆ 2   ┆ 3   │
   * └─────┴─────┴─────┘
   *
   * // Include the header as a separate column
   * >>> df.transpose({
   * ...     includeHeader:true,
   * ...     headerName:"foo",
   * ...     columnNames:["a", "b", "c"]
   * ... })
   * shape: (2, 4)
   * ┌─────┬─────┬─────┬─────┐
   * │ foo ┆ a   ┆ b   ┆ c   │
   * │ --- ┆ --- ┆ --- ┆ --- │
   * │ str ┆ i64 ┆ i64 ┆ i64 │
   * ╞═════╪═════╪═════╪═════╡
   * │ a   ┆ 1   ┆ 2   ┆ 3   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ b   ┆ 1   ┆ 2   ┆ 3   │
   * └─────┴─────┴─────┴─────┘
   *
   * // Replace the auto generated column with column names from a generator function
   * >>> function *namesGenerator() {
   * ...     const baseName = "my_column_";
   * ...     let count = 0;
   * ...     let name = `${baseName}_${count}`;
   * ...     count++;
   * ...     yield name;
   * ... }
   * >>> df.transpose({includeHeader:false, columnNames:namesGenerator})
   * shape: (2, 3)
   * ┌─────────────┬─────────────┬─────────────┐
   * │ my_column_0 ┆ my_column_1 ┆ my_column_2 │
   * │ ---         ┆ ---         ┆ ---         │
   * │ i64         ┆ i64         ┆ i64         │
   * ╞═════════════╪═════════════╪═════════════╡
   * │ 1           ┆ 2           ┆ 3           │
   * ├╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
   * │ 1           ┆ 2           ┆ 3           │
   * └─────────────┴─────────────┴─────────────┘
   */
  transpose(options?: {includeHeader?: boolean, headerName?: string, columnNames?: Iterable<string>})
   /**
   * Aggregate the columns of this DataFrame to their variance value.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.var()
   * shape: (1, 3)
   * ╭─────┬─────┬──────╮
   * │ foo ┆ bar ┆ ham  │
   * │ --- ┆ --- ┆ ---  │
   * │ f64 ┆ f64 ┆ str  │
   * ╞═════╪═════╪══════╡
   * │ 1   ┆ 1   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  var(): DataFrame
  /**
   * Grow this DataFrame vertically by stacking a DataFrame to it.
   * @param df - DataFrame to stack.
   * @example
   * ```
   * >>> df1 = pl.DataFrame({
   * >>>   "foo": [1, 2],
   * >>>   "bar": [6, 7],
   * >>>   "ham": ['a', 'b']
   * >>> })
   * >>> df2 = pl.DataFrame({
   * >>>   "foo": [3, 4],
   * >>>   "bar": [8 , 9],
   * >>>   "ham": ['c', 'd']
   * >>> })
   * >>> df1.vstack(df2)
   * shape: (4, 3)
   * ╭─────┬─────┬─────╮
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ "a" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ "b" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   ┆ "c" │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 4   ┆ 9   ┆ "d" │
   * ╰─────┴─────┴─────╯
   * ```
   */
  vstack(df: DataFrame): DataFrame
  /**
   * Return a new DataFrame with the column added or replaced.
   * @param column - Series, where the name of the Series refers to the column in the DataFrame.
   */
  withColumn(column: Series<any> | Expr): DataFrame
  withColumn(column: Series<any> | Expr): DataFrame
  withColumns(column: Series<any> | Expr, ...columns: Expr[] | Series<any>[] ): DataFrame
  /**
   * Return a new DataFrame with the column renamed.
   * @param existingName
   * @param newName
   */
  withColumnRenamed(existing: string, replacement: string): DataFrame
  withColumnRenamed(opts: {existing: string, replacement: string}): DataFrame
  /**
   * Add a column at index 0 that counts the rows.
   * @param name - name of the column to add
   */
  withRowCount(name?: string): DataFrame
  /** @see {@link filter} */
  where(predicate: any): DataFrame
}

function prepareOtherArg<T>(anyValue: T | Series<T>): Series<T> {
  if(Series.isSeries(anyValue)) {

    return anyValue;
  } else {
    return Series([anyValue]) as Series<T>;
  }
}

function map<T>(df: DataFrame, fn: (...args: any[]) => T[]) {

  return df.rows().map(fn);
}

/**
 * @ignore
 */
export const dfWrapper = (_df: JsDataFrame): DataFrame => {
  const unwrap = <U>(method: string, args?: object, df=_df): U => {

    return pli.df[method]({_df: df, ...args });
  };
  const wrap = (method, args?, df=_df): DataFrame => {
    return dfWrapper(unwrap(method, args, df));
  };
  const noArgWrap = (method: string) => () => wrap(method);
  const noArgUnwrap = <U>(method: string) => () => unwrap<U>(method);

  const writeToStreamOrString = (dest, format: "csv" | "json", options)  => {
    if(dest instanceof Writable) {
      unwrap(`write_${format}_stream`, {writeStream: dest, ...options});

      dest.end("");

    } else if (typeof dest === "string") {
      unwrap(`write_${format}_path`, {path: dest, ...options});

    } else {
      let body = "";
      const writeStream = new Stream.Writable({
        write(chunk, _encoding, callback) {
          body += chunk;
          callback(null);
        }
      });

      unwrap(`write_${format}_stream`, {writeStream, ...options, ...dest});
      writeStream.end("");

      return body;
    }
  };

  const writeToBufferOrStream = (dest, format: "ipc" | "parquet", options) => {
    if(dest instanceof Writable) {
      unwrap(`write_${format}_stream`, {writeStream: dest, ...options});

      dest.end("");


    } else if (typeof dest === "string") {
      return unwrap(`write_${format}_path`, {path: dest, ...options});

    } else {
      let buffers: Buffer[] = [];
      const writeStream = new Stream.Writable({
        write(chunk, _encoding, callback) {
          buffers.push(chunk);
          callback(null);
        }
      });

      unwrap(`write_${format}_stream`, {writeStream, ...options, ...dest});
      writeStream.end("");

      return Buffer.concat(buffers);
    }
  };
  const df = {
    /** @ignore */
    _df,
    [inspect]() {
      return unwrap<Buffer>("as_str").toString();
    },
    *[Symbol.iterator]() {

      let start = 0;
      let len = this.width;

      while (start < len) {
        const s = this.toSeries(start);
        start++;
        yield s;
      }
    },
    get dtypes() {
      return unwrap<DataType[]>("dtypes");
    },
    get height() {
      return unwrap<number>("height");
    },
    get width() {
      return unwrap<number>("width");
    },
    get shape() {
      return {height: this.height, width: this.width};
    },
    get columns() {
      return unwrap<string[]>("columns");
    },
    set columns(names) {
      unwrap<string[]>("set_column_names", {names});
    },
    clone: noArgWrap("clone"),
    describe() {
      const describeCast = (df: DataFrame) => {
        return DataFrame(df.getColumns().map(s => {
          if(s.isNumeric() || s.isBoolean()) {

            return s.cast(DataType.Float64);
          } else {
            return s;
          }
        }));
      };
      const summary = concat([
        describeCast(this.mean()),
        describeCast(this.std()),
        describeCast(this.min()),
        describeCast(this.max()),
        describeCast(this.median())
      ]);
      summary.insertAtIdx(
        0,
        Series(
          "describe",
          ["mean", "std", "min", "max", "median"]
        )
      );

      return summary;
    },
    drop(...names) {
      if(!Array.isArray(names[0]) && names.length === 1) {
        return wrap("drop", {name: names[0]});
      }

      const df = this.clone();

      names.flat(2).forEach((name) => {
        unwrap("drop_in_place", {name}, df._df);
      });

      return df;


    },
    dropNulls(...subset) {
      if(subset.length) {
        return wrap("drop_nulls", {subset: subset.flat(2)});
      } else {
        return wrap("drop_nulls");
      }
    },
    distinct(opts: any = false, subset?, keep = "first") {
      const defaultOptions = {
        maintainOrder: false,
        keep,
      };

      if(typeof opts === "boolean") {
        return wrap("distinct", {...defaultOptions, maintainOrder: opts, subset, keep});
      }

      if(opts.subset) {
        opts.subset = [opts.subset].flat(3);
      }

      return wrap("distinct", {...defaultOptions, ...opts});
    },
    dropDuplicates(opts: any=false, subset?) {
      return this.distinct(opts, subset);
    },
    explode(...columns)  {
      return dfWrapper(_df)
        .lazy()
        .explode(columns)
        .collectSync({noOptimization:true});
    },
    extend(other) {
      return wrap("extend", {other: other._df});
    },
    filter(predicate)  {
      return this
        .lazy()
        .filter(predicate)
        .collectSync();
    },
    fillNull: (strategy) => wrap("fill_null", {strategy}),
    findIdxByName: (name) => unwrap("find_idx_by_name", {name}),
    fold(fn: (s1, s2) => Series<any>) {
      if(this.width === 1) {
        return this.toSeries(0);
      }

      return this.getColumns().reduce((acc, curr) => fn(acc, curr));

    },
    frameEqual(other, nullEqual=true) {
      return unwrap<boolean>("frame_equal", {other: other._df, nullEqual});
    },
    getColumn(name) {
      return seriesWrapper(unwrap<any[]>("column", {name}));
    },
    getColumns() {
      return unwrap<any[]>("get_columns").map(s => seriesWrapper(s));
    },
    groupBy: (...by) => GroupBy(_df, columnOrColumnsStrict(by)),
    hashRows(obj: any = 0, k1=1, k2=2, k3=3) {
      if(typeof obj === "number" || typeof obj === "bigint") {
        return seriesWrapper(unwrap("hash_rows", { k0: obj, k1: k1, k2: k2, k3: k3 })) as any;
      }

      return seriesWrapper(unwrap("hash_rows", { k0: 0, k1, k2, k3, ...obj })) as any;
    },
    head: (length=5) => wrap("head", {length}),
    hstack(columns) {
      if(!Array.isArray(columns)) {
        columns = columns.getColumns();
      }

      return wrap("hstack", {
        columns: columns.map(col => col._series),
        in_place: false
      });
    },
    insertAtIdx: (index, s) => unwrap("insert_at_idx", {index, new_col: s._series}),
    interpolate() {
      return this.select(col("*").interpolate());
    },
    isDuplicated: () => seriesWrapper(unwrap("is_duplicated")),
    isEmpty: () => unwrap("height") === 0,
    isUnique: () => seriesWrapper(unwrap("is_unique")),
    join(df: DataFrame, options): DataFrame  {
      options =  {how: "inner", suffix: "right", ...options};
      const on = columnOrColumns(options.on);
      const how = options.how;
      const suffix = options.suffix;

      let leftOn = columnOrColumns(options.leftOn);
      let rightOn = columnOrColumns(options.rightOn);

      if(on) {
        leftOn = on;
        rightOn = on;
      }
      if((leftOn && !rightOn) || (rightOn && !leftOn)) {
        throw new TypeError("You should pass the column to join on as an argument.");
      }

      return wrap("join", {
        other: df._df,
        on,
        how,
        left_on: leftOn,
        right_on: rightOn,
        suffix,
      });
    },
    lazy: () => LazyDataFrame(unwrap("lazy")),
    limit: (length=5) => wrap("head", {length}),
    max(axis=0) {
      if(axis === 1) {
        return seriesWrapper(unwrap("hmax")) as any;
      } else {
        return wrap("max");
      }
    },
    mean(axis=0, nullStrategy="ignore") {
      if(axis === 1) {
        return seriesWrapper(unwrap("hmean", {nullStrategy})) as any;
      }

      return wrap("mean");
    },
    median: noArgWrap("median"),
    melt(ids, values) {
      return wrap("melt", {
        idVars: columnOrColumns(ids),
        valueVars: columnOrColumns(values)
      });
    },
    min(axis=0) {
      if(axis === 1) {
        return seriesWrapper(unwrap("hmin")) as any;
      } else {
        return wrap("min");
      }
    },
    nChunks: noArgUnwrap("n_chunks"),
    nullCount: noArgWrap("null_count"),
    quantile: (quantile) => wrap("quantile", {quantile}),
    rechunk: noArgWrap("rechunk"),
    rename(mapping)  {
      const df = this.clone();
      Object.entries(mapping).forEach(([column, new_col]) => {
        unwrap("rename", {column, new_col}, df._df);
      });

      return df;
    },
    replaceAtIdx(index, newColumn) {
      unwrap("replace_at_idx", {
        index,
        newColumn: newColumn._series
      });

      return this;
    },
    rows: noArgUnwrap("to_rows"),
    sample(opts?, frac?, withReplacement = false) {
      if(opts?.n  !== undefined || opts?.frac  !== undefined) {
        return this.sample(opts.n, opts.frac, opts.withReplacement);
      }
      if (typeof opts === "number") {
        return wrap("sample_n", {
          n: opts,
          withReplacement
        });
      }
      if(typeof frac === "number") {
        return wrap("sample_frac", {
          frac,
          withReplacement,
        });
      }
      else {
        throw new TypeError("must specify either 'frac' or 'n'");
      }
    },
    schema: noArgUnwrap("schema"),
    select(...selection) {
      const hasExpr = selection.flat().some(s => Expr.isExpr(s));
      if(hasExpr) {
        return dfWrapper(_df)
          .lazy()
          .select(selection)
          .collectSync();
      } else {

        return wrap("select", {selection: columnOrColumnsStrict(selection as any)});
      }
    },
    shift: (opt) => wrap("shift", {periods: opt?.periods ?? opt }),
    shiftAndFill(periods: any, fillValue?)  {
      return dfWrapper(_df)
        .lazy()
        .shiftAndFill(periods, fillValue)
        .collectSync();
    },
    shrinkToFit(inPlace: any=false): any {
      if(inPlace) {
        unwrap("shrink_to_fit");
      } else {
        const d = this.clone();
        unwrap("shrink_to_fit", {}, d._df);

        return d;
      }
    },
    slice(opts, length?) {
      if(typeof opts === "number") {
        return wrap("slice", {offset: opts, length});
      }

      return wrap("slice", opts);
    },
    sort(arg,  reverse=false)  {

      if(arg?.by  !== undefined) {
        return this.sort(arg.by, arg.reverse);
      }
      if(Array.isArray(arg) || Expr.isExpr(arg)) {
        return dfWrapper(_df).lazy()
          .sort(arg, reverse)
          .collectSync({noOptimization: true, stringCache: false});

      }

      return wrap("sort", {by: [arg].flat(), reverse});

    },
    std: noArgWrap("std"),
    sum(axis=0, nullStrategy="ignore") {
      if(axis === 1) {
        return seriesWrapper(unwrap("hsum", {nullStrategy})) as any;
      }

      return wrap("sum");
    },
    tail: (length=5) => wrap("tail", {length}),
    toCSV(dest?, options?) {
      options = { hasHeader:true, sep: ",", ...options};
      // toCSV(options)
      if(dest?.hasHeader !== undefined || dest?.sep !== undefined) {
        return writeToStreamOrString(null, "csv", {...options, ...dest});

      } else {
        // toCSV()
        // toCSV("path/to/some/file", options)
        // toCSV(writeStream, options)
        return writeToStreamOrString(dest, "csv", options);
      }
    },
    toObject(options?) {
      if(options?.orient === "row") {
        return unwrap("to_row_objects");

      }
      if(options?.orient === "col") {
        return this.getColumns().reduce((acc, curr: Series<any>) => ({
          ...acc,
          [curr.name]: curr.toArray()
        }), {});
      }

      return unwrap("to_js");
    },
    toJSON(arg0?, options?): any {
      // JSON.stringify(df)
      // passes `""` by default then stringifies the JS output
      if(arg0 === "") {
        return unwrap("to_js");
      }

      // df.toJSON();
      // same thing as to_js except skips serializing to JS & returns a json buffer
      if(arg0 === undefined) {
        return unwrap<any>("to_json").toString();
      }

      // toJSON(options)
      if(arg0?.orient === "row" || arg0?.multiline) {
        return writeToStreamOrString(null, "json", arg0);

      // toJSON({orient:"col"})
      } else if(arg0?.orient === "col") {
        // TODO!
        // do this on the rust side for better performance
        return JSON.stringify(this.toObject({orient: "col"}));
      }
      else {
        // toJSON("path/to/some/file", options)
        // toJSON(writeStream, options)
        return writeToStreamOrString(arg0, "json", options);
      }
    },
    toParquet(dest?, options = {compression: "uncompressed"}) {
      if(dest?.compression !== undefined) {
        return writeToBufferOrStream(null, "parquet", dest);
      } else {
        return writeToBufferOrStream(dest, "parquet", options);
      }
    },
    toIPC(dest?, options = {compression: "uncompressed"}) {
      if(dest?.compression !== undefined) {
        return writeToBufferOrStream(null, "ipc", dest);
      } else {
        return writeToBufferOrStream(dest, "ipc", options);
      }
    },
    toSeries: (index) => seriesWrapper(unwrap("select_at_idx", {index})),
    toString: () => noArgUnwrap<any>("as_str")().toString(),
    transpose(options?) {

      let df = wrap("transpose", options);
      if(options?.columnNames) {

        function *namesIter() {
          if(options?.includeHeader) {
            yield options.headerName;
          }
          const gen = (options as any).columnNames[Symbol.iterator]();
          let next;
          // eslint-disable-next-line no-cond-assign
          while (next = gen.next()) {
            yield next.value;
          }
        }

        const newColumns = Array.from(
          {length: df.width},
          (i => () => i.next().value)(namesIter())
        );

        df.columns = newColumns;
      }

      return df;
    },
    var: noArgWrap("var"),
    map: (fn) => map(dfWrapper(_df), fn as any) as any,
    row: (index) => unwrap("to_row", {idx: index}),
    vstack: (other) => wrap("vstack", {other: other._df}),
    withColumn(column: Series<any> | Expr) {
      if(Series.isSeries(column)) {
        return wrap("with_column", {_series: column._series});
      } else {
        return this.withColumns(column);
      }
    },
    withColumns(column, ...columns: Expr[] | Series<any>[]) {
      columns.unshift(column as any);

      if(isSeriesArray(columns)) {
        return columns.reduce((acc, curr) => acc.withColumn(curr), dfWrapper(_df));
      } else {
        return this
          .lazy()
          .withColumns(columns)
          .collectSync({noOptimization: true, stringCache: false});
      }

    },
    withColumnRenamed(opt, replacement?) {
      if(typeof opt === "string") {
        return this.rename({[opt]: replacement});
      } else {
        return this.rename({[opt.existing]: opt.replacement});
      }
    },
    withRowCount: (name="row_nr") => wrap("with_row_count", {name}),
    where(predicate) {
      return this.filter(predicate);
    },

    add: (other) =>  wrap("add", {other: prepareOtherArg(other)._series}),
    sub: (other) =>  wrap("sub", {other: prepareOtherArg(other)._series}),
    div: (other) =>  wrap("div", {other: prepareOtherArg(other)._series}),
    mul: (other) =>  wrap("mul", {other: prepareOtherArg(other)._series}),
    rem: (other) =>  wrap("rem", {other: prepareOtherArg(other)._series}),
    plus: (other) =>  wrap("add", {other: prepareOtherArg(other)._series}),
    minus: (other) =>  wrap("sub", {other: prepareOtherArg(other)._series}),
    divideBy: (other) =>  wrap("div", {other: prepareOtherArg(other)._series}),
    multiplyBy: (other) =>  wrap("mul", {other: prepareOtherArg(other)._series}),
    modulo: (other) =>  wrap("rem", {other: prepareOtherArg(other)._series}),
  } as DataFrame;

  return new Proxy(df, {
    get(target: DataFrame, prop, receiver) {
      if(typeof prop === "string" && target.columns.includes(prop)) {

        return target.getColumn(prop);
      } if(typeof prop !== "symbol" && !Number.isNaN(Number(prop))) {

        return target.row(Number(prop));
      } else {

        return Reflect.get(target, prop, receiver);
      }
    },
    set(target: DataFrame, prop, receiver) {
      if(Series.isSeries(receiver)) {
        if(typeof prop === "string" && target.columns.includes(prop)) {
          const idx = target.columns.indexOf(prop);
          target.replaceAtIdx(idx, receiver.alias(prop));

          return true;
        }
      }

      Reflect.set(target, prop, receiver);

      return true;
    },
    has(target, p) {
      return target.columns.includes(p as any);
    },
    ownKeys(target) {
      return target.columns as any;
    },
    getOwnPropertyDescriptor(target, prop) {
      return {
        configurable: true,
        enumerable: true,
        value: target.getColumn(prop as any)
      };
    }
  });

};

export interface DataFrameConstructor {
  (): DataFrame
  (data: Record<string, any>[]): DataFrame
  (data: Series<any>[]): DataFrame
  (data: any): DataFrame
  (data: any[][], options: {columns?: any[], orient?: "row" | "col"}): DataFrame
  isDataFrame(arg: any): arg is DataFrame;
}
function DataFrameConstructor(data?, options?): DataFrame {

  if(!data) {
    return dfWrapper(objToDF({}));
  }

  if (Array.isArray(data)) {
    return dfWrapper(arrayToJsDataFrame(data, options?.columns, options?.orient));
  }

  return dfWrapper(objToDF(data as any));
}

function objToDF(obj: Record<string, Array<any>>): any {
  const columns =  Object.entries(obj).map(([name, values]) => {
    if(Series.isSeries(values)) {
      return values.rename(name)._series;
    }

    return Series(name, values)._series;
  });

  return pli.df.read_columns({columns});
}
const isDataFrame = (ty: any): ty is DataFrame => isExternal(ty?._df);
const fromRowArrays = (rows, inferSchemaLength) => {
  return dfWrapper(pli.df.read_rows({rows, inferSchemaLength}));
};

export namespace pl {
  export const DataFrame: DataFrameConstructor = Object.assign(DataFrameConstructor, {isDataFrame, fromRowArrays});
}

export const DataFrame: DataFrameConstructor = Object.assign(DataFrameConstructor, {isDataFrame, fromRowArrays});
