/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
import polarsInternal from './internals/polars_internal';
import { arrayToJsDataFrame, arrayToJsSeries } from './internals/construction';
import util from 'util';
import { DataType, JoinOptions, ReadCsvOptions, ReadJsonOptions, WriteCsvOptions} from './datatypes';
import {Series, _wrapSeries} from './series';
import {Stream} from 'stream';
import fs from 'fs';
import {isPath, columnOrColumns, columnOrColumnsStrict, ColumnSelection, range} from './utils';
import {GroupBy} from './groupby';

const todo = () => new Error('not yet implemented');


const defaultJoinOptions: JoinOptions = {
  how: "inner",
  suffix: "_right"
};

const readCsvDefaultOptions: Partial<ReadCsvOptions> = {
  inferSchemaLength: 10,
  batchSize: 10,
  ignoreErrors: true,
  hasHeader: true,
  sep: ",",
  rechunk: false,
  startRows: 0,
  encoding: "utf8",
  lowMemory: false,
  parseDates: true,
};


const readJsonDefaultOptions: Partial<ReadJsonOptions> = {
  batchSize: 1000,
  inline: false,
  inferSchemaLength: 10
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
export class DataFrame {
  private _df: any;

  constructor(df: any) {
    this._df = df;
  }
  /**
 * __Get dtypes of columns in DataFrame.__
 *
 * Dtypes can also be found in column headers when printing the DataFrame.
 * ___
 * @example
 * ```
 * > df = pl.DataFrame({
 * >   "foo": [1, 2, 3],
 * >   "bar": [6.0, 7.0, 8.0],
 * >   "ham": ['a', 'b', 'c']
 * > })
 * > df.dtypes
 *
 * ['Int64', 'Float64', 'Utf8']
 * df
 * shape: (3, 3)
 * ╭─────┬─────┬─────╮
 * │ foo ┆ bar ┆ ham │
 * │ --- ┆ --- ┆ --- │
 * │ i64 ┆ f64 ┆ str │
 * ╞═════╪═════╪═════╡
 * │ 1   ┆ 6   ┆ "a" │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 2   ┆ 7   ┆ "b" │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 3   ┆ 8   ┆ "c" │
 * ╰─────┴─────┴─────╯
 * ```
 */
  get dtypes (): Array<string> {
    return this.unwrap<number[]>('dtypes').map(d => DataType[d]) as any;
  }
  /**
   * __Get the height of the DataFrame.__
   * ___
   * @example
   * ```
   * df = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
   * df.height
   * 5
   * ```
   */
  get height (): number {
    return this.unwrap('height');
  }
  /**
   * __Get the shape of the DataFrame.__
   * ___
   * @example
   * ```
   * df = pl.DataFrame({"foo": [1,2,3,4,5]})
   * df.shape
   * {
   *   height: 5,
   *   width: 1
   * }
   * ```
   */
  get shape ():  {height: number, width: number} {
    return {
      height: this.height,
      width: this.width
    };
  }
  /**
   * __Get the width of the DataFrame.__
   * ___
   * @example
   * ```
   * df = pl.DataFrame({"foo": [1, 2, 3, 4, 5]})
   * df.width
   * 5
   * ```
   */
  get width ():  number {
    return this.unwrap('width');
  }

  /**
   * __Get or set column names.__
   * ___
   * @example
   * ```
   * > df = pl.DataFrame({
   * >   "foo": [1, 2, 3],
   * >   "bar": [6, 7, 8],
   * >   "ham": ['a', 'b', 'c']
   * > })
   *
   * > df.columns
   *
   * ['foo', 'bar', 'ham']
   *
   * // Set column names
   *
   * > df.columns = ['apple', 'banana', 'orange']
   *
   * shape: (3, 3)
   * ╭───────┬────────┬────────╮
   * │ apple ┆ banana ┆ orange │
   * │ ---   ┆ ---    ┆ ---    │
   * │ i64   ┆ i64    ┆ str    │
   * ╞═══════╪════════╪════════╡
   * │ 1     ┆ 6      ┆ "a"    │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   * │ 2     ┆ 7      ┆ "b"    │
   * ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌┤
   * │ 3     ┆ 8      ┆ "c"    │
   * ╰───────┴────────┴────────╯
   * ```
   */
  get columns ():  Array<string> {
    return this.unwrap('columns');
  }

  set columns(columns: Array<string>) {
    this.unwrap('set_column_names', {columns});
  }
  static [Symbol.for("wrap")](
    df: any,
    method: string,
    args?: object,
  ):DataFrame {
    return new DataFrame(polarsInternal.df[method]({ _df: df, ...args }));

  }
  /**
   * __Read a CSV file or string into a Dataframe.__
   * ___
    @param options
    @param options.file - Path to a file or a file like string. Any valid filepath can be used. Example: `file.csv`.
        Any string containing the contents of a csv can also be used
    @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
        If set to `null`, a full table scan will be done (slow).
    @param options.batchSize - Number of lines to read into the buffer at once. Modify this to change performance.
    @param options.hasHeader - Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`,
        `x` being an enumeration over every column in the dataset.
    @param options.ignoreErrors -Try to keep reading lines if some lines yield errors.
    @param options.endRows -After n rows are read from the CSV, it stops reading.
        During multi-threaded parsing, an upper bound of `n` rows
        cannot be guaranteed.
    @param options.startRows -Start reading after `startRows` position.
    @param options.projection -Indices of columns to select. Note that column indices start at zero.
    @param options.sep -Character to use as delimiter in the file.
    @param options.columns -Columns to select.
    @param options.rechunk -Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
    @param options.encoding -Allowed encodings: `utf8`, `utf8-lossy`. Lossy means that invalid utf8 values are replaced with `�` character.
    @param options.numThreads -Number of threads to use in csv parsing. Defaults to the number of physical cpu's of your system.
    @param options.dtype -Overwrite the dtypes during inference.
    @param options.lowMemory - Reduce memory usage in expense of performance.
    @param options.commentChar - character that indicates the start of a comment line, for instance '#'.
    @param options.quotChar -character that is used for csv quoting, default = ''. Set to null to turn special handling and escaping of quotes off.
    @param options.nullValues - Values to interpret as null values. You can provide a
        - `string` -> all values encountered equal to this string will be null
        - `Array<string>` -> A null value per column.
        - `Record<string,string>` -> An object or map that maps column name to a null value string.Ex. {"column_1": 0}
    @param options.parseDates -Whether to attempt to parse dates or not
    @returns DataFrame
   */
  static readCSV(options: Partial<ReadCsvOptions>): DataFrame
  static readCSV(path: string): DataFrame
  static readCSV(path: string, options: Partial<ReadCsvOptions>): DataFrame
  static readCSV(arg: Partial<ReadCsvOptions> | string, options?: any) {
    if(typeof arg === 'string') {

      return DataFrame.readCSV({...options, file: arg, inline: !isPath(arg)});
    }
    options = {...readCsvDefaultOptions, ...arg};

    return new DataFrame(polarsInternal.df.read_csv(options));
  }

  /**
   * __Read a JSON file or string into a DataFrame.__
   *
   * _Note: Currently only newline delimited JSON is supported_
   * @param options
   * @param options.file - Path to a file, or a file like string
   * @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
   *    If set to `null`, a full table scan will be done (slow).
   * @param options.batchSize - Number of lines to read into the buffer at once. Modify this to change performance.
   * @returns ({@link DataFrame})
   * @example
   * ```
   * const jsonString = `
   * {"a", 1, "b", "foo", "c": 3}
   * {"a": 2, "b": "bar", "c": 6}
   * `
   * > const df = pl.readJSON({file: jsonString})
   * > console.log(df)
   *   shape: (2, 3)
   * ╭─────┬─────┬─────╮
   * │ a   ┆ b   ┆ c   │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ str ┆ i64 │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ foo ┆ 3   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ bar ┆ 6   │
   * ╰─────┴─────┴─────╯
   * ```
   */
  static readJSON(options: ReadJsonOptions): DataFrame
  static readJSON(path: string): DataFrame
  static readJSON(path: string, options: ReadJsonOptions): DataFrame
  static readJSON(arg: ReadJsonOptions | string, options?: any) {

    if(typeof arg === 'string') {

      return DataFrame.readJSON({...options, file: arg, inline: !isPath(arg)});
    }
    options = {...readJsonDefaultOptions, ...arg};

    return new DataFrame(polarsInternal.df.read_json(options));
  }

  static of(data: Record<string, any[]>): DataFrame
  static of(data: Series<any>[]): DataFrame
  static of(data: any[][]): DataFrame
  static of(data: any[][], options: {columns?: any[], orient?: 'row' | 'col'}): DataFrame
  static of(
    data: Record<string, any[]> | any[][] | Series<any>[],
    options?: {columns?: any[], orient?: 'row' | 'col'}
  ): DataFrame {

    if(!data) {
      return new DataFrame(obj_to_df({}));
    }

    if (Array.isArray(data)) {
      return new DataFrame(arrayToJsDataFrame(data, options?.columns, options?.orient));
    }

    return new DataFrame(obj_to_df(data as any));
  }

  [util.inspect.custom](): string {
    return this.unwrap<any>('as_str');
  }
  /**
   * TODO
   * @param func
   */
  apply<U>(func: <T>(s: T) => U): DataFrame {
    throw todo();
  }

  /**
   * Very cheap deep clone.
   */
  clone(): DataFrame {
    return this.wrap('clone');
  }

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
  describe(): DataFrame {
    throw todo();
  }
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
  drop(name: string | Array<string>): DataFrame {
    if(Array.isArray(name)) {
      const df = this.clone();

      name.forEach((name) => {
        this.unwrap('drop_in_place', {name}, df._df);
      });

      return df;
    }

    return this.wrap('drop', {name});
  }
  /**
 * __Drop duplicate rows from this DataFrame.__
 *
 * Note that this fails if there is a column of type `List` in the DataFrame.
 * @param maintainOrder
 * @param subset - subset to drop duplicates for
 */
  dropDuplicates(maintainOrder:boolean, subset?: string | Array<string>): DataFrame
  /**
   * __Drop duplicate rows from this DataFrame.__
   *
   * Note that this fails if there is a column of type `List` in the DataFrame.
   * @param options
   * @param options.maintainOrder
   * @param options.subset - subset to drop duplicates for
   */
  dropDuplicates(options: {maintainOrder?: boolean, subset?: string | Array<string>} | boolean, subset?: string | Array<string>): DataFrame {
    throw todo();
  }
  /**
   * Drop in place.
   * ___
   *
   * @param name - Column to drop.
   * @example
   * ```
   * > df = pl.DataFrame({
   * >   "foo": [1, 2, 3],
   * >   "bar": [6, 7, 8],
   * >   "ham": ['a', 'b', 'c']
   * > })
   * > df.dropInPlace("ham")
   * shape: (3, 2)
   * ╭─────┬─────╮
   * │ foo ┆ bar │
   * │ --- ┆ --- │
   * │ i64 ┆ i64 │
   * ╞═════╪═════╡
   * │ 1   ┆ 6   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 3   ┆ 8   │
   * ╰─────┴─────╯
   * ```
   */
  dropInPlace(name: string): void {
    this.unwrap('drop_in_place', {name});
  }

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
  dropNulls(subset?: string | Array<string>): DataFrame {
    subset = columnOrColumns(subset);

    return this.wrap('drop_nulls', {subset});
  }

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
  explode(columns: string | Array<string>): DataFrame {
    throw todo();
  }

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
  fillNull(strategy: "backward" | "forward" | "mean" | "min" | "max" | "zero" | "one"): DataFrame {
    return this.wrap('fill_null', {strategy});
  }
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
  filter(predicate: any): DataFrame {
    throw todo();
  }

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
  findIdxByName(name: string): number {
    return this.unwrap('find_idx_by_name', {name});
  }
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

  fold<T, U, V>(operation: (s1: Series<T>, s2: Series<U>) => Series<V>): Series<V> {
    if(this.width === 1) {
      return this.toSeries(0);
    }
    const df = this;
    let acc: Series<any> = operation(df.toSeries(0), df.toSeries(1));

    for(let i of range(2, df.width)) {
      acc = operation(acc, df.toSeries(i));
    }

    return acc;
  }
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
  frameEqual({other,nullEqual}: {other: DataFrame, nullEqual?: boolean}): boolean
  frameEqual(arg: DataFrame | object, nullEqual=false): boolean {
    if(arg instanceof DataFrame) {
      return this.frameEqual({other: arg._df, nullEqual});
    } else {
      return this.unwrap('frame_equal', arg);
    }
  }

  /**
   * Get a single column as Series by name.
   */
  getColumn(name: string): Series<any> {
    return _wrapSeries(this.unwrap<any[]>('column', {name}));
  }
  /**
   * Get the DataFrame as an Array of Series.
   */
  getColumns(): Array<Series<any>> {
    return this.unwrap<any[]>('get_columns').map(s => _wrapSeries(s));
  }
  /**
   * Start a groupby operation.
   * ___
   * @param by - Column(s) to group by.
   */
  groupBy(...by: ColumnSelection[]): GroupBy {
    console.log({columns: columnOrColumnsStrict(by)});

    return GroupBy(
      this._df,
      columnOrColumnsStrict(by)
    );
  }

  // /**
  //  * Hash and combine the rows in this DataFrame. _(Hash value is UInt64)_
  //  * @param k0 - seed parameter
  //  * @param k1 - seed parameter
  //  * @param k2 - seed parameter
  //  * @param k3 - seed parameter
  //  */
  /**
   * Hash and combine the rows in this DataFrame. _(Hash value is UInt64)_
   * @param options
   * @param options.k0 - seed parameter
   * @param options.k1 - seed parameter
   * @param options.k2 - seed parameter
   * @param options.k3 - seed parameter
   */
  hashRows(options: {k0?: number,k1?: number,k2?: number,k3?: number}): Series<number>
  hashRows(k0?:number, k1?: number,k2?: number,k3?: number): Series<number> // hash_row
  hashRows(options?: object | number, k1?, k2?, k3?): Series<number> {
    const defaults = {k0:0, k1:1, k2:2, k3:3};

    if(options && typeof options !== 'number') {
      return _wrapSeries(this.unwrap<number>('hash_rows', {...defaults, ...options}));
    } else {
      return this.hashRows({
        ...defaults,
        k0: options ?? 0,
      });
    }
  }
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
  head(length = 5): DataFrame {
    return this.wrap('head', {length});
  }
  /**
   * Return a new DataFrame grown horizontally by stacking multiple Series to it.
   * @param columns - array of Series or DataFrame to stack
   * @param inPlace - Modify in place
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>     "foo": [1, 2, 3],
   * >>>     "bar": [6, 7, 8],
   * >>>     "ham": ['a', 'b', 'c']
   * >>>     })
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
  hStack(columns: Array<Series<any>> | DataFrame): DataFrame
  hStack(columns: Array<Series<any>> | DataFrame, inPlace?: boolean): void //hstack
  hStack(columns: Array<Series<any>> | DataFrame, inPlace?: boolean): DataFrame | void {
    if(inPlace) {
      throw todo();
    } else {
      if(!Array.isArray(columns)) {
        columns = columns.getColumns();
      }

      return this.wrap('hstack', {columns: columns.map(col => col._series), in_place: !!inPlace} );
    }
  }
  /**
   * Insert a Series at a certain column index. This operation is in place.
   * @param index - Column position to insert the new `Series` column.
   * @param series - `Series` to insert
   */
  insertAtIdx(index: number, series: Series<any>): void {
    this.unwrap('insert_at_idx', {index, new_col: series._series});
  }
  /**
   * Interpolate intermediate values. The interpolation method is linear.
   */
  interpolate(): DataFrame {
    return this.wrap('interpolate');
  }

  /**
   * Get a mask of all duplicated rows in this DataFrame.
   */
  isDuplicated(): Series<boolean> {
    return _wrapSeries(this.unwrap('is_duplicated'));
  }
  /**
   * Check if the dataframe is empty
   */
  isEmpty(): boolean {
    return this.height === 0;
  }

  /**
   * Get a mask of all unique rows in this DataFrame.
   */
  isUnique(): Series<boolean> {
    return _wrapSeries(this.unwrap('is_unique'));
  }

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
  join(df: DataFrame, options: JoinOptions): DataFrame {
    options =  {...defaultJoinOptions, ...options};
    const on = columnOrColumns(options.on);
    const how = options.how;
    const suffix = options.suffix;

    let leftOn = columnOrColumns(options.leftOn);
    let rightOn = columnOrColumns(options.rightOn);

    if(on) {
      leftOn = on;
      rightOn = on;
    }

    if(!leftOn && !rightOn) {
      throw new RangeError("You should pass the column to join on as an argument.");
    }

    return this.wrap('join', {
      other: df._df,
      on,
      how,
      left_on: leftOn,
      right_on: rightOn,
      suffix,
    });
  }
  /**
   * Get first N rows as DataFrame.
   * @see {@link head}
   */
  limit(length?: number): DataFrame {
    return this.head(length);
  }
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
  max(axis=0): DataFrame | Series<any>{
    if(axis === 0 ) {
      return this.wrap('max');
    }

    if(axis === 1) {
      return _wrapSeries(this.wrap('hmax'));
    }
    throw new RangeError("axis must be 0 or 1");
  }
  /**
   * Aggregate the columns of this DataFrame to their mean value.
   * ___
   *
   * @param axis - either 0 or 1
   * @param nullStrategy - this argument is only used if axis == 1
   */
  mean(axis: 0 | 1, nullStrategy?: 'ignore' | 'propagate'): DataFrame | Series<any>
  mean(axis=0, nullStrategy='ignore'): DataFrame | Series<any> {
    if(axis === 0 ) {
      return this.wrap('mean');
    }

    if(axis === 1) {
      return _wrapSeries(this.unwrap('hmean', {nullStrategy}));
    }
    throw new RangeError("axis must be 0 or 1");
  }
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
   * │ 1   ┆ 1   ┆ null │
   * ╰─────┴─────┴──────╯
   * ```
   */
  median(): DataFrame {
    return this.wrap('median');
  }
  /**
   * Unpivot DataFrame to long format.
   * ___
   *
   * @param idVars - Columns to use as identifier variables.
   * @param valueVars - Values to use as identifier variables.
   */
  melt(idVars: string | Array<string>, valueVars: string | Array<string>): DataFrame {
    return this.wrap('melt', {
      idVars: columnOrColumns(idVars),
      valueVars: columnOrColumns(valueVars)
    });
  }
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
  min(axis=0): DataFrame | Series<any> {
    if(axis === 0 ) {
      return this.wrap('min');
    }

    if(axis === 1) {
      return _wrapSeries(this.wrap('hmin'));
    }
    throw new RangeError("axis must be 0 or 1");
  }
  /**
   * Get number of chunks used by the ChunkedArrays of this DataFrame.
   */
  nChunks(): number{
    return this.unwrap('n_chunks');
  }

  /**
   * Create a new DataFrame that shows the null counts per column.
   * ___
   * @example
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
   */
  nullCount(): DataFrame {
    return this.wrap('null_count');
  }
  peek(): DataFrame {
    console.log(this);

    return this;
  }
  /**
   * Apply a function on Self.
   */
  pipe<T>(func: (df: DataFrame, ...args: any[]) => T, ...args: any[]): T {
    throw todo();
  }
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
  quantile(quantile: number): DataFrame {
    return this.wrap('quantile', {quantile});
  }
  /**
   * __Rechunk the data in this DataFrame to a contiguous allocation.__
   *
   * This will make sure all subsequent operations have optimal and predictable performance.
   */
  rechunk(): DataFrame {
    return this.wrap('rechunk');
  }
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
  rename(mapping: Record<string,string>): DataFrame {
    const df = this.clone();

    Object.entries(mapping).forEach(([key, value]) => {
      this.unwrap('rename', {column: key, new_col: value}, df._df);
    });

    return df;
  }
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
  replaceAtIdx(index: number, newColumn: Series<any>): void {
    this.unwrap('replace_at_idx', {index, newColumn: newColumn._series});
  }
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
  row(index: number): Array<any> {
    throw todo();
  }
  /**
   * Convert columnar data to rows as arrays
   */
  rows(): Array<Array<any>> {
    return this.unwrap('to_rows');
  }
  /**
   * Sample from this DataFrame by setting either `n` or `frac`.
   * @param n - Number of samples < self.len() .
   * @param frac - Fraction between 0.0 and 1.0 .
   * @param withReplacement - Sample with replacement.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>     "foo": [1, 2, 3],
   * >>>     "bar": [6, 7, 8],
   * >>>     "ham": ['a', 'b', 'c']
   * >>>     })
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
  sample({n,frac, withReplacement}: {n?: number, frac?:number, withReplacement?: boolean}): DataFrame
  sample(n: number, frac?:number, withReplacement?: boolean): DataFrame
  sample(options: {n?: number, frac?:number, withReplacement?: boolean} | number, frac?: number, withReplacement=false): DataFrame {
    if (typeof options === 'number') {
      return this.wrap('sample_n', {n: options, withReplacement});
    }

    if(frac) {
      return this.wrap('sample_frac', {frac, withReplacement});
    }

    if(options.n) {
      return this.wrap('sample_n', options);
    }

    if(options.frac) {
      return this.wrap('sample_frac', options);
    }
    else {
      throw new TypeError('invalid arguments');
    }
  }
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
  select(...selection: ColumnSelection[]): DataFrame {
    selection = columnOrColumnsStrict(selection);

    return this.wrap('select', {selection});
  }
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
  shift(periods:number): DataFrame
  /**
   * Shift the values by a given period and fill the parts that will be empty due to this operation
   * with `Nones`.
   * ___
   * @param opt.periods - Number of places to shift (may be negative).
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> })
   * >>> df.shift({periods:1})
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
   * >>> df.shift({periods:-1})
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
  shift({periods}: {periods: number}): DataFrame
  shift(arg: number | {periods: number}) : DataFrame {
    if(typeof arg === 'number') {
      return this.wrap('shift', {periods: arg});
    } else {
      return this.wrap('shift', arg);
    }
  }
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
  shiftAndFill(periods: number, fillValue: number | string): DataFrame // shift_and_fill
  shiftAndFill(opts: {periods: number, fillValue: number | string}): DataFrame
  shiftAndFill(opts: {periods: number, fillValue: number | string} | number, fillValue?: number | string): DataFrame {
    throw todo();
  }
  /**
   * Shrink memory usage of this DataFrame to fit the exact capacity needed to hold the data.
   * @param inPlace - optionally shrink in place
   */
  shrinkToFit(): DataFrame
  shrinkToFit(inPlace?: boolean): DataFrame | void
  shrinkToFit(inPlace=false): DataFrame | void {
    if(inPlace) {
      this.unwrap('shrink_to_fit');
    } else {
      const df = this.clone();
      this.unwrap('shrink_to_fit', {}, df._df);

      return df;
    }
  }
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
  slice(opts: {offset: number, length: number}): DataFrame
  slice(offset: number, length: number): DataFrame
  slice(opts: {offset: number, length: number} | number, length?: number): DataFrame {
    if(typeof opts === 'number') {
      return this.wrap('slice', {offset: opts, length});
    } else {
      return this.wrap('slice', opts);
    }
  }
  /**
   * Sort the DataFrame by column.
   * ___
   * @param by - By which column to sort. Only accepts string.
   * @param reverse - Reverse/descending sort.
   * @param inPlace - Perform operation in-place.
   */
  sort(by: string, reverse?: boolean, inPlace?: boolean): DataFrame
  /**
   * Sort the DataFrame by column.
   * ___
   * @param opts
   * @param opts.by - By which column to sort. Only accepts string.
   * @param opts.reverse - Reverse/descending sort.
   * @param opts.inPlace - Perform operation in-place.
   */
  sort(opts: {by: string, reverse?: boolean, inPlace?: boolean}): DataFrame
  sort(arg: {by: string, reverse?: boolean, inPlace?: boolean} | string,  reverse=false,  inPlace=false): DataFrame {
    if(typeof arg === "string") {
      return this.sort({
        by: arg,
        reverse,
        inPlace
      });
    } else {
      if(inPlace) {
        const df = this.clone();
        this.unwrap('sort_in_place', arg, df._df);

        return df;
      } else {
        return this.wrap('sort', arg);
      }
    }
  }
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
  std(): DataFrame {
    return this.wrap('std');
  }

  /**
   * Aggregate the columns of this DataFrame to their mean value.
   * ___
   *
   * @param axis - either 0 or 1
   * @param nullStrategy - this argument is only used if axis == 1
   */
  sum(axis: 0): DataFrame
  sum(axis: 1): Series<any>
  sum(axis: 1, nullStrategy?: 'ignore' | 'propagate'): Series<any>
  sum(axis: 0 | 1, nullStrategy='ignore'): DataFrame | Series<any> {
    if(axis === 0 ) {
      return this.wrap('sum');
    }

    if(axis === 1) {
      return _wrapSeries(this.wrap('hsum', {nullStrategy}));
    }
    throw new RangeError("axis must be 0 or 1");
  }

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
  tail(length=5): DataFrame {
    return this.wrap('tail', {length});
  }

  /**
   * __Write DataFrame to comma-separated values file (csv).__
   *
   * If no options are specified, it will return a new string containing the contents
   * ___
   * @param options
   * @param options.dest - path to file, or writeable stream
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
   * >>> df.head(1).toCSV({dest: "./foo.csv"})
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
   * >>> df.head(1).toCSV({dest: writeStream, hasHeader: false})
   * writeStream: '1,6,a'
   * ```
   */

  toCSV(): string;
  toCSV(options: WriteCsvOptions): string;
  toCSV(dest: string | Stream): void;
  toCSV(dest: string | Stream, options: WriteCsvOptions): void;
  toCSV(dest?: string | Stream | WriteCsvOptions, options?: WriteCsvOptions): void | string {
    options = { hasHeader:true, sep: ',', ...options};

    if(dest instanceof Stream.Writable) {
      this.unwrap('to_csv', {writeStream: dest, ...options});
    } else if (typeof dest === 'string') {
      const writeStream = fs.createWriteStream(dest);
      this.unwrap('to_csv', {writeStream, ...options});
    } else if (!dest || (dest.constructor.name === 'Object' && !dest['dest'])) {
      let body = '';
      const writeStream = new Stream.Writable({
        write(chunk, _encoding, callback) {
          body += chunk;
          callback(null);
        }
      });
      this.unwrap('to_csv', {writeStream, ...options, ...dest});

      return body;
    }
    throw new TypeError("unknown destination type, Supported types are 'string' and 'Stream.Writeable'");


  }

  toJS(): object {
    return this.unwrap('to_js');
  }

  toJSON(): string
  toJSON(dest: string | Stream): void
  toJSON(dest?: string | Stream): void | string {
    if(dest instanceof Stream.Writable) {
      this.unwrap('write_json', {writeStream: dest});
    } else if (typeof dest === 'string') {
      const writeStream = fs.createWriteStream(dest);
      this.unwrap('write_json', {writeStream});
    } else if (!dest) {
      let body = '';
      const writeStream = new Stream.Writable({
        write(chunk, _encoding, callback) {
          body += chunk;
          callback(null);
        }
      });
      this.unwrap('write_json', {writeStream});

      return body;
    }
    throw new TypeError("unknown destination type, Supported types are 'string' and 'Stream.Writeable'");

  }
  toSeries(index:number): Series<any> {
    return _wrapSeries(this.unwrap('select_at_idx', {index}));
  }

  add(other: Series<any>): DataFrame {
    return this.wrap('add', {other});
  }
  sub(other: Series<any>): DataFrame  {
    return this.wrap('sub', {other});
  }
  div(other: Series<any>): DataFrame {
    return this.wrap('div', {other});
  }
  mul(other: Series<any>): DataFrame {
    return this.wrap('mul', {other});
  }
  rem(other: Series<any>): DataFrame  {
    return this.wrap('rem', {other});
  }
  plus(other: Series<any>): DataFrame {
    return this.wrap('add', {other});
  }
  minus(other: Series<any>): DataFrame  {
    return this.wrap('sub', {other});
  }
  divide(other: Series<any>): DataFrame {
    return this.wrap('div', {other});
  }
  times(other: Series<any>): DataFrame {
    return this.wrap('mul', {other});
  }
  remainder(other: Series<any>): DataFrame  {
    return this.wrap('rem', {other});
  }
  inner(): any {
    return this._df;
  }


  /**
   * Wraps the internal `_df` into the `DataFrame` class
   */
  private wrap<U>(
    method: string,
    args?: object,
    _df = this._df,
  ): DataFrame {
    // console.log("wrap", {method, args});

    return new DataFrame(polarsInternal.df[method]({ _df, ...args }));
  }

  private unwrap<T>(method: string, args?: object, _df = this._df): T {
    // console.log("unwrap", {method, args});

    return polarsInternal.df[method]({ _df, ...args });
  }
}


function obj_to_df(obj: Record<string, Array<any>>, columns?: Array<string>): any {
  const data =  Object.entries(obj).map(([key, value], idx) => {
    return Series(columns?.[idx] ?? key, value)._series;
  });

  return polarsInternal.df.read_columns({columns: data});
}