import { ReadCsvOptions, ReadJsonOptions } from "./datatypes";
import pli from "./internals/polars_internal";
import {DataFrame, dfWrapper} from "./dataframe";
import { isPath } from "./utils";
import {LazyDataFrame} from "./lazy/dataframe";
import path from "path";
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
 * __Read a CSV file or string into a Dataframe.__
 * ___
 * @param options
 * @param options.file - Path to a file or a file like string. Any valid filepath can be used. Example: `file.csv`.
 *     Any string containing the contents of a csv can also be used
 * @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
 *     If set to `null`, a full table scan will be done (slow).
 * @param options.batchSize - Number of lines to read into the buffer at once. Modify this to change performance.
 * @param options.hasHeader - Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`,
 *     `x` being an enumeration over every column in the dataset.
 * @param options.ignoreErrors -Try to keep reading lines if some lines yield errors.
 * @param options.endRows -After n rows are read from the CSV, it stops reading.
 *     During multi-threaded parsing, an upper bound of `n` rows
 *     cannot be guaranteed.
 * @param options.startRows -Start reading after `startRows` position.
 * @param options.projection -Indices of columns to select. Note that column indices start at zero.
 * @param options.sep -Character to use as delimiter in the file.
 * @param options.columns -Columns to select.
 * @param options.rechunk -Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
 * @param options.encoding -Allowed encodings: `utf8`, `utf8-lossy`. Lossy means that invalid utf8 values are replaced with `�` character.
 * @param options.numThreads -Number of threads to use in csv parsing. Defaults to the number of physical cpu's of your system.
 * @param options.dtype -Overwrite the dtypes during inference.
 * @param options.lowMemory - Reduce memory usage in expense of performance.
 * @param options.commentChar - character that indicates the start of a comment line, for instance '#'.
 * @param options.quotChar -character that is used for csv quoting, default = ''. Set to null to turn special handling and escaping of quotes off.
 * @param options.nullValues - Values to interpret as null values. You can provide a
 *     - `string` -> all values encountered equal to this string will be null
 *     - `Array<string>` -> A null value per column.
 *     - `Record<string,string>` -> An object or map that maps column name to a null value string.Ex. {"column_1": 0}
 * @param options.parseDates -Whether to attempt to parse dates or not
 * @returns DataFrame
 */
export function readCSV(options: Partial<ReadCsvOptions>): DataFrame
export function readCSV(path: string): DataFrame
export function readCSV(path: string, options: Partial<ReadCsvOptions>): DataFrame
export function readCSV(arg: Partial<ReadCsvOptions> | string, options?: any) {
  const extensions = [".tsv", ".csv"];
  if(typeof arg === "string") {
    const inline = !isPath(arg, extensions);
    const file = inline || path.isAbsolute(arg) ? arg : path.resolve(process.cwd(), arg);

    return readCSV({...options, file, inline});
  }
  options = {...readCsvDefaultOptions, ...arg};

  return dfWrapper(pli.df.read_csv(options));
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
export function readJSON(options: ReadJsonOptions): DataFrame
export function readJSON(path: string): DataFrame
export function readJSON(path: string, options: ReadJsonOptions): DataFrame
export function readJSON(arg: ReadJsonOptions | string, options?: any) {
  const extensions = [".ndjson", ".json", ".jsonl"];
  if(typeof arg === "string") {
    const inline = !isPath(arg, extensions);
    const file = inline || path.isAbsolute(arg) ? arg : path.resolve(process.cwd(), arg);

    return readJSON({...options, file, inline});
  }
  options = {...readJsonDefaultOptions, ...arg};

  return dfWrapper(pli.df.read_json(options));
}

/**
 * Read into a DataFrame from a csv file.
 */
export function scanCSV(options: Partial<ReadCsvOptions>): LazyDataFrame
export function scanCSV(path: string): LazyDataFrame
export function scanCSV(path: string, options: Partial<ReadCsvOptions>): LazyDataFrame
export function scanCSV(arg: Partial<ReadCsvOptions> | string, options?: any): LazyDataFrame {
  if(typeof arg === "string") {
    return scanCSV({...options, path: arg});
  }
  options = {...readCsvDefaultOptions, ...arg};

  return LazyDataFrame(pli.ldf.scanCSV(options));
}
export function readParquet() {}
export function scanParquet() {}
export function readIPC() {}
export function scanIPC() {}
export function scanJSON() {}
