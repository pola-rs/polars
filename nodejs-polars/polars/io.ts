import { ReadCsvOptions, ReadJsonOptions, ReadParquetOptions } from "./datatypes";
import pli from "./internals/polars_internal";
import {DataFrame, dfWrapper} from "./dataframe";
import { isPath } from "./utils";
import {LazyDataFrame} from "./lazy/dataframe";
import {Readable, Stream} from "stream";
import {concat} from "./functions";

const readCsvDefaultOptions: Partial<ReadCsvOptions> = {
  inferSchemaLength: 50,
  batchSize: 10000,
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
  batchSize: 10000,
  inferSchemaLength: 50
};

function readCSVBuffer(buff, options) {
  return  dfWrapper(pli.df.readCSVBuffer({...readCsvDefaultOptions, ...options, buff}));
}
function readCSVPath(path, options) {
  return  dfWrapper(pli.df.readCSVPath({...readCsvDefaultOptions, ...options, path}));
}
function readJSONBuffer(buff, options) {
  return  dfWrapper(pli.df.readJSONBuffer({...readJsonDefaultOptions, ...options, buff}));
}
function readJSONPath(path, options) {
  return  dfWrapper(pli.df.readJSONPath({...readJsonDefaultOptions, ...options, path}));
}
/**
 * __Read a CSV file or string into a Dataframe.__
 * ___
 * @param pathOrBody - path or buffer or string
 *   - path: Path to a file or a file like string. Any valid filepath can be used. Example: `file.csv`.
 *   - body: String or buffer to be read as a CSV
 * @param options
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
export function readCSV(pathOrBody: string | Buffer, options?: Partial<ReadCsvOptions>): DataFrame {
  const extensions = [".tsv", ".csv"];

  if(Buffer.isBuffer(pathOrBody)) {
    return readCSVBuffer(pathOrBody, options);
  }

  if(typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, extensions);
    if(inline) {
      return readCSVBuffer(Buffer.from(pathOrBody, "utf-8"), options);
    } else {
      return readCSVPath(pathOrBody, options);
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}


/**
 * __Read a JSON file or string into a DataFrame.__
 *
 * _Note: Currently only newline delimited JSON is supported_
 * @param pathOrBody - path or buffer or string
 *   - path: Path to a file or a file like string. Any valid filepath can be used. Example: `file.csv`.
 *   - body: String or buffer to be read as a CSV
 * @param options
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
 * > const df = pl.readJSON(jsonString)
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
export function readJSON(pathOrBody: string | Buffer, options?: Partial<ReadJsonOptions>): DataFrame {
  const extensions = [".ndjson", ".json", ".jsonl"];
  if(Buffer.isBuffer(pathOrBody)) {
    return readJSONBuffer(pathOrBody, options);
  }

  if(typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, extensions);
    if(inline) {
      return readJSONBuffer(Buffer.from(pathOrBody, "utf-8"), options);
    } else {
      return readJSONPath(pathOrBody, options);
    }
  } else {
    throw new Error("must supply either a path or body");
  }
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

export function readParquet(path: string, options?: ReadParquetOptions): DataFrame {
  return dfWrapper(pli.df.readParquet({path, ...options}));
}

export function scanParquet() {}
export function readIPC() {}
export function scanIPC() {}
export function scanJSON() {}

// utility to read streams as lines.
class LineBatcher extends Stream.Transform {

  #lines: Buffer[];
  #accumulatedLines: number;
  #batchSize: number;

  constructor(options) {
    super(options);
    this.#lines = [];
    this.#accumulatedLines = 0;
    this.#batchSize = options.batchSize;
  }

  _transform(chunk, _encoding, done) {

    var begin = 0;
    for (var i = 0; i < chunk.length; i++) {
      if (chunk[i] === 10) { // '\n'
        this.#accumulatedLines++;
        if (this.#accumulatedLines == this.#batchSize) {
          this.#lines.push(chunk.slice(begin, i + 1));
          this.push(Buffer.concat(this.#lines));
          this.#lines = [];
          this.#accumulatedLines = 0;
          begin = i + 1;
        }
      }
    }

    this.#lines.push(chunk.slice(begin));

    done();
  }
  _flush(done) {
    this.push(Buffer.concat(this.#lines));

    done();
  }
}

/**
 * __Read a stream into a Dataframe.__
 *
 * **Warning:** this is much slower than `scanCSV` or `readCSV`
 *
 * This will consume the entire stream into a single buffer and then call `readCSV`
 * Only use it when you must consume from a stream, or when performance is not a major consideration
 *
 * ___
 * @param stream - readable stream containing csv data
 * @param options
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
 * @returns Promise<DataFrame>
 *
 * @example
 * ```
 * >>> const readStream = new Stream.Readable({read(){}});
 * >>> readStream.push(`a,b\n`);
 * >>> readStream.push(`1,2\n`);
 * >>> readStream.push(`2,2\n`);
 * >>> readStream.push(`3,2\n`);
 * >>> readStream.push(`4,2\n`);
 * >>> readStream.push(null);
 *
 * >>> pl.readCSVStream(readStream).then(df => console.log(df));
 * shape: (4, 2)
 * ┌─────┬─────┐
 * │ a   ┆ b   │
 * │ --- ┆ --- │
 * │ i64 ┆ i64 │
 * ╞═════╪═════╡
 * │ 1   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 2   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 3   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 4   ┆ 2   │
 * └─────┴─────┘
 * ```
 */
export function readCSVStream(stream: Readable, options?: ReadCsvOptions): Promise<DataFrame>  {
  let batchSize = options?.batchSize ?? 10000;
  let count = 0;
  let end = options?.endRows ?? Number.POSITIVE_INFINITY;

  return new Promise((resolve, reject) => {
    const s = stream.pipe(new LineBatcher({batchSize}));
    const chunks: any[] = [];

    s.on("data", (chunk) => {
      // early abort if 'end rows' is specified
      if (count <= end) {
        chunks.push(chunk);
      } else {
        s.end();
      }
      count += batchSize;
    }).on("end", () => {
      try {
        let buff = Buffer.concat(chunks);
        const df = readCSVBuffer(buff, options);
        resolve(df);
      } catch (err) {
        reject(err);
      }
    });
  });
}

/**
 * __Read a newline delimited JSON stream into a DataFrame.__
 *
 * @param stream - readable stream containing json data
 * @param options
 * @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
 *    If set to `null`, a full table scan will be done (slow).
 *    Note: this is done per batch
 * @param options.batchSize - Number of lines to read into the buffer at once. Modify this to change performance.
 * @example
 * ```
 * >>> const readStream = new Stream.Readable({read(){}});
 * >>> readStream.push(`${JSON.stringify({a: 1, b: 2})} \n`);
 * >>> readStream.push(`${JSON.stringify({a: 2, b: 2})} \n`);
 * >>> readStream.push(`${JSON.stringify({a: 3, b: 2})} \n`);
 * >>> readStream.push(`${JSON.stringify({a: 4, b: 2})} \n`);
 * >>> readStream.push(null);
 *
 * >>> pl.readJSONStream(readStream).then(df => console.log(df));
 * shape: (4, 2)
 * ┌─────┬─────┐
 * │ a   ┆ b   │
 * │ --- ┆ --- │
 * │ i64 ┆ i64 │
 * ╞═════╪═════╡
 * │ 1   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 2   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 3   ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 4   ┆ 2   │
 * └─────┴─────┘
 * ```
 */
export function readJSONStream(stream: Readable, options?: ReadJsonOptions): Promise<DataFrame>  {
  let batchSize = options?.batchSize ?? 10000;

  return new Promise((resolve, reject) => {
    const chunks: any[] = [];

    stream
      .pipe(new LineBatcher({batchSize}))
      .on("data", (chunk) => {
        try {
          const df = readJSONBuffer(chunk, options);
          chunks.push(df);
        } catch(err) {
          reject(err);
        }
      })
      .on("end", () => {
        try {
          const df = concat(chunks);
          resolve(df);
        } catch (err) {
          reject(err);
        }
      });

  });
}
