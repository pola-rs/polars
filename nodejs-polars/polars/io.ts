import {DataType} from "./datatypes";
import pli from "./internals/polars_internal";
import {DataFrame, _DataFrame} from "./dataframe";
import {isPath} from "./utils";
import {LazyDataFrame, _LazyDataFrame} from "./lazy/dataframe";
import {Readable, Stream} from "stream";
import {concat} from "./functions";


const readCsvDefaultOptions = {
  inferSchemaLength: 100,
  hasHeader: true,
  ignoreErrors: true,
  chunkSize: 10000,
  skipRows: 0,
  sep: ",",
  rechunk: false,
  encoding: "utf8",
  lowMemory: false,
  parseDates: false,
  skipRowsAfterHeader: 0
};

const readJsonDefaultOptions = {
  batchSize: 10000,
  inferSchemaLength: 50,
  format: "lines"
};


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
    var position = 0;

    let i = 0;
    while (i < chunk.length) {
      if (chunk[i] === 10) { // '\n'
        this.#accumulatedLines++;
        if (this.#accumulatedLines == this.#batchSize) {
          this.#lines.push(chunk.subarray(begin, i + 1));
          this.push(Buffer.concat(this.#lines));
          this.#lines = [];
          this.#accumulatedLines = 0;
          begin = i + 1;
        }
      }
      i++;
    }

    this.#lines.push(chunk.subarray(begin));

    done();
  }
  _flush(done) {
    this.push(Buffer.concat(this.#lines));

    done();
  }
}

// helper functions

function readCSVBuffer(buff, options) {
  return _DataFrame(pli.readCsv(buff, {...readCsvDefaultOptions, ...options}));
}

export function readRecords(records: Record<string, any>[], options?: {schema: Record<string, DataType>}): DataFrame;
export function readRecords(records: Record<string, any>[], options?: {inferSchemaLength?: number}): DataFrame;
export function readRecords(records: Record<string, any>[], options): DataFrame {

  if (options?.schema) {
    return _DataFrame(pli.fromRows(records, options.schema));
  } else {
    return _DataFrame(pli.fromRows(records, undefined, options?.inferSchemaLength));
  }
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
export function readCSV(pathOrBody: string | Buffer, options?: any): DataFrame;
export function readCSV(pathOrBody, options?) {
  const extensions = [".tsv", ".csv"];

  if (Buffer.isBuffer(pathOrBody)) {
    return _DataFrame(pli.readCsv(pathOrBody, {...readCsvDefaultOptions, ...options}));
  }
  if (typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, extensions);
    if (inline) {
      const buf = Buffer.from(pathOrBody, "utf-8");

      return _DataFrame(pli.readCsv(buf, {...readCsvDefaultOptions, ...options}));
    } else {
      return _DataFrame(pli.readCsv(pathOrBody, {...readCsvDefaultOptions, ...options}));
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}


const scanCsvDefaultOptions = {
  inferSchemaLength: 100,
  cache: true,
  hasHeader: true,
  ignoreErrors: true,
  skipRows: 0,
  sep: ",",
  rechunk: false,
  encoding: "utf8",
  lowMemory: false,
  parseDates: false,
  skipRowsAfterHeader: 0
};

/**
   * __Lazily read from a CSV file or multiple files via glob patterns.__
   *
   * This allows the query optimizer to push down predicates and
   * projections to the scan level, thereby potentially reducing
   * memory overhead.
   * ___
   * @param path path to a file
   * @param options.hasHeader - Indicate if first row of dataset is header or not. If set to False first row will be set to `column_x`,
   *     `x` being an enumeration over every column in the dataset.
   * @param options.sep -Character to use as delimiter in the file.
   * @param options.commentChar - character that indicates the start of a comment line, for instance '#'.
   * @param options.quoteChar -character that is used for csv quoting, default = ''. Set to null to turn special handling and escaping of quotes off.
   * @param options.skipRows -Start reading after `skipRows` position.
   * @param options.nullValues - Values to interpret as null values. You can provide a
   *     - `string` -> all values encountered equal to this string will be null
   *     - `Array<string>` -> A null value per column.
   *     - `Record<string,string>` -> An object or map that maps column name to a null value string.Ex. {"column_1": 0}
   * @param options.ignoreErrors -Try to keep reading lines if some lines yield errors.
   * @param options.cache Cache the result after reading.
   * @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
   *     If set to `null`, a full table scan will be done (slow).
   * @param options.n_rows -After n rows are read from the CSV, it stops reading.
   *     During multi-threaded parsing, an upper bound of `n` rows
   *     cannot be guaranteed.
   * @param options.rechunk -Make sure that all columns are contiguous in memory by aggregating the chunks into a single array.
   * @param options.lowMemory - Reduce memory usage in expense of performance.
   * ___
   *
   */
export function scanCSV(path: string, options?: any): LazyDataFrame
export function scanCSV(path, options?) {
  options = {...scanCsvDefaultOptions, ...options};

  return _LazyDataFrame(pli.scanCsv(path, options));
}
/**
   * __Read a JSON file or string into a DataFrame.__
   *
   * @param pathOrBody - path or buffer or string
   *   - path: Path to a file or a file like string. Any valid filepath can be used. Example: `file.csv`.
   *   - body: String or buffer to be read as a CSV
   * @param options
   * @param options.inferSchemaLength -Maximum number of lines to read to infer schema. If set to 0, all columns will be read as pl.Utf8.
   *    If set to `null`, a full table scan will be done (slow).
   * @param options.jsonFormat - Either "lines" or "json"
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
export function readJSON(pathOrBody: string | Buffer, options?: any): DataFrame
export function readJSON(pathOrBody, options = readJsonDefaultOptions) {
  options = {...readJsonDefaultOptions, ...options};
  let method = options.format === "lines" ? pli.readJsonLines : pli.readJson;
  const extensions = [".ndjson", ".json", ".jsonl"];
  if (Buffer.isBuffer(pathOrBody)) {
    return _DataFrame(pli.readJson(pathOrBody, options));
  }

  if (typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, extensions);
    if (inline) {
      return _DataFrame(method(Buffer.from(pathOrBody, "utf-8"), options));
    } else {
      return _DataFrame(method(pathOrBody, options));
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}
/**
   * __Read a JSON file or string into a DataFrame.__
   *
   * _Note: Currently only newline delimited JSON is supported_
   * @param path - path to json file
   *   - path: Path to a file or a file like string. Any valid filepath can be used. Example: `./file.json`.
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
   * > const df = pl.scanJson(jsonString).collectSync()
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
export function scanJson(path: string, options?: Partial<{inferSchemaLength: number, batchSize: number}>): LazyDataFrame
export function scanJson(path, options?) {
  options = {...readJsonDefaultOptions, ...options};

  return _LazyDataFrame(pli.scanJson(path, options));
}


interface ReadParquetOptions {
  columns?: string[] | number[],
  numRows?: number,
  parallel?: "auto" | "columns" | "row_groups" | "none",
  rowCount?: RowCount
}


/**
   * Read into a DataFrame from a parquet file.
   * @param pathOrBuffer
   * Path to a file, list of files, or a file like object. If the path is a directory, that directory will be used
   * as partition aware scan.
   * @param options.columns Columns to select. Accepts a list of column indices (starting at zero) or a list of column names.
   * @param options.numRows  Stop reading from parquet file after reading ``numRows``.
   * @param options.parallel
   *    Any of  'auto' | 'columns' |  'row_groups' | 'none'
        This determines the direction of parallelism. 'auto' will try to determine the optimal direction.
        Defaults to 'auto'
   * @param options.rowCount Add row count as column
   */
export function readParquet(pathOrBody: string | Buffer, options?: ReadParquetOptions): DataFrame {
  const pliOptions: any = {};

  if (typeof options?.columns?.[0] === "number") {
    pliOptions.projection = options?.columns;
  } else {
    pliOptions.columns = options?.columns;
  }

  pliOptions.nRows = options?.numRows;
  pliOptions.rowCount = options?.rowCount;
  const parallel = options?.parallel ?? "auto";

  if (Buffer.isBuffer(pathOrBody)) {
    return _DataFrame(pli.readParquet(pathOrBody, pliOptions, parallel));
  }

  if (typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, [".parquet"]);
    if (inline) {
      return _DataFrame(pli.readParquet(Buffer.from(pathOrBody), pliOptions, parallel));
    } else {
      return _DataFrame(pli.readParquet(pathOrBody, pliOptions, parallel));
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}

/**
 * Read into a DataFrame from an avro file.
 * @param pathOrBuffer
 * Path to a file, list of files, or a file like object. If the path is a directory, that directory will be used
 * as partition aware scan.
 * @param options.columns Columns to select. Accepts a list of column names.
 * @param options.projection -Indices of columns to select. Note that column indices start at zero.
 * @param options.nRows  Stop reading from avro file after reading ``n_rows``.
 */
export function readAvro(pathOrBody: string | Buffer, options?: any): DataFrame
export function readAvro(pathOrBody, options = {}) {
  if (Buffer.isBuffer(pathOrBody)) {
    return _DataFrame(pli.readAvro(pathOrBody, options));
  }

  if (typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, [".avro"]);
    if (inline) {
      return _DataFrame(pli.readAvro(Buffer.from(pathOrBody), options));
    } else {
      return _DataFrame(pli.readAvro(pathOrBody, options));
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}


interface RowCount {
  name: string;
  offset: string;
}

interface ScanParquetOptions {
  columns?: string[] | number[],
  numRows?: number,
  parallel?: "auto" | "columns" | "row_groups" | "none",
  rowCount?: RowCount,
  cache?: boolean,
  rechunk?: boolean
}

/**
   * __Lazily read from a parquet file or multiple files via glob patterns.__
   * ___
   * This allows the query optimizer to push down predicates and projections to the scan level,
   * thereby potentially reducing memory overhead.
   * @param path Path to a file or or glob pattern
   * @param options.numRows Stop reading from parquet file after reading ``numRows``.
   * @param options.cache Cache the result after reading.
   * @param options.parallel Read the parquet file in parallel. The single threaded reader consumes less memory.
   * @param options.rechunk In case of reading multiple files via a glob pattern rechunk the final DataFrame into contiguous memory chunks.
   */
export function scanParquet(path: string, options: ScanParquetOptions = {}) {
  const pliOptions: any = {};

  pliOptions.nRows = options?.numRows;
  pliOptions.rowCount = options?.rowCount;
  const parallel = options?.parallel ?? "auto";

  return _LazyDataFrame(pli.scanParquet(path, pliOptions, parallel));
}

/**
   * __Read into a DataFrame from Arrow IPC (Feather v2) file.__
   * ___
   * @param pathOrBody - path or buffer or string
   *   - path: Path to a file or a file like string. Any valid filepath can be used. Example: `file.ipc`.
   *   - body: String or buffer to be read as Arrow IPC
   * @param options.columns Columns to select. Accepts a list of column names.
   * @param options.numRows Stop reading from parquet file after reading ``n_rows``.
   */
export function readIPC(pathOrBody: string | Buffer, options?: any): DataFrame
export function readIPC(pathOrBody, options = {}) {
  if (Buffer.isBuffer(pathOrBody)) {
    return _DataFrame(pli.readIpc(pathOrBody, options));
  }

  if (typeof pathOrBody === "string") {
    const inline = !isPath(pathOrBody, [".ipc"]);
    if (inline) {
      return _DataFrame(pli.readIpc(Buffer.from(pathOrBody, "utf-8"), options));
    } else {
      return _DataFrame(pli.readIpc(pathOrBody, options));
    }
  } else {
    throw new Error("must supply either a path or body");
  }
}

/**
   * __Lazily read from an Arrow IPC (Feather v2) file or multiple files via glob patterns.__
   * ___
   * @param path Path to a IPC file.
   * @param options.numRows Stop reading from IPC file after reading ``numRows``
   * @param options.cache Cache the result after reading.
   * @param options.rechunk Reallocate to contiguous memory when all chunks/ files are parsed.
   */
export function scanIPC(path: string, options?: any): LazyDataFrame
export function scanIPC(path, options = {}) {

  return _LazyDataFrame(pli.scanIpc(path, options));
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
export function readCSVStream(stream: Readable, options?: any): Promise<DataFrame>
export function readCSVStream(stream, options?) {
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
export function readJSONStream(stream: Readable, options?: any): Promise<DataFrame>
export function readJSONStream(stream, options = readJsonDefaultOptions) {
  options = {...readJsonDefaultOptions, ...options};

  return new Promise((resolve, reject) => {
    const chunks: any[] = [];

    stream
      .pipe(new LineBatcher({batchSize: options.batchSize}))
      .on("data", (chunk) => {
        try {
          const df = _DataFrame(pli.readJson(chunk, options));
          chunks.push(df);
        } catch (err) {
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
