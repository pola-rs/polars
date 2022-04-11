import pli from "../internals/polars_internal";
import {DataType} from "../datatypes";
import {JsSeries, Series, seriesWrapper} from "./series";
import {regexToString} from "../utils";
import {col} from "../lazy/functions";


/**
 * namespace containing series string functions
 */
export interface StringFunctions {
  /**
   * Vertically concat the values in the Series to a single string value.
   * @example
   * ```
   * >>> pl.Series([1, null, 2]).str.concat("-")[0]
   * '1-null-2'
   * ```
   */
  concat(delimiter: string): Series<string>
  /**
   * Check if strings in Series contain regex pattern.
   * @param pattern A valid regex pattern
   * @returns Boolean mask
   */
  contains(pattern: string | RegExp): Series<boolean>
  /**
   * Decodes a value using the provided encoding
   * @param encoding - hex | base64
   * @param strict - how to handle invalid inputs
   *
   *     - true: method will throw error if unable to decode a value
   *     - false: unhandled values will be replaced with `null`
   * @example
   * ```
   * s = pl.Series("strings", ["666f6f", "626172", null])
   * s.str.decode("hex")
   * shape: (3,)
   * Series: 'strings' [str]
   * [
   *     "foo",
   *     "bar",
   *     null
   * ]
   * ```
   */
  decode(encoding: "hex" | "base64", strict?: boolean): Series<string>
  decode(options: {encoding: "hex" | "base64", strict?: boolean}): Series<string>
  /**
   * Encodes a value using the provided encoding
   * @param encoding - hex | base64
   * @example
   * ```
   * s = pl.Series("strings", ["foo", "bar", null])
   * s.str.encode("hex")
   * shape: (3,)
   * Series: 'strings' [str]
   * [
   *     "666f6f",
   *     "626172",
   *     null
   * ]
   * ```
   */
  encode(encoding: "hex" | "base64"): Series<string>
  /**
   * Extract the target capture group from provided patterns.
   * @param pattern A valid regex pattern
   * @param groupIndex Index of the targeted capture group.
   * Group 0 mean the whole pattern, first group begin at index 1
   * Default to the first capture group
   * @returns Utf8 array. Contain null if original value is null or regex capture nothing.
   * @example
   * ```
   * > df = pl.DataFrame({
   * ...   'a': [
   * ...       'http://vote.com/ballon_dor?candidate=messi&ref=polars',
   * ...       'http://vote.com/ballon_dor?candidat=jorginho&ref=polars',
   * ...       'http://vote.com/ballon_dor?candidate=ronaldo&ref=polars'
   * ...   ]})
   * > df.getColumn("a").str.extract(/candidate=(\w+)/, 1)
   * shape: (3, 1)
   * ┌─────────┐
   * │ a       │
   * │ ---     │
   * │ str     │
   * ╞═════════╡
   * │ messi   │
   * ├╌╌╌╌╌╌╌╌╌┤
   * │ null    │
   * ├╌╌╌╌╌╌╌╌╌┤
   * │ ronaldo │
   * └─────────┘
   * ```
   */
  extract(pattern: string | RegExp, groupIndex: number): Series<string>
  /**
   * Extract the first match of json string with provided JSONPath expression.
   * Throw errors if encounter invalid json strings.
   * All return value will be casted to Utf8 regardless of the original value.
   * @see https://goessner.net/articles/JsonPath/
   * @param jsonPath - A valid JSON path query string
   * @returns Utf8 array. Contain null if original value is null or the `jsonPath` return nothing.
   * @example
   * ```
   * >>> s = pl.Series('json_val', [
   * ...   '{"a":"1"}',
   * ...   null,
   * ...   '{"a":2}',
   * ...   '{"a":2.1}',
   * ...   '{"a":true}'
   * ... ])
   * >>> s.str.jsonPathMatch('$.a')
   * shape: (5,)
   * Series: 'json_val' [str]
   * [
   *     "1"
   *     null
   *     "2"
   *     "2.1"
   *     "true"
   * ]
   * ```
   */
  jsonPathMatch(jsonPath: string): Series<string>
  /**  Get length of the string values in the Series. */
  lengths(): Series<number>
  /** Remove leading whitespace. */
  lstrip(): Series<string>
  /**
   * Replace first regex match with a string value.
   * @param pattern A valid regex pattern
   * @param value Substring to replace.
   */
  replace(pattern: string | RegExp, value: string): Series<string>
  /**
   * Replace all regex matches with a string value.
   * @param pattern - A valid regex pattern
   * @param value Substring to replace.
   */
  replaceAll(pattern: string | RegExp, value: string): Series<string>
  /** Modify the strings to their lowercase equivalent. */
  toLowerCase(): Series<string>
  /** Modify the strings to their uppercase equivalent. */
  toUpperCase(): Series<string>
  /** Remove trailing whitespace. */
  rstrip(): Series<string>
  /** Remove leading and trailing whitespace. */
  strip(): Series<string>
  /**
   * Create subslices of the string values of a Utf8 Series.
   * @param start - Start of the slice (negative indexing may be used).
   * @param length - Optional length of the slice.
   */
  slice(start: number, length?: number): Series<string>
  /**
   * Split a string into substrings using the specified separator.
   * The return type will by of type List<Utf8>
   * @param separator — A string that identifies character or characters to use in separating the string.
   * @param inclusive Include the split character/string in the results
   */
  split(separator: string, options?: {inclusive?: boolean} | boolean): Series<Series<string>>
  /**
   * Parse a Series of dtype Utf8 to a Date/Datetime Series.
   * @param datatype Date or Datetime.
   * @param fmt formatting syntax. [Read more](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html)
   */
  strftime(datatype: DataType.Date, fmt?: string): Series<Date>
  strftime(datatype: DataType.Datetime, fmt?: string): Series<Date>
}

export const StringFunctions = (_s: JsSeries): StringFunctions => {
  const wrap = (method, args?, _series = _s): any => {
    return seriesWrapper(pli.series.str[method]({_series, ...args }));
  };
  const callExpr = (method) => (...args) => {
    const s = seriesWrapper(_s);

    return s
      .toFrame()
      .select(
        col(s.name)
          .lst[method](...args)
          .as(s.name)
      )
      .getColumn(s.name);
  };

  const handleDecode = (encoding, strict) => {
    switch (encoding) {
    case "hex":
      return wrap(`decodeHex`, {strict});
    case "base64":
      return wrap(`decodeBase64`, {strict});
    default:
      throw new RangeError("supported encodings are 'hex' and 'base64'");
    }
  };

  return {
    concat(delimiter: string) {
      const s = seriesWrapper(_s);

      return s
        .toFrame()
        .select(
          col(s.name)
            .str
            .concat(delimiter)
            .as(s.name)
        )
        .getColumn(s.name);
    },
    contains(pat: string | RegExp) {
      return wrap("contains", {pat: regexToString(pat)});
    },
    decode(arg, strict=false) {
      if(typeof arg === "string") {
        return handleDecode(arg, strict);
      }

      return handleDecode(arg.encoding, arg.strict);
    },
    encode(encoding) {
      switch (encoding) {
      case "hex":
        return wrap(`encodeHex`);
      case "base64":
        return wrap(`encodeBase64`);
      default:
        throw new RangeError("supported encodings are 'hex' and 'base64'");
      }
    },
    extract(pat: string | RegExp, groupIndex: number) {
      return wrap("extract", {pat: regexToString(pat), groupIndex});
    },
    jsonPathMatch(pat: string) {
      return wrap("jsonPathMatch", {pat});
    },
    lengths() {
      return wrap("lengths");
    },
    lstrip() {
      return wrap("replace", {pat: /^\s*/.source, val: ""});
    },
    replace(pat: RegExp, val: string) {
      return wrap("replace", {pat: regexToString(pat), val});
    },
    replaceAll(pat: RegExp, val: string) {
      return wrap("replaceAll", {pat: regexToString(pat), val});
    },
    rstrip() {
      return wrap("replace", {pat: /[ \t]+$/.source, val: ""});
    },
    slice(start: number, length?: number) {
      return wrap("slice", {start, length});
    },
    split(by: string, options?) {
      const inclusive = typeof options === "boolean" ? options : options?.inclusive;

      const s = seriesWrapper(_s);

      return s
        .toFrame()
        .select(
          col(s.name)
            .str
            .split(by, inclusive)
            .as(s.name)
        )
        .getColumn(s.name);


    },
    // splitExact(by: string, options?, inclusive?) {
    //   const n = typeof options === "boolean" ? options : options?.n;
    //   inclusive = typeof inclusive === "boolean" ? inclusive : options.inclusive;
    //   const s = seriesWrapper(_s);

    //   return s
    //     .toFrame()
    //     .select(
    //       col(s.name)
    //         .str
    //         .splitExact(by, n, inclusive)
    //         .as(s.name)
    //     )
    //     .getColumn(s.name);
    // },
    strip() {
      const s = seriesWrapper(_s);

      return s
        .toFrame()
        .select(
          col(s.name)
            .str
            .strip()
            .as(s.name)
        )
        .getColumn(s.name);

    },
    strftime(dtype, fmt?) {
      if (dtype === DataType.Date) {
        return wrap("parseDate", {fmt});
      } else if (dtype === DataType.Datetime) {
        return wrap("parseDateTime", {fmt});
      } else {
        throw new Error(`only "DataType.Date" and "DataType.Datetime" are supported`);
      }
    },
    toLowerCase() {
      return wrap("toLowerCase");
    },
    toUpperCase() {
      return wrap("toUpperCase");
    },
  };
};
