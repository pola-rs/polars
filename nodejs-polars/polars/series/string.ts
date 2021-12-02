import pli from "../internals/polars_internal";
import {DataType, DtypeToPrimitive} from "../datatypes";
import {JsSeries, Series, _wrapSeries} from "../series";


/**
 * Series.str functions.
 */
export interface StringFunctions {
  /**
   * Parse a Series of dtype Utf8 to a Date/Datetime Series.
   * @param datatype Date or Datetime.
   * @param fmt formatting syntax. [Read more](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html)
   */
  strptime(datatype: DataType.Date , fmt?: string) : Series<Date>
  strptime(datatype: DataType.Datetime , fmt?: string) : Series<number>
  /**
   * Get length of the string values in the Series.
   */
  lengths() : Series<number>
  /**
   * Check if strings in Series contain regex pattern.
   * @param pattern A valid regex pattern
   * @returns Boolean mask
   */
  contains(pattern: RegExp) : Series<boolean>
  /**
   * Extract the first match of json string with provided JSONPath expression.
   * Throw errors if encounter invalid json strings.
   * All return value will be casted to Utf8 regardless of the original value.
   * @see https://goessner.net/articles/JsonPath/
   * @param jsonPath - A valid JSON path query string
   * @returns Utf8 array. Contain null if original value is null or the `jsonPath` return nothing.
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * 'json_val':['{"a":"1"}',None,'{"a":2}', '{"a":2.1}', '{"a":true}'
   * })
   * >>> df.select(pl.col('json_val').str.jsonPathMatch('$.a')
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
  jsonPathMatch(jsonPath: string) : Series<string>
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
   * > df.select(pl.col('a').str.extract(/candidate=(\w+)/, 1))
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
  extract(pattern: RegExp, groupIndex: number) : Series<string>
  /**
   * Replace first regex match with a string value.
   * @param pattern A valid regex pattern
   * @param value Substring to replace.
   */
  replace(pattern: RegExp, value: string) : Series<string>
  /**
   * Replace all regex matches with a string value.
   * @param pattern - A valid regex pattern
   * @param value Substring to replace.
   */
  replaceAll(pattern: RegExp, value: string) : Series<string>
  /**
   * Modify the strings to their lowercase equivalent.
   */
  toLowerCase() : Series<string>
  /**
   * Modify the strings to their uppercase equivalent.
   */
  toUpperCase() : Series<string>
  /**
   * Remove trailing whitespace.
   */
  rstrip() : Series<string>
  /**
   * Remove leading whitespace.
   *
   */
  lstrip() : Series<string>
  /**
   * Create subslices of the string values of a Utf8 Series.
   * @param start - Start of the slice (negative indexing may be used).
   * @param length - Optional length of the slice.
   */
  slice(start: number, length?:number) : Series<string>
}

export const StringFunctions = (_s: JsSeries): StringFunctions => {
  const unwrap = <U>(method: string, args?: object, _series = _s): U => {
    return pli.series.str[method]({_series, ...args });
  };

  const wrap = <U>(method, args?, _series = _s): Series<U> => {
    return _wrapSeries(unwrap(method, args, _series));
  };
  const strptime = (dtype: DataType, fmt?): Series<any> => {
    if (dtype === DataType.Date) {
      return wrap<Date>("parse_date", {fmt});
    } else if (dtype === DataType.Datetime) {
      return wrap<number>("parse_datetime", {fmt});
    } else {
      throw new Error("only \"DataType.Date\" and \"DataType.Datetime\" are supported");
    }
  };

  return {
    strptime,
    lengths: () => wrap("lengths"),
    contains: (pat) => wrap("contains", {pat: pat.source}),
    jsonPathMatch: (jsonPath) => wrap("json_path_match", {jsonPath}),
    extract: (pat, groupIndex=1) => wrap("extract", {pat: pat.source, groupIndex}),
    replace: (pat, val) => wrap("replace", {pat: pat.source, val}),
    replaceAll: (pat, val) => wrap("replace_all", {pat: pat.source, val}),
    toLowerCase: () => wrap("to_lowercase"),
    toUpperCase: () => wrap("to_uppercase"),
    rstrip: () => wrap("replace", {pat: /[ \t]+$/.source, val: ""}),
    lstrip: () => wrap("replace", {pat: /^\s*/.source, val: ""}),
    slice: (start, length) => wrap("slice", {start, length}),
  };
};