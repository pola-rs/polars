import polars_internal from "../internals/polars_internal";

export type Option<T> = T | undefined;

export interface Expr {
  _and(expr: Expr) : Expr
  _or(expr: Expr) : Expr
  _xor(expr: Expr) : Expr
  abs() : Expr
  aggGroups() : Expr
  alias(name: string) : Expr
  argMax() : Expr
  argMin() : Expr
  argSort(reverse: boolean) : Expr
  argUnique() : Expr
  arrLengths() : Expr
  backwardFill() : Expr
  cast(dataType: any, strict: boolean) : Expr
  count() : Expr
  cummax(reverse: boolean) : Expr
  cummin(reverse: boolean) : Expr
  cumprod(reverse: boolean) : Expr
  cumsum(reverse: boolean) : Expr
  day() : Expr
  dot(other: Expr) : Expr
  eq(other: Expr) : Expr
  exclude(columns: Array<String>) : Expr
  explode() : Expr
  fillNan(expr: Expr) : Expr
  fillNull(expr: Expr) : Expr
  fillNullWithStrategy(strategy: string) : Expr
  filter(predicate: Expr) : Expr
  first() : Expr
  floor() : Expr
  forwardFill() : Expr
  gt(other: Expr) : Expr
  gtEq(other: Expr) : Expr
  hash(k0: number, k1: number, k2: number, k3: number) : Expr
  head(n: Option<number>) : Expr
  hour() : Expr
  interpolate() : Expr
  isDuplicated() : Expr
  isFinite() : Expr
  isFirst() : Expr
  isIn(expr: Expr) : Expr
  isInfinite() : Expr
  isNot() : Expr
  isNotNull() : Expr
  isNull() : Expr
  isUnique() : Expr
  keepName() : Expr
  last() : Expr
  list() : Expr
  lt(other: Expr) : Expr
  ltEq(other: Expr) : Expr
  map(func: (...args: any[]) => any, outputType: any, aggList: boolean) : Expr
  max() : Expr
  mean() : Expr
  median() : Expr
  min() : Expr
  minute() : Expr
  mode() : Expr
  month() : Expr
  nanosecond() : Expr
  neq(other: Expr) : Expr
  nUnique() : Expr
  ordinalDay() : Expr
  over(partitionBy: Array<Expr>) : Expr
  pow(exponent: number) : Expr
  prefix(prefix: string) : Expr
  quantile(quantile: number) : Expr
  reinterpret(signed: boolean) : Expr
  repeatBy(by: Expr) : Expr
  reverse() : Expr
  rollingApply(windowSize: number, func: (...args: any[]) => any) : Expr
  round(decimals: number) : Expr
  second() : Expr
  shift(periods: number) : Expr
  shiftAndFill(periods: number, fillValue: Expr) : Expr
  slice(offset: number, length: number) : Expr
  sort(reverse: boolean) : Expr
  sortBy(by: Array<Expr>, reverse: Array<boolean>) : Expr
  std() : Expr
  strContains(pat: String) : Expr
  strExtract(pat: String, groupIndex: number) : Expr
  strftime(fmt: String) : Expr
  strJsonPathMatch(pat: String) : Expr
  strLengths() : Expr
  strParseDate(fmt: Option<String>) : Expr
  strParseDatetime(fmt: Option<String>) : Expr
  strReplace(pat: String, val: String) : Expr
  strReplaceAll(pat: String, val: String) : Expr
  strSlice(start: number, length: Option<number>) : Expr
  strToLowercase() : Expr
  strToUppercase() : Expr
  suffix(suffix: string) : Expr
  sum() : Expr
  tail(n: Option<number>) : Expr
  take(idx: Expr) : Expr
  takeEvery(n: number) : Expr
  timestamp() : Expr
  unique() : Expr
  var() : Expr
  week() : Expr
  weekday() : Expr
  year() : Expr
}


export function col(name: string) {
  return polars_internal.col({name});
}