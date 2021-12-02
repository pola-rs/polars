import {DataType} from "../datatypes";
import pli from "../internals/polars_internal";
import {col, lit} from "../functions";
import {ColumnSelection, ExpressionSelection, isExpr} from "../utils";

type JsExpr = any;
type ColumnsOrExpr = ColumnSelection | ExpressionSelection

export interface ExprListFunctions {
  lengths(): Expr
  max(): Expr
  mean(): Expr
  min(): Expr
  reverse(): Expr
  sort(reverse?: boolean): Expr
  sum(): Expr
  unique(): Expr
}

export interface ExprStringFunctions {
  concat(delimiter: string): Expr
  contains(pat: RegExp): Expr
  extract(pat: RegExp, groupIndex: number): Expr
  jsonPathMatch(pat: string): Expr
  lengths(): Expr
  parseDate(fmt?: string): Expr
  parseDateTime(fmt?: string): Expr
  replace(pat: RegExp, val: string): Expr
  replaceAll(pat: RegExp, val: string): Expr
  toLowercase(): Expr
  toUppercase(): Expr
  slice(start: number, length?: number): Expr
}

interface ExprDateTimeFunctions {
  day(): Expr
  hour(): Expr
  minute(): Expr
  month(): Expr
  nanosecond(): Expr
  ordinalDay(): Expr
  second(): Expr
  strftime(fmt: string): Expr
  timestamp(): Expr
  week(): Expr
  weekday(): Expr
  year(): Expr
}

export interface Expr {
  _expr: any
  get date(): ExprDateTimeFunctions
  get str(): ExprStringFunctions
  get lst(): ExprListFunctions
  abs(): Expr
  aggGroups(): Expr
  argMax(): Expr
  argMin(): Expr
  argUnique(): Expr
  backwardFill(): Expr
  count(): Expr
  explode(): Expr
  first(): Expr
  floor(): Expr
  forwardFill(): Expr
  interpolate(): Expr
  isDuplicated(): Expr
  isFinite(): Expr
  isFirst(): Expr
  isInfinite(): Expr
  isNan(): Expr
  isNotNan(): Expr
  isNotNull(): Expr
  isNull(): Expr
  isUnique(): Expr
  keepName(): Expr
  last(): Expr
  list(): Expr
  lowerBound(): Expr
  max(): Expr
  mean(): Expr
  median(): Expr
  min(): Expr
  mode(): Expr
  not(): Expr
  nUnique(): Expr
  reverse(): Expr
  std(): Expr
  sum(): Expr
  unique(): Expr
  upperBound(): Expr
  var(): Expr
  and(other: Expr): Expr
  dot(other: Expr): Expr
  eq(other: Expr): Expr
  fillNan(other: Expr): Expr
  fillNull(other: Expr): Expr
  filter(other: Expr): Expr
  gt(other: Expr): Expr
  gtEq(other: Expr): Expr
  isIn(other: Expr): Expr
  lt(other: Expr): Expr
  ltEq(other: Expr): Expr
  neq(other: Expr): Expr
  or(other: Expr): Expr
  repeatBy(other: Expr): Expr
  take(other: Expr): Expr
  xor(other: Expr): Expr
  argSort(reverse?: boolean): Expr
  argSort({reverse}: {reverse: boolean}): Expr
  cumCount(reverse?: boolean): Expr
  cumCount({reverse}: {reverse: boolean}): Expr
  cumMax(reverse?: boolean): Expr
  cumMax({reverse}: {reverse: boolean}): Expr
  cumMin(reverse?: boolean): Expr
  cumMin({reverse}: {reverse: boolean}): Expr
  cumProd(reverse?: boolean): Expr
  cumProd({reverse}: {reverse: boolean}): Expr
  cumSum(reverse?: boolean): Expr
  cumSum({reverse}: {reverse: boolean}): Expr
  sort(reverse?: boolean): Expr
  sort({reverse}: {reverse: boolean}): Expr
  reinterpret(signed?: boolean): Expr
  reinterpret({signed}: {signed: boolean}): Expr
  skew(bias?: boolean): Expr
  skew({bias}: {bias: boolean}): Expr
  head(length?: number): Expr
  head({length}: {length: number}): Expr
  tail(length?: number): Expr
  tail({length}: {length: number}): Expr
  round(decimals?: number): Expr
  round({decimals}: {decimals: number}): Expr
  rollingMedian(windowSize: number): Expr
  rollingMedian({windowSize}: {windowSize: number}): Expr
  pow(exponent: number): Expr
  pow({exponent}: {exponent: number}): Expr
  quantile(quantile: number): Expr
  quantile({quantile}: {quantile: number}): Expr
  shift(periods: number): Expr
  shift({periods}: {periods: number}): Expr
  suffix(suffix: string): Expr
  suffix({suffix}: {suffix: string}): Expr
  alias(name: string): Expr
  alias({name}: {name: string}): Expr
  prefix(prefix: string): Expr
  prefix({prefix}: {prefix: string}): Expr
  rank(method: string): Expr
  rank({method}: {method: string}): Expr
  fillNullWithStrategy(strategy: string): Expr
  fillNullWithStrategy({strategy}: {strategy: string}): Expr
  takeEvery(n: number): Expr
  takeEvery({n}: {n: number}): Expr
  exclude(columns: string[]): Expr
  exclude({columns}: {columns: string[]}): Expr
  over(...partitionBy: ColumnsOrExpr[]): Expr
  over({partitionBy}: {partitionBy: Expr[]}): Expr
  reshape(dims: number[]): Expr
  reshape({dims}: {dims: number[]}): Expr
  slice(offset:number, length:number): Expr
  diff(n:number, nullBehavior: string): Expr
  rollingQuantile(windowSize: number, quantile: number): Expr
  shiftAndFill(periods: number, fillValue: Expr): Expr
  cast(dtype:DataType, strict?: boolean): Expr
  rollingSkew(windowSize: number, bias?: boolean): Expr
  kurtosis(fisher?: boolean, bias?: boolean): Expr
  hash(k0?: number, k1?: number, k2?: number, k3?: number): Expr


}

const ExprListFunctions = (_expr: JsExpr): ExprListFunctions => {
  const wrap = <U>(method, args?): Expr => {
    return Expr(pli.expr.lst[method]({_expr, ...args }));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);

  return {
    lengths: wrapNullArgs("lengths"),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    min: wrapNullArgs("min"),
    reverse: wrapNullArgs("reverse"),
    sort: (reverse=false) => wrap("sort", {reverse: reverse?.["reverse"] ?? reverse}),
    sum: wrapNullArgs("sum"),
    unique: wrapNullArgs("unique"),
  };
};
const ExprDateTimeFunctions = (_expr: JsExpr): ExprDateTimeFunctions => {
  const wrap = <U>(method, args?): Expr => {

    return Expr(pli.expr.date[method]({_expr, ...args }));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);

  return {
    day: wrapNullArgs("day"),
    hour: wrapNullArgs("hour"),
    minute: wrapNullArgs("minute"),
    month: wrapNullArgs("month"),
    nanosecond: wrapNullArgs("nanosecond"),
    ordinalDay: wrapNullArgs("ordinalDay"),
    second: wrapNullArgs("second"),
    strftime: (fmt) => wrap("strftime", {fmt}),
    timestamp: wrapNullArgs("timestamp"),
    week: wrapNullArgs("week"),
    weekday: wrapNullArgs("weekday"),
    year: wrapNullArgs("year"),
  };
};

export const Expr = (_expr: JsExpr): Expr => {

  const wrap = <U>(method, args?): Expr => {
    console.log({method, args, _expr});

    return Expr(pli.expr[method]({_expr, ...args }));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);
  const wrapExprArg = (method: string) => (other: Expr) => wrap(method, {other: other._expr});
  const wrapUnary = (method: string, key: string) => (val) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapUnaryWithDefault = (method: string, key: string, otherwise) => (val=otherwise) => wrap(method, {[key]: val?.[key] ?? val});
  const wrapBinary = (method: string, key0: string, key1: string) => (val0, val1) => wrap(
    method, {
      [key0]: val0?.[key0] ?? val0,
      [key1]: val1?.[key1] ?? val1
    }
  );
  const kurtosis = (obj?, bias=true) => {
    return wrap("kurtosis", {
      fisher: obj?.["fisher"] ?? (typeof obj === "boolean" ? obj : true),
      bias : obj?.["bias"] ?? bias,
    });
  };

  const hash = (obj?: object | number, k1=1, k2=2, k3=3) => {
    return wrap<bigint>("hash", {
      k0: obj?.["k0"] ?? (typeof obj === "number" ? obj : 0),
      k1: obj?.["k1"] ?? k1,
      k2: obj?.["k2"] ?? k2,
      k3: obj?.["k3"] ?? k3
    });
  };
  const over = (...exprs) => {

    const partitionBy = exprs
      .flat(3)
      .map(e => typeof e === "string"? exprToLitOrExpr(e, false)._expr : e._expr);

    return wrap("over", {partitionBy});
  };

  return {
    _expr,
    get lst() {return ExprListFunctions(_expr);},
    get date() {return ExprDateTimeFunctions(_expr);},
    abs: wrapNullArgs("abs"),
    aggGroups: wrapNullArgs("aggGroups"),
    argMax: wrapNullArgs("argMax"),
    argMin: wrapNullArgs("argMin"),
    argUnique: wrapNullArgs("argUnique"),
    backwardFill: wrapNullArgs("backwardFill"),
    count: wrapNullArgs("count"),
    explode: wrapNullArgs("explode"),
    first: wrapNullArgs("first"),
    floor: wrapNullArgs("floor"),
    forwardFill: wrapNullArgs("forwardFill"),
    interpolate: wrapNullArgs("interpolate"),
    isDuplicated: wrapNullArgs("isDuplicated"),
    isFinite: wrapNullArgs("isFinite"),
    isFirst: wrapNullArgs("isFirst"),
    isInfinite: wrapNullArgs("isInfinite"),
    isNan: wrapNullArgs("isNan"),
    isNotNan: wrapNullArgs("isNotNan"),
    isNotNull: wrapNullArgs("isNotNull"),
    isNull: wrapNullArgs("isNull"),
    isUnique: wrapNullArgs("isUnique"),
    keepName: wrapNullArgs("keepName"),
    last: wrapNullArgs("last"),
    list: wrapNullArgs("list"),
    lowerBound: wrapNullArgs("lowerBound"),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    median: wrapNullArgs("median"),
    min: wrapNullArgs("min"),
    mode: wrapNullArgs("mode"),
    not: wrapNullArgs("not"),
    nUnique: wrapNullArgs("nUnique"),
    reverse: wrapNullArgs("reverse"),
    std: wrapNullArgs("std"),
    sum: wrapNullArgs("sum"),
    unique: wrapNullArgs("unique"),
    upperBound: wrapNullArgs("upperBound"),
    var: wrapNullArgs("var"),
    and: wrapExprArg("and"),
    dot: wrapExprArg("dot"),
    eq: wrapExprArg("eq"),
    fillNan: wrapExprArg("fillNan"),
    fillNull: wrapExprArg("fillNull"),
    filter: wrapExprArg("filter"),
    gt: wrapExprArg("gt"),
    gtEq: wrapExprArg("gtEq"),
    isIn: wrapExprArg("isIn"),
    lt: wrapExprArg("lt"),
    ltEq: wrapExprArg("ltEq"),
    neq: wrapExprArg("neq"),
    or: wrapExprArg("or"),
    repeatBy: wrapExprArg("repeatBy"),
    take: wrapExprArg("take"),
    xor: wrapExprArg("xor"),
    rollingMedian: wrapUnary("rollingMedian", "windowSize"),
    pow: wrapUnary("pow", "exponent"),
    quantile: wrapUnary("quantile", "quantile"),
    shift: wrapUnary("shift", "periods"),
    suffix: wrapUnary("suffix", "suffix"),
    alias: wrapUnary("alias", "name"),
    prefix: wrapUnary("prefix", "prefix"),
    rank: wrapUnary("rank", "method"),
    fillNullWithStrategy: wrapUnary("fillNullWithStrategy", "strategy"),
    takeEvery: wrapUnary("takeEvery", "n"),
    exclude: wrapUnary("exclude", "columns"),
    over,
    reshape: wrapUnary("reshape", "dims"),
    argSort: wrapUnaryWithDefault("argSort", "reverse", false),
    cumCount: wrapUnaryWithDefault("cumCount", "reverse", false),
    cumMax: wrapUnaryWithDefault("cumMax", "reverse", false),
    cumMin: wrapUnaryWithDefault("cumMin", "reverse", false),
    cumProd: wrapUnaryWithDefault("cumProd", "reverse", false),
    cumSum: wrapUnaryWithDefault("cumSum", "reverse", false),
    sort: wrapUnaryWithDefault("sort", "reverse", false),
    reinterpret: wrapUnaryWithDefault("reinterpret", "signed", true),
    diff: wrapBinary("diff", "n", "nullBehavior"),
    slice: wrapBinary("slice", "offset", "length"),
    rollingQuantile: wrapBinary("rollingQuantile", "windowSize", "quantile"),
    shiftAndFill: wrapBinary("shiftAndFill", "periods", "fillValue"),
    cast: (dtype, strict=false) => wrap("cast", {dtype, strict}),
    rollingSkew: (windowSize, bias=false) => wrap("rollingSkew", {windowSize, bias}),
    kurtosis,
    hash
  } as any;
};


export const exprToLitOrExpr = (expr: any, stringToLit = true)  => {
  if(typeof expr === "string" && !stringToLit) {
    return col(expr);
  } else if (isExpr(expr)) {
    return expr;
  } else if (Array.isArray(expr)) {
    return expr.map(e => exprToLitOrExpr(e, stringToLit));
  } else {
    return lit(expr);
  }
};