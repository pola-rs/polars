import {Expr, exprToLitOrExpr} from "./lazy/expr";
import {Series} from "./series/series";
import {DataFrame} from "./dataframe";
import path from "path";
import {isExternal, isRegExp} from "util/types";
export type ValueOrArray<T> = T | Array<ValueOrArray<T>>;
export type ColumnSelection = ValueOrArray<string>
export type ExpressionSelection = ValueOrArray<Expr>
export type ColumnsOrExpr = ColumnSelection | ExpressionSelection
export type ExprOrString = Expr | string
export type DownsampleRule =  "month" | "week" | "day" | "hour" | "minute" | "second"
export type FillNullStrategy = "backward" |  "forward" | "mean" | "min" | "max" | "zero" | "one"
export type RankMethod = "average" | "min" | "max" | "dense" | "ordinal" | "random";
export type RollingOptions = {
  windowSize: number,
  weights?: Array<number>,
  minPeriods?: number,
  center?: boolean
};

export function columnOrColumns(columns: ColumnSelection |  string | Array<string> | undefined): Array<string> | undefined {
  if (columns) {
    return columnOrColumnsStrict(columns);
  }
}
export function columnOrColumnsStrict(...columns: string[] | ValueOrArray<string>[]): Array<string> {
  return columns.flat(3) as any;
}
export function selectionToExprList(columns: any[], stringToLit?) {
  return [columns].flat(3).map(expr => exprToLitOrExpr(expr, stringToLit)._expr);
}

export function isPath(s: string, expectedExtensions?: string[]): boolean {
  const {base, ext, name} = path.parse(s);

  return Boolean(base && ext && name) && !!(expectedExtensions?.includes(ext));
}

export const range = (start: number, end: number) => {
  const length = end - start;

  return Array.from({ length }, (_, i) => start + i);
};


export const isDataFrameArray = (ty: any): ty is DataFrame[] => Array.isArray(ty) &&  DataFrame.isDataFrame(ty[0]);
export const isSeriesArray = <T>(ty: any): ty is Series[] => Array.isArray(ty) &&  Series.isSeries(ty[0]);
export const isExprArray = (ty: any): ty is Expr[] => Array.isArray(ty) && Expr.isExpr(ty[0]);
export const isIterator = <T>(ty: any): ty is Iterable<T> => ty !== null && typeof ty[Symbol.iterator] === "function";
export const regexToString = (r: string | RegExp): string => {
  if(isRegExp(r)) {
    return r.source;
  }

  return r;
};

export const INSPECT_SYMBOL = Symbol.for("nodejs.util.inspect.custom");
