import {Expr, exprToLitOrExpr} from "./lazy/expr";
import type {Series} from "./series";
import type {DataFrame} from "./dataframe";
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
export function selectionToExprList(columns: any[], stringToLit?): Expr[] {
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


export const isDataFrame = (ty: any): ty is DataFrame => isExternal(ty?._df);
export const isDataFrameArray = (ty: any): ty is DataFrame[] => Array.isArray(ty) &&  isExternal(ty[0]?._df);
export const isSeries = <T>(ty: any): ty is Series<T> => isExternal(ty._series);
export const isSeriesArray = <T>(ty: any): ty is Series<T>[] => Array.isArray(ty) &&  isExternal(ty[0]?._series);
export const isExpr = (ty: any): ty is Expr => isExternal(ty?._expr);
export const isExprArray = (ty: any): ty is Expr[] => Array.isArray(ty) && isExternal(ty[0]?._expr);
export const regexToString = (r: string | RegExp): string => {
  if(isRegExp(r)) {
    return r.source;
  }

  return r;
};
