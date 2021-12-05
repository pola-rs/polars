import type {Expr} from "./lazy/expr";
import type {Series} from "./series";
import type {DataFrame} from "./dataframe";
import path from "path";

export type ValueOrArray<T> = T | Array<ValueOrArray<T>>;
export type ColumnSelection = ValueOrArray<string>
export type ExpressionSelection = ValueOrArray<Expr>
export type ColumnsOrExpr = ColumnSelection | ExpressionSelection
export type Option<T> = T | undefined;
export type DownsampleRule =  "month" | "week" | "day" | "hour" | "minute" | "second"
export type FillNullStrategy = "backward" | "forward" | "mean" | "min" | "max" | "zero" | "one"
export type RankMethod = "average" | "min" | "max" | "dense" | "ordinal" | "random";

export function columnOrColumns(columns: ColumnSelection |  string | Array<string> | undefined): Array<string> | undefined {
  if (columns) {
    return columnOrColumnsStrict(columns);
  }
}
export function columnOrColumnsStrict(...columns: string[] | ValueOrArray<string>[]): Array<string> {
  return columns.flat(3) as any;
}

export function isPath(s: string, expectedExtensions?: string[]): boolean {
  const {base, ext, name} = path.parse(s);

  return Boolean(base && ext && name) && !!(expectedExtensions?.includes(ext));
}

export const range = (start:number, end:number) => {
  const length = end - start;

  return Array.from({ length }, (_, i) => start + i);
};


export const isDataFrame = (ty: any): ty is DataFrame => ty._df !== undefined;
export const isDataFrameArray = (ty: any): ty is DataFrame[] => Array.isArray(ty) &&  ty[0]._df !== undefined;
export const isSeries = <T>(ty: any): ty is Series<T> => ty._series !== undefined;
export const isSeriesArray = <T>(ty: any): ty is Series<T>[] => Array.isArray(ty) &&  ty[0]._series !== undefined;
export const isExpr = (ty: any): ty is Expr => ty._expr !== undefined;
export const isExprArray = (ty: any): ty is Expr[] => Array.isArray(ty) && ty[0]._expr !== undefined;