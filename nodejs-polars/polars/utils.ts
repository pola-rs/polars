import type {Expr} from "./lazy/expr";
import path from "path";
import type {Series} from "./series";

export function columnOrColumns(columns: ColumnSelection |  string | Array<string> | undefined): Array<string> | undefined {
  if (columns) {
    return columnOrColumnsStrict(columns);
  }
}

export type ValueOrArray<T> = T | Array<ValueOrArray<T>>;
export type ColumnSelection = ValueOrArray<string>
export type ExpressionSelection = ValueOrArray<Expr>
export type ColumnsOrExpr = ColumnSelection | ExpressionSelection
export type Option<T> = T | undefined;


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

export const isSeries = <T>(ty: any): ty is Series<T> => ty._series !== undefined;
export const isExpr = (ty: any): ty is Expr => ty._expr !== undefined;