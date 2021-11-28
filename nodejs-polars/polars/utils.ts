import path from 'path/posix';

export function columnOrColumns(columns: string | Array<string> | undefined): Array<string> | undefined {
  if (columns) {
    return columnOrColumnsStrict(columns);
  }
}

export type ValueOrArray<T> = T | Array<ValueOrArray<T>>;
export type ColumnSelection = ValueOrArray<string>


export function columnOrColumnsStrict(...columns: string[] | ValueOrArray<string>[]): Array<string> {
  return columns.flat(3) as any;
}

export function isPath(s: string): boolean {
  const {base, ext, name} = path.parse(s);

  return Boolean(base && ext && name);
}

export const range = (start:number, end:number) => {
  const length = end - start;

  return Array.from({ length }, (_, i) => start + i);
};
