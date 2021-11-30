import pl from "../polars";

const col = (...args: any[]) => ({
  [Symbol.toPrimitive](hint) {
    if(hint === "number") {
      return this;
    } else {
      return null;
    }
  },
  sort: () => "sorted"
});

const x = [1,2];

const y = [1];
console.log(x < y);

col("a") as any  > 2;


// const c = pl.col("foo");

// console.log(c);


// type AnyFunc = (...args: any[]) => any
// type Expr = any;
// export type Option<T> = T | undefined;

// export interface LazyGroupBy {
//   agg(aggs: Array<Expr>): LazyDataFrame
//   head(n: number): LazyDataFrame
//   tail(n: number): LazyDataFrame
//   apply(func: AnyFunc): LazyDataFrame
// }
// /**
//  * Representation of a Lazy computation graph/ query.
//  */
// export interface LazyDataFrame {
//   new_from_csv(): String
//   describe_optimized_plan(): Promise<String>
//   to_dot(optimized: boolean): Promise<String>
//   optimization_toggle(by_column: string, reverse: boolean): LazyDataFrame
//   sort_by_exprs(by_column: Array<Expr>, reverse: Array<boolean>): LazyDataFrame
//   cache(): LazyDataFrame
//   collect(): Promise<LazyDataFrame>
//   fetch(n_rows: number): Promise<LazyDataFrame>
//   filter(self, predicate: Expr): LazyDataFrame
//   select(self, exprs: Array<Expr>): LazyDataFrame
//   groupby(self, by: Array<Expr>, maintain_order: boolean): LazyGroupBy
//   join(self, expr: Expr): LazyDataFrame
//   with_columns(self, exprs: Array<Expr>): LazyDataFrame
//   rename(self, existing: Array<String>, _new: Array<String>): LazyDataFrame
//   with_column_renamed(self, existing: string, _new: string): LazyDataFrame
//   reverse(): LazyDataFrame
//   shift(periods: number): LazyDataFrame
//   shift_and_fill(periods: number, fill_value: Expr): LazyDataFrame
//   fill_null(fill_value: Expr): LazyDataFrame
//   fill_nan(fill_value: Expr): LazyDataFrame
//   min(): LazyDataFrame
//   max(): LazyDataFrame
//   sum(): LazyDataFrame
//   mean(): LazyDataFrame
//   std(): LazyDataFrame
//   var(): LazyDataFrame
//   median(): LazyDataFrame
//   quantile(quantile: number): LazyDataFrame
//   explode(column: Array<Expr>): LazyDataFrame
//   drop_duplicates(maintain_order: boolean, subset: Option<Array<String>>): LazyDataFrame
//   drop_nulls(subset: Option<Array<String>>): LazyDataFrame
//   slice(offset: number, len: number): LazyDataFrame
//   tail(n: number): LazyDataFrame
//   melt(id_vars: Array<String>, value_vars: Array<String>): LazyDataFrame
//   with_row_count(name: string): LazyDataFrame
//   map(func: AnyFunc, predicate_pd: boolean, projection_pd: boolean): LazyDataFrame
//   drop_columns(cols: Array<String>): LazyDataFrame
//   clone(): LazyDataFrame
//   columns(): Array<String>
// }