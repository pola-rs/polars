import {DataFrame} from "@polars/dataframe";
import {ColumnSelection, ExpressionSelection} from "@polars/utils";

type AnyFunc = (...args: any[]) => any
type Expr = any;
export type Option<T> = T | undefined;
type ColumnsOrExpr = ColumnSelection | ExpressionSelection

type LazyJoinOptions =  {
  leftOn?: ColumnsOrExpr;
  rightOn?: ColumnsOrExpr;
  on?: ColumnsOrExpr;
  how?: "left" | "inner" | "outer" | "cross";
  suffix?: string,
  allowParallel: boolean,
  forceParallel: boolean
};

export interface LazyGroupBy {
  agg(aggs: Array<Expr>): LazyDataFrame
  head(n: number): LazyDataFrame
  tail(n: number): LazyDataFrame
  apply(func: AnyFunc): LazyDataFrame
}

type LazyOptions = {
  typeCoercion?: boolean,
  predicatePushdown?: boolean,
  projectionPushdown?: boolean,
  simplifyExpression?: boolean,
  stringCache?: boolean,
  noOptimization?: boolean,
}

/**
 * Representation of a Lazy computation graph/ query.
 */
export interface LazyDataFrame {
  get columns(): string
  /**
   * Cache the result once the execution of the physical plan hits this node.
   */
  cache(): LazyDataFrame
  /**
   *
   * Collect into a DataFrame.
   * Note: use `fetch` if you want to run this query on the first `n` rows only.
   * This can be a huge time saver in debugging queries.
   * @param typeCoercion -Do type coercion optimization.
   * @param predicatePushdown - Do predicate pushdown optimization.
   * @param projectionPushdown - Do projection pushdown optimization.
   * @param simplifyExpression - Run simplify expressions optimization.
   * @param stringCache - Use a global string cache in this query.
   *     This is needed if you want to join on categorical columns.
   *     Caution!
   * *  If you already have set a global string cache, set this to `false` as this will reset the
   * *  global cache when the query is finished.
   * @param noOptimization - Turn off optimizations.
   * @return DataFrame
   *
   */
  collect(opts?: LazyOptions): DataFrame
  /**
   * A string representation of the optimized query plan.
   */
  describeOptimizedPlan(opts?: LazyOptions): string
  /**
   * A string representation of the unoptimized query plan.
   */
  describePlan(): string
  /**
   * Drop duplicate rows from this DataFrame.
   * Note that this fails if there is a column of type `List` in the DataFrame.
   */
  dropDuplicates(opts: {maintainOrder?: boolean, subset?: ColumnSelection}): LazyDataFrame
  dropDuplicates(maintainOrder?: boolean, subset?: ColumnSelection): LazyDataFrame
  /**
   * Drop rows with null values from this DataFrame.
   * This method only drops nulls row-wise if any single value of the row is null.
   */

  dropNulls(subset?: ColumnSelection): LazyDataFrame
  dropNulls(opts: {subset: ColumnSelection}): LazyDataFrame
  /**
   * Remove one or multiple columns from a DataFrame.
   * @param columns - column or list of columns to be removed
   */
  drop(columns: ColumnSelection): LazyDataFrame
  drop(opts: {columns: ColumnSelection}): LazyDataFrame
  /**
   * Explode lists to long format.
   */
  explode(columns: ColumnsOrExpr): LazyDataFrame
  explode(opts: {columns: ColumnsOrExpr}): LazyDataFrame
  /**
   * Fetch is like a collect operation, but it overwrites the number of rows read by every scan
   *
   *
   * Note that the fetch does not guarantee the final number of rows in the DataFrame.
   * Filter, join operations and a lower number of rows available in the scanned file influence
   * the final number of rows.
   * @param numRows - collect 'n' number of rows from data source
   * @param typeCoercion -Do type coercion optimization.
   * @param predicatePushdown - Do predicate pushdown optimization.
   * @param projectionPushdown - Do projection pushdown optimization.
   * @param simplifyExpression - Run simplify expressions optimization.
   * @param stringCache - Use a global string cache in this query.

   */
  fetch(opts: LazyOptions & {numRows: number})
  /**
   * Fill missing values
   * @param fillValue value to fill the missing values with
   */
  fillNull(fillValue: string | number | Expr): LazyDataFrame
  /**
   * Filter the rows in the DataFrame based on a predicate expression.
   * @param predicate - Expression that evaluates to a boolean Series.
   * @example
   * ```
   * >>> lf = pl.DataFrame({
   * >>>   "foo": [1, 2, 3],
   * >>>   "bar": [6, 7, 8],
   * >>>   "ham": ['a', 'b', 'c']
   * >>> }).lazy()
   * >>> // Filter on one condition
   * >>> lf.filter(pl.col("foo").lt(3)).collect()
   * shape: (2, 3)
   * ┌─────┬─────┬─────┐
   * │ foo ┆ bar ┆ ham │
   * │ --- ┆ --- ┆ --- │
   * │ i64 ┆ i64 ┆ str │
   * ╞═════╪═════╪═════╡
   * │ 1   ┆ 6   ┆ a   │
   * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
   * │ 2   ┆ 7   ┆ b   │
   * └─────┴─────┴─────┘
   * ```
   */
  filter(predicate: Expr | string): LazyDataFrame
  /**
   * Get the first row of the DataFrame.
   */
  first(): LazyDataFrame
  /**
   * Start a groupby operation.
   */
  groupby(by: ColumnsOrExpr, maintainOrder?: boolean): LazyGroupBy
  groupby(by: ColumnsOrExpr, opts: {maintainOrder: boolean}): LazyGroupBy
  /**
   * Gets the first `n` rows of the DataFrame. You probably don't want to use this!
   *
   * Consider using the `fetch` operation.
   * The `fetch` operation will truly load the first `n`rows lazily.
   */
  head(length: number): LazyDataFrame
  /**
   * Prints the value that this node in the computation graph evaluates to and passes on the value.
   */
  inspect(): LazyDataFrame
  /**
   * Interpolate intermediate values. The interpolation method is linear.
   */
  interpolate()
  /**
   * Add a join operation to the Logical Plan.
   */
  join(df: LazyDataFrame, joinOptions: LazyJoinOptions): LazyDataFrame
  /**
   * Get the last row of the DataFrame.
   */
  last(): LazyDataFrame
  /**
   * @see {@link head}
   */
  limit(): LazyDataFrame
  /**
   * Apply a custom function. It is important that the function returns a Polars DataFrame.
   * @param func - Lambda/ function to apply.
   * @param opts -
   * @param opts.predicatePushdown
   * @param opts.projectionPushdown
   * @param opts.noOptimizations
   * @see {@link LazyOptions}
   */
  map(func: (df: DataFrame) => DataFrame, opts?: Pick<LazyOptions, "predicatePushdown"| "projectionPushdown" | "noOptimization">): LazyDataFrame
  /**
   * @see {@link DataFrame.max}
   */
  max(): LazyDataFrame
  /**
   * @see {@link DataFrame.mean}
   */
  mean(): LazyDataFrame
  /**
   * @see {@link DataFrame.median}
   */
  median(): LazyDataFrame
  /**
   * @see {@link DataFrame.melt}
   */
  melt(idVars: ColumnSelection, valueVars: ColumnSelection): LazyDataFrame
  /**
   * @see {@link DataFrame.min}
   */
  min(): LazyDataFrame
  pipe()
  /**
   * @see {@link DataFrame.quantile}
   */
  quantile(quantile: number): LazyDataFrame
  /**
   * @see {@link DataFrame.rename}
   */
  rename(mapping: Record<string,string>)
  /**
   * Reverse the DataFrame.
   */
  reverse()
  /**
   * @see {@link DataFrame.select}
   */
  select(...columns: ColumnsOrExpr[]): LazyDataFrame
  /**
   * @see {@link DataFrame.shiftAndFill}
   */
  shiftAndFill(periods: number, fillValue: number | string): LazyDataFrame
  shiftAndFill(opts: {periods: number, fillValue: number | string}): LazyDataFrame
  /**
   * @see {@link DataFrame.shift}
   */
  shift(periods: number): LazyDataFrame
  shift(opts: {periods: number}): LazyDataFrame
  /**
   * @see {@link DataFrame.slice}
   */
  slice(opts: {offset: number, length: number}): LazyDataFrame
  slice(offset: number, length: number): LazyDataFrame
  /**
   * @see {@link DataFrame.sort}
   */
  sort(by: string, reverse?: boolean): LazyDataFrame
  sort(opts: {by: string, reverse?: boolean}): LazyDataFrame
  /**
   * @see {@link DataFrame.std}
   */
  std(): LazyDataFrame
  /**
   * Aggregate the columns in the DataFrame to their sum value.
   */
  sum(): LazyDataFrame
  /**
   * Get the last `n` rows of the DataFrame.
   * @see {@link DataFrame.tail}
   */
  tail(length: number): LazyDataFrame
  /**
   * Aggregate the columns in the DataFrame to their variance value.
   */
  var(): LazyDataFrame
  withColumnRenamed(existing: string, replacement: string): LazyDataFrame
  /**
   * Add or overwrite column in a DataFrame.
   * @param expr - Expression that evaluates to column.
   */
  withColumn(expr: Expr): LazyDataFrame
  /**
   * Add or overwrite multiple columns in a DataFrame.
   * @param exprs - List of Expressions that evaluate to columns.
   *
   */
  with_columns(...exprs: ExpressionSelection[]): LazyDataFrame
  /**
   * Add a column at index 0 that counts the rows.
   * @see {@link DataFrame.withRowCount}
   */
  withRowCount()
}
