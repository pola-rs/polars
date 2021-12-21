
import {DataFrame, dfWrapper} from "../dataframe";
import {Expr, exprToLitOrExpr} from "./expr";
import {
  columnOrColumnsStrict,
  ColumnSelection,
  ColumnsOrExpr,
  ExprOrString,
  selectionToExprList,
  ValueOrArray
} from "../utils";
import pli from "../internals/polars_internal";
import {LazyGroupBy} from "./groupby";

type JsLazyFrame = any;

type LazyJoinOptions =  {
  how?: "left" | "inner" | "outer" | "cross";
  suffix?: string,
  allowParallel?: boolean,
  forceParallel?: boolean
};


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
  _ldf: any;
  get columns(): string[]
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
  collect(opts?: LazyOptions): Promise<DataFrame>
  collectSync(opts?: LazyOptions): DataFrame
  /**
   * A string representation of the optimized query plan.
   */
  describeOptimizedPlan(opts?: LazyOptions): string
  /**
   * A string representation of the unoptimized query plan.
   */
  describePlan(): string
  /**
   * Remove one or multiple columns from a DataFrame.
   * @param columns - column or list of columns to be removed
   */
  drop(name: string): LazyDataFrame
  drop(names: string[]): LazyDataFrame
  drop(name: string, ...names: string[]): LazyDataFrame
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
  dropNulls(column: string): LazyDataFrame
  dropNulls(columns: string[]): LazyDataFrame
  dropNulls(...columns: string[]): LazyDataFrame

  /**
   * Explode lists to long format.
   */
  explode(column: ExprOrString): LazyDataFrame
  explode(columns: ExprOrString[]): LazyDataFrame
  explode(column: ExprOrString, ...columns: ExprOrString[]): LazyDataFrame
  /**
   * Fetch is like a collect operation, but it overwrites the number of rows read by every scan
   *
   *
   * Note that the fetch does not guarantee the final number of rows in the DataFrame.
   * Filter, join operations and a lower number of rows available in the scanned file influence
   * the final number of rows.
   * @param numRows - collect 'n' number of rows from data source
   * @param opts
   * @param opts.typeCoercion -Do type coercion optimization.
   * @param opts.predicatePushdown - Do predicate pushdown optimization.
   * @param opts.projectionPushdown - Do projection pushdown optimization.
   * @param opts.simplifyExpression - Run simplify expressions optimization.
   * @param opts.stringCache - Use a global string cache in this query.
   */
  fetch(numRows?: number): DataFrame
  fetch(numRows: number, opts: LazyOptions): DataFrame
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
  first(): DataFrame
  /**
   * Start a groupby operation.
   */
  groupBy(by: ColumnsOrExpr, maintainOrder?: boolean): LazyGroupBy
  groupBy(by: ColumnsOrExpr, opts: {maintainOrder: boolean}): LazyGroupBy
  /**
   * Gets the first `n` rows of the DataFrame. You probably don't want to use this!
   *
   * Consider using the `fetch` operation.
   * The `fetch` operation will truly load the first `n`rows lazily.
   */
  head(length?: number): LazyDataFrame
  /**
   * Add a join operation to the Logical Plan.
   */
  join(df: LazyDataFrame, joinOptions: {on: ValueOrArray<string | Expr>} & LazyJoinOptions): LazyDataFrame
  join(df: LazyDataFrame, joinOptions:  {leftOn: ValueOrArray<string | Expr>, rightOn: ValueOrArray<string | Expr>} & LazyJoinOptions): LazyDataFrame
  /**
   * Get the last row of the DataFrame.
   */
  last(): LazyDataFrame
  /**
   * @see {@link head}
   */
  limit(n?: number): LazyDataFrame
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
  // pipe() todo
  /**
   * @see {@link DataFrame.quantile}
   */
  quantile(quantile: number): LazyDataFrame
  /**
   * @see {@link DataFrame.rename}
   */
  rename(mapping: Record<string, string>)
  /**
   * Reverse the DataFrame.
   */
  reverse()
  /**
   * @see {@link DataFrame.select}
   */
  select(column: ExprOrString): LazyDataFrame
  select(columns: ExprOrString[]): LazyDataFrame
  select(column: ExprOrString, ...columns: ExprOrString[]): LazyDataFrame
  /**
   * @see {@link DataFrame.shift}
   */
  shift(periods: number): LazyDataFrame
  shift(opts: {periods: number}): LazyDataFrame
  /**
   * @see {@link DataFrame.shiftAndFill}
   */
  shiftAndFill(periods: number, fillValue: number | string | Expr): LazyDataFrame
  shiftAndFill(opts: {periods: number, fillValue: number | string | Expr}): LazyDataFrame
  /**
   * @see {@link DataFrame.slice}
   */
  slice(offset: number, length: number): LazyDataFrame
  slice(opts: {offset: number, length: number}): LazyDataFrame
  /**
   * @see {@link DataFrame.sort}
   */
  sort(by: ColumnsOrExpr, reverse?: ValueOrArray<boolean> ): LazyDataFrame
  sort(opts: {by: ColumnsOrExpr, reverse?: ValueOrArray<boolean>}): LazyDataFrame
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
  tail(length?: number): LazyDataFrame
  /**
   * Aggregate the columns in the DataFrame to their variance value.
   */
  var(): LazyDataFrame
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
  withColumns(exprs: Expr[]): LazyDataFrame
  withColumns(expr: Expr, ...exprs: Expr[]): LazyDataFrame
  withColumnRenamed(existing: string, replacement: string): LazyDataFrame
  /**
   * Add a column at index 0 that counts the rows.
   * @see {@link DataFrame.withRowCount}
   */
  withRowCount()
}


export const LazyDataFrame = (ldf: JsLazyFrame): LazyDataFrame => {
  const unwrap = (method: string, args?: object, _ldf=ldf): any => {
    return pli.ldf[method]({_ldf, ...args });
  };
  const wrap = (method, args?, _ldf=ldf): LazyDataFrame => {
    return LazyDataFrame(unwrap(method, args, _ldf));
  };
  const wrapNullArgs = (method: string) => () => wrap(method);
  const withOptimizationToggle = (method) =>  (lazyOptions?: LazyOptions) => {
    const ldf = unwrap("optimizationToggle", lazyOptions);

    return unwrap(method, {}, ldf);

  };

  return {
    _ldf: ldf,
    get columns() {
      return unwrap("columns");
    },
    describePlan: () => unwrap("describePlan"),
    describeOptimizedPlan: withOptimizationToggle("describeOptimizedPlan"),
    cache: wrapNullArgs("cache"),
    collectSync: () => dfWrapper(unwrap("collectSync")),
    collect: () => unwrap("collect").then(dfWrapper),
    drop: (...cols) => wrap("dropColumns", {cols: cols.flat(2)}),
    dropDuplicates(opts: any=true, subset?){
      if(opts?.maintainOrder !== undefined) {
        return this.dropDuplicates(opts.maintainOrder, opts.subset);
      }
      if(subset) {
        subset = [subset].flat(2);
      }

      return wrap("dropDuplicates", {maintainOrder: opts.maintainOrder, subset});
    },
    dropNulls(...subset) {
      if(subset.length) {
        return wrap("dropNulls", {subset: subset.flat(2)});
      } else {
        return wrap("dropNulls");
      }
    },
    explode(...columns) {
      if(!columns.length) {
        const cols = selectionToExprList(LazyDataFrame(ldf).columns, false);

        return wrap("explode", {column: cols});
      }
      const column = selectionToExprList(columns, false);

      return wrap("explode", {column});
    },
    fetch(numRows, opts?: LazyOptions) {
      if(opts?.noOptimization) {
        opts.predicatePushdown = false;
        opts.projectionPushdown = false;
      }
      if(opts) {
        const _ldf = unwrap("optimizationToggle", opts);

        return dfWrapper(unwrap("fetchSync", {numRows}, _ldf));

      }

      return dfWrapper(unwrap("fetchSync", {numRows}));

    },
    first() {
      return this.fetch(1);
    },
    fillNull(exprOrValue) {
      const fillValue = exprToLitOrExpr(exprOrValue)._expr;

      return wrap("fillNull", {fillValue});
    },
    filter(exprOrValue) {
      const predicate = exprToLitOrExpr(exprOrValue, false)._expr;

      return wrap("filter", {predicate});

    },
    groupBy(opt, maintainOrder: any=true) {
      if(opt?.by !== undefined) {
        return LazyGroupBy(ldf, opt.by, opt.maintainOrder);
      }

      return LazyGroupBy(ldf, opt, maintainOrder);
    },
    head: (len=5) => wrap("slice", {offset: 0, len}),
    join(df, options: {[k: string]: any} & LazyJoinOptions ) {
      options =  {
        how: "inner",
        suffix: "right",
        allowParallel: true,
        forceParallel: false,
        ...options
      };
      const {how, suffix, allowParallel, forceParallel} = options;
      let leftOn;
      let rightOn;
      if(options.on) {
        const on = selectionToExprList(options.on, false);
        leftOn = on;
        rightOn = on;
      } else if((options.leftOn && !options.rightOn) || (options.rightOn && !options.leftOn)) {
        throw new TypeError("You should pass the column to join on as an argument.");
      } else {
        leftOn = selectionToExprList(options.leftOn, false);
        rightOn = selectionToExprList(options.rightOn, false);
      }

      return wrap("join", {
        other: df._ldf,
        how,
        leftOn,
        rightOn,
        suffix,
        allowParallel,
        forceParallel
      });
    },
    last: () => wrap("tail", {length: 1}),
    limit: (len=5) => wrap("slice", {offset: 0, len}),
    max: wrapNullArgs("max"),
    mean: wrapNullArgs("mean"),
    median: wrapNullArgs("median"),
    melt(ids, values) {
      return wrap("melt", {
        idVars: columnOrColumnsStrict(ids),
        valueVars: columnOrColumnsStrict(values)
      });
    },
    min: wrapNullArgs("min"),
    quantile: (quantile) => wrap("quantile", {quantile}),
    rename(mapping) {
      const existing = Object.keys(mapping);
      const replacements = Object.values(mapping);

      return wrap("rename", {existing, replacements});
    },
    reverse: wrapNullArgs("reverse"),
    select(...exprs) {
      const predicate = selectionToExprList(exprs, false);

      return wrap("select", {predicate});
    },
    shift: (periods) => wrap("shift", {periods}),
    shiftAndFill(optOrPeriods, fillValue?) {
      if(typeof optOrPeriods === "number") {
        fillValue = exprToLitOrExpr(fillValue)._expr;

        return wrap("shiftAndFill", {periods: optOrPeriods, fillValue});

      }
      else {
        fillValue = exprToLitOrExpr(optOrPeriods.fillValue)._expr;
        const periods = optOrPeriods.periods;

        return wrap("shiftAndFill", {periods, fillValue});
      }
    },
    slice(opt, len?) {
      if(opt?.offset !== undefined) {
        return wrap("slice", {offset: opt.offset, len: opt.length});
      }

      return wrap("slice", {offset: opt, len});
    },
    sort(arg, reverse=false)  {
      if(arg?.by  !== undefined) {
        return this.sort(arg.by, arg.reverse);
      }
      if(typeof arg === "string") {
        return wrap("sort", {by: arg, reverse});
      } else {
        reverse = [reverse].flat(3) as any;
        const by = selectionToExprList(arg, false);

        return wrap("sort_by_exprs", {by, reverse});
      }
    },
    std: wrapNullArgs("std"),
    sum: wrapNullArgs("sum"),
    var: wrapNullArgs("var"),
    tail: (length=5) => wrap("tail", {length}),
    withColumn: (expr) => wrap("withColumn", {expr: expr._expr}),
    withColumns(...columns){
      const exprs = selectionToExprList(columns, false);

      return wrap("withColumns", {exprs});
    },
    withColumnRenamed: (existing, replacement) => wrap("withColumnRenamed", {existing, replacement}),
    withRowCount: (name="row_nr") => wrap("withRowCount", {name}),
  };
};
