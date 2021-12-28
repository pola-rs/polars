/* eslint-disable no-redeclare */
import {DataFrame, dfWrapper, _wrapDataFrame} from "./dataframe";
import * as utils from "./utils";
import type {ColumnSelection} from "./utils";
const inspect = Symbol.for("nodejs.util.inspect.custom");
import util from "util";
import {InvalidOperationError, todo} from "./error";
import {Expr} from "./lazy/expr";


import {col, exclude} from "./lazy/lazy_functions";

const inspectOpts = {colors:true, depth:null};

/**
 * Starts a new GroupBy operation.
 */
export interface GroupBy {
  [inspect](): string,
  /**
   * Aggregate the groups into Series.
   */
  aggList(): DataFrame
  /**
   * __Use multiple aggregations on columns.__
   * This can be combined with complete lazy API and is considered idiomatic polars.
   * ___
   * @param columns - map of 'col' -> 'agg'
   *
   *  - using lazy API (recommended): `[col('foo').sum(), col('bar').min()]`
   *  - using multiple aggs per column: `{'foo': ['sum', 'numUnique'], 'bar': ['min'] }`
   *  - using single agg per column:  `{'foo': ['sum'], 'bar': 'min' }`
   * @example
   * ```
   * // use lazy api rest parameter style
   * >>> df.groupBy('foo', 'bar')
   * >>>   .agg(pl.sum('ham'), col('spam').tail(4).sum())
   *
   * // use lazy api array style
   * >>> df.groupBy('foo', 'bar')
   * >>>   .agg([pl.sum('ham'), col('spam').tail(4).sum()])
   *
   * // use a mapping
   * >>> df.groupBy('foo', 'bar')
   * >>>   .agg({'spam': ['sum', 'min']})
   *
   * ```
   */
   agg(...columns: Expr[]): DataFrame
   agg(columns: Record<string, keyof Expr | (keyof Expr)[]>): DataFrame
  /**
   * Count the number of values in each group.
   */
  count(): DataFrame
  /**
   * Aggregate the first values in the group.
   */
  first(): DataFrame

  /**
   * Return a `DataFrame` with:
   *   - the groupby keys
   *   - the group indexes aggregated as lists
   */
  groups(): DataFrame
  /**
   * Return first n rows of each group.
   * @param n -Number of values of the group to select
   * @example
   * ```
   * >>> df = pl.DataFrame({
   * >>>   "letters": ["c", "c", "a", "c", "a", "b"],
   * >>>   "nrs": [1, 2, 3, 4, 5, 6]
   * >>> })
   * >>> df
   * shape: (6, 2)
   * ╭─────────┬─────╮
   * │ letters ┆ nrs │
   * │ ---     ┆ --- │
   * │ str     ┆ i64 │
   * ╞═════════╪═════╡
   * │ "c"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 2   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 3   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 4   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 5   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 6   │
   * ╰─────────┴─────╯
   * >>> df.groupby("letters")
   * >>>   .head(2)
   * >>>   .sort("letters");
   * >>>
   * shape: (5, 2)
   * ╭─────────┬─────╮
   * │ letters ┆ nrs │
   * │ ---     ┆ --- │
   * │ str     ┆ i64 │
   * ╞═════════╪═════╡
   * │ "a"     ┆ 3   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "a"     ┆ 5   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "b"     ┆ 6   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 1   │
   * ├╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
   * │ "c"     ┆ 2   │
   * ╰─────────┴─────╯
   * ```
   */
  head(n: number): DataFrame
  /**
   * Aggregate the last values in the group.
   */
  last(): DataFrame
  /**
   * Reduce the groups to the maximal value.
   */
  max(): DataFrame
  /**
   * Reduce the groups to the mean values.
   */
  mean(): DataFrame
  /**
   * Return the median per group.
   */
  median(): DataFrame
  /**
   * Reduce the groups to the minimal value.
   */
  min(): DataFrame
  /**
   * Count the unique values per group.
   */
  nUnique(): DataFrame
  /**
   * Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
   * @param pivotCol - Column to pivot.
   * @param valuesCol - Column that will be aggregated.
   *
   */
  pivot({pivotCol, valuesCol}: {pivotCol: string, valuesCol: string}): PivotOps
  pivot(pivotCol: string, valuesCol: string): PivotOps
  /**
   * Compute the quantile per group.
   */
  quantile(quantile: number): DataFrame
  /**
   * Reduce the groups to the sum.
   */
  sum(): DataFrame
  tail(): DataFrame
  toString(): string


}


export type PivotOps = Pick<GroupBy,
  "count"
  | "first"
  | "max"
  | "mean"
  | "median"
  | "min"
  | "sum"
> & {[inspect](): string}

export function GroupBy(
  df: DataFrame,
  by: string[],
  maintainOrder = false,
  downsample = false
) {
  const customInspect = () => util.formatWithOptions(inspectOpts, "GroupBy {by: %O}", by);

  const pivot = (opts: {pivotCol: string, valuesCol: string} | string, valuesCol?: string): PivotOps => {
    if(downsample) {
      throw new InvalidOperationError("pivot", "downsample");
    }

    if(typeof opts === "string") {
      if(valuesCol) {
        return pivot({pivotCol: opts, valuesCol});
      } else {
        throw new Error("must specify both pivotCol and valuesCol");
      }
    }

    return PivotOps(df, by, opts.pivotCol, opts.valuesCol);
  };

  const agg = (...aggs): DataFrame => {

    if(utils.isExprArray(aggs))  {
      aggs = [aggs].flat(2);

      return dfWrapper(df).lazy()
        .groupBy(by, maintainOrder)
        .agg(...aggs)
        .collectSync({noOptimization:true});
    } else {
      let pairs = Object.entries(aggs[0])
        .flatMap(([key, values]) => {
          return [values].flat(2).map(v => col(key)[v as any]());
        });

      return dfWrapper(df)
        .lazy()
        .groupBy(by, maintainOrder)
        .agg(...pairs)
        .collectSync({noOptimization:true});
    }
  };


  return Object.seal({
    [inspect]: customInspect,
    agg,
    pivot,
    aggList: () => agg(exclude(by as any).list()),
    count: () => _wrapDataFrame(df, "groupby", {by, agg: "count"}),
    first: () => agg(exclude(by as any).first()),
    groups: () => _wrapDataFrame(df, "groupby", {by, agg: "groups"}),
    head: (n=5) => agg(exclude(by as any).head(n)),
    last: () => agg(exclude(by as any).last()),
    max: () => agg(exclude(by as any).max()),
    mean: () => agg(exclude(by as any).mean()),
    median: () => agg(exclude(by as any).median()),
    min: () => agg(exclude(by as any).min()),
    nUnique: () => agg(exclude(by as any).nUnique()),
    quantile: (q: number) =>  agg(exclude(by as any).quantile(q)),
    sum: () => agg(exclude(by as any).sum()),
    tail: (n=5) => agg(exclude(by as any).tail(n)),
    toString: () => "GroupBy",
  }) as GroupBy;
}

function PivotOps(
  df: DataFrame,
  by: string | string[],
  pivotCol: string,
  valueCol: string
): PivotOps {

  const pivot =  (agg) => () =>  _wrapDataFrame(df, "pivot", {by, pivotCol, valueCol, agg});
  const customInspect = () => util.formatWithOptions(inspectOpts, "PivotOps {by: %O}", by);

  return {
    [inspect]: customInspect,
    first: pivot("first"),
    sum: pivot("sum"),
    min: pivot("min"),
    max: pivot("max"),
    mean: pivot("mean"),
    count: pivot("count"),
    median: pivot("median"),
  };
}
