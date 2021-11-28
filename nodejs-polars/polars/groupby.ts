/* eslint-disable no-redeclare */
import {DataFrame} from './dataframe';
import * as utils from './utils';
import type {ColumnSelection} from './utils';
const inspect = Symbol.for('nodejs.util.inspect.custom');
import util from 'util';
import {todo} from './internals/utils';

const _wrap = Symbol.for('wrap');
class UnsupportedError extends Error {
  constructor(prop, target) {
    super(`${prop} is not supported for ${target}`);
  }
}
/**
 * Starts a new GroupBy operation.
 */
export type GroupBy = {
  (...columns: string[]): GroupBySelection,
  (columns: string | string[]): GroupBySelection,
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
   * // use lazy api array styple
   * >>> df.groupBy('foo', 'bar')
   * >>>   .agg({pl.sum('ham'), col('spam').tail(4).sum()])
   * 
   * // use a mapping
   * >>> df.groupBy('foo', 'bar')
   * >>>   .agg({'spam': ['sum', 'min']})
   * 
   * ```
   */
  agg(columns: Record<string, string | string[]>): DataFrame
  /**
   * Apply a function over the groups as a sub-DataFrame.
   * @param func Function to apply on groupings
   */
  apply(func: (df: DataFrame) => DataFrame): DataFrame
  /**
   * Count the number of values in each group.
   */
  count(): DataFrame
  /**
   * Aggregate the first values in the group.
   */
  first(): DataFrame
  /**
   * Select a single group as a new DataFrame.
   * @param groupValue - Group to select.
   */
  getGroup(): DataFrame
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
  numUnique(): DataFrame
  /**
   * Do a pivot operation based on the group key, a pivot column and an aggregation function on the values column.
   * @param pivotCol - Column to pivot.
   * @param valuesCol - Column that will be aggregated.
   * 
   */
  pivot({pivotCol, valuesCol}: {pivotCol: string, valuesCol: string}): PivotOps
  pivot(pivotCol: string, valuesCol: string): PivotOps
  quantile(quantile: number): DataFrame
  /**
   * Reduce the groups to the sum.
   */
  sum(): DataFrame
  tail(): DataFrame


}

export type GroupBySelection = Pick<GroupBy, 
   'aggList'
  | 'apply'
  | 'count'
  | 'first'
  | 'last'
  | 'max'
  | 'mean'
  | 'median'
  | 'min'
  | 'numUnique'
  | 'quantile'
  | 'sum'
>

export type PivotOps = Pick<GroupBy, 
  'count'
  | 'first'
  | 'max'
  | 'mean'
  | 'median'
  | 'min'
  | 'sum'
>

export function GroupBy(
  df: DataFrame,
  by: string[],
  downsample = false,
  rule?: string,
  downsampleN = 0
) {
  const pivot = (opts: {pivotCol: string, valuesCol: string} | string, valuesCol?: string): PivotOps => {
    if(downsample) {
      throw new UnsupportedError('pivot', 'downsample');
    }

    if(typeof opts === 'string') {
      if(valuesCol) {
        return pivot({pivotCol: opts, valuesCol});
      } else {
        throw new Error("must specify both pivotCol and valuesCol");
      }
    }
 
    return PivotOps(df, by, opts.pivotCol, opts.valuesCol);
  };
  
  const select = (...columns: ColumnSelection[]): GroupBySelection => {
    if(downsample) {
      throw new Error("select not supported in downsample operation");
    }

    return GroupBySelection(
      df, 
      by, 
      utils.columnOrColumnsStrict(columns)
    );
  };

  const selectAll = (): GroupBySelection  => GroupBySelection(
    df, 
    by, 
    undefined, 
    downsample, 
    rule, 
    downsampleN
  );  
  const inspectOpts = {colors:true, depth:null};
  const customInspect = () => util.formatWithOptions(inspectOpts, 'GroupBy {\n    by: %O\n}', by, );

  return  Object.seal(
    Object.assign(
      select, {
        aggList: selectAll().aggList,
        agg: () => {throw todo();},
        count: selectAll().count,
        first: selectAll().first,
        getGroup: () => {throw todo();},
        groups: () => DataFrame[_wrap](df, 'groupby', {by, agg: 'groups'}),
        head: (n=5) => {throw todo();},
        last: selectAll().last,
        max: selectAll().max,
        mean: selectAll().mean,
        median: selectAll().median,
        min: selectAll().min,
        numUnique: selectAll().numUnique,
        pivot,
        quantile: selectAll().quantile,
        sum: selectAll().sum,
        tail: (n=5) => {throw todo();},
        [Symbol.isConcatSpreadable]: true,
        [inspect]: customInspect
      }
    )
  ) as GroupBy;


}

function GroupBySelection(
  df: DataFrame,
  by: string | string[],
  selection?: string[],
  downsample?: boolean,
  rule?: string,
  n?: number,
): GroupBySelection {

  const wrapCall = (agg: string) => () => {
    if(downsample) {
      return DataFrame[_wrap](df, 'downsample', {rule, n, agg});
    } else {
      return DataFrame[_wrap](df, 'groupby', {by, selection, agg});
    }
  };

  const quantile = (quantile:number) => {
    if(downsample) {
      throw new UnsupportedError('quantile', 'downsample');
    } else {
      return DataFrame[_wrap](df, 'groupby', {by, selection, agg: 'quantile', quantile});
    }
  };

  return {
    apply: (fn: (df: DataFrame) => DataFrame) => {throw todo();},
    aggList: wrapCall('agg_list'),
    count: wrapCall('count'),
    first: wrapCall('first'),
    last: wrapCall('last'),
    max: wrapCall('max'),
    mean: wrapCall('mean'),
    median: wrapCall('median'),
    min: wrapCall('min'),
    numUnique: wrapCall('n_unique'),
    quantile,
    sum: wrapCall('sum')
  };
}

function PivotOps(
  df: DataFrame,
  by: string | string[],
  pivotCol: string,
  valueCol: string
): PivotOps {
  
  const pivot =  (agg:string) => () =>  DataFrame[_wrap](df, 'pivot', {by, pivotCol, valueCol, agg});

  return {
    first: pivot('first'),
    sum: pivot('sum'),
    min: pivot('min'),
    max: pivot('max'),
    mean: pivot('mean'),
    count: pivot('count'),
    median: pivot('median'),
  };
}