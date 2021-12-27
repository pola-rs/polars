/* eslint-disable no-redeclare */
import {jsTypeToPolarsType} from "./internals/construction";
import {Series, seriesWrapper} from "./series";
import {DataFrame, dfWrapper} from "./dataframe";
import pli from "./internals/polars_internal";
import {isDataFrameArray, isSeriesArray} from "./utils";

type ConcatOptions = {rechunk?: boolean, how?: "vertical" | "horizontal"}

/**
 * _Repeat a single value n times and collect into a Series._
 * @param value - Value to repeat.
 * @param n - Number of repeats
 * @param name - Optional name of the Series
 * @example
 *
 * ```
 *
 * > const s = pl.repeat("a", 5)
 * > s.toArray()
 * ["a", "a", "a", "a", "a"]
 *
 * ```
 */
export function repeat<V>(value: V, n: number, name= ""): Series<V>{
  const dtype = jsTypeToPolarsType(value);
  const s = pli.repeat({name, value, dtype, n});

  return seriesWrapper(s);
}

/**
 * Aggregate all the Dataframes/Series in a List of DataFrames/Series to a single DataFrame/Series.
 * @param items DataFrames/Series/LazyFrames to concatenate.
 * @param options.rechunk rechunk the final DataFrame/Series.
 * @param options.how Only used if the items are DataFrames. *Defaults to 'vertical'*
 *     - Vertical: Applies multiple `vstack` operations.
 *     - Horizontal: Stacks Series horizontall and fills with nulls if the lengths don't match.
 *
 * @example
 * >>> const df1 = pl.DataFrame({"a": [1], "b": [3]})
 * >>> const df2 = pl.DataFrame({"a": [2], "b": [4]})
 * >>> pl.concat([df1, df2])
 * shape: (2, 2)
 * ┌─────┬─────┐
 * │ a   ┆ b   │
 * │ --- ┆ --- │
 * │ i64 ┆ i64 │
 * ╞═════╪═════╡
 * │ 1   ┆ 3   │
 * ├╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 2   ┆ 4   │
 * └─────┴─────┘
 */
export function concat(items: Array<DataFrame>, options?: ConcatOptions): DataFrame;
export function concat<T>(items: Array<Series<T>>, options?: {rechunk: boolean}): Series<T>;
export function concat<T>(items, options: ConcatOptions =  {rechunk: true, how: "vertical"}) {
  const {rechunk, how} = options;

  if(!items.length) {
    throw new RangeError("cannot concat empty list");
  }

  if(isDataFrameArray(items)) {
    let df;
    if(how === "vertical") {
      df =  items.reduce((acc, curr) => acc.vstack(curr));

    } else {
      df = dfWrapper(pli.horizontalConcatDF({items: items.map(i => i._df)}));
    }

    return rechunk ? df.rechunk() : df;
  }

  if(isSeriesArray<T>(items)) {
    const s =  items.reduce((acc, curr) => acc.concat(curr));

    return rechunk ? s.rechunk() : s;
  }
  throw new TypeError("can only concat series and dataframes");
}
