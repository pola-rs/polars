/* eslint-disable no-redeclare */
import {jsTypeToPolarsType} from "./internals/construction";
import {Series, seriesWrapper} from "./series";
import {DataFrame} from "./dataframe";
import pli from "./internals/polars_internal";
import {isDataFrameArray, isSeriesArray} from "./utils";

type ConcatOptions = {rechunk: boolean, how?: "vertical"}

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

export function concat(item: Array<DataFrame>): DataFrame;
export function concat<T>(item: Array<Series<T>>): Series<T>;
export function concat(item: Array<DataFrame>, options: ConcatOptions): DataFrame;
export function concat<T>(item: Array<Series<T>>, options: ConcatOptions): Series<T>;
export function concat<T>(items, options: ConcatOptions =  {rechunk: true, how: "vertical"}): DataFrame | Series<T> {
  const {rechunk, how} = options;

  if(!items.length) {
    throw new RangeError("cannot concat empty list");
  }
  if(how !== "vertical") {
    throw new Error("unsupported operation. only 'vertical' is supported at this time");
  }

  if(isDataFrameArray(items)) {
    const df =  items.reduce((acc, curr) => acc.vstack(curr));

    return rechunk ? df.rechunk() : df;
  }

  if(isSeriesArray<T>(items)) {
    const s =  items.reduce((acc, curr) => acc.concat(curr));

    return rechunk ? s.rechunk() : s;
  }
  throw new Error("can only concat series and dataframes");
}
