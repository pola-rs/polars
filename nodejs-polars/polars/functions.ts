/* eslint-disable no-redeclare */
import {jsTypeToPolarsType} from "./internals/construction";
import {Series, _wrapSeries} from "./series";
import {DataFrame} from "./dataframe";
import polars_internal from "./internals/polars_internal";


type ConcatItems = Array<DataFrame> | Array<Series<any>>
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
 * > s.dtype
 * 'Utf8Type'
 *
 * ```
 */
export function repeat<V>(value: V, n: number, name= ""): Series<V>{
  const dtype = jsTypeToPolarsType(value);
  const s = polars_internal.repeat({name, value, dtype, n});

  return _wrapSeries(s);
}

export function concat(item: Array<DataFrame>): DataFrame;
export function concat(item: Array<Series<any>>): Series<any>;
export function concat(items: ConcatItems, options?: ConcatOptions): DataFrame | Series<any> {
  const {rechunk, how} = {rechunk: true, how: "vertical", ...options};

  if(!items) {
    throw new RangeError("cannot concat empty list");
  }

  if((items[0] as any)?._df) {
    if(how === "vertical") {
      let df = items.shift() as DataFrame;

      items.forEach(other => {
        polars_internal.df.vstack({_df: df.inner(), other: other.inner(), in_place: true});
      });

      return rechunk ? df.rechunk() : df;
    } else {
      throw new Error("unsupported operation. only 'vertical' is supported at this time");
    }
  }

  if((items[0] as any)?._series) {
    const s =  (items as Series<any>[]).reduce((acc,curr) => acc.concat(curr));

    return rechunk ? s.rechunk() : s;
  }
  throw new Error();
}