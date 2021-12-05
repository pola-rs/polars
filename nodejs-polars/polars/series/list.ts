import pli from "../internals/polars_internal";
import {ValueOrArray} from "../utils";
import {JsSeries, Series, seriesWrapper} from "../series";
import {todo} from "../internals/utils";
import {col} from "../lazy/lazy_functions";

export interface ListFunctions {
  /**
   * Concat the arrays in a Series dtype List in linear time.
   * @param other Series<any[]> | Series<any[]>[] | ...Series<any[]>
   * @example
   * ```
   * > series.list.concat(other)
   * > series.list.concat(other, other2)
   * > series.list.concat([other, other2])
   *```
   */
  concat(...other: ValueOrArray<Series<any>>[]): Series<any>
  /**
   * Get the length of the arrays as UInt32.
   */
  lengths(): Series<number>
  /**
   * Compute the max value of the arrays in the list
   */
  max(): Series<any>
  /**
   * Compute the mean value of the arrays in the list
   */
  mean(): Series<any>
  /**
   * Compute the min value of the arrays in the list
   */
  min(): Series<any>
  /**
   * Reverse the arrays in the list
   */
  reverse(): Series<any>
  /**
   *Sort the arrays in the list
   */
  sort(reverse?: boolean): Series<any>
  /**
   *
   */
  sum(): Series<any>
  /**
   * Get the unique/distinct values in the list
   */
  unique(): Series<any>
}

export const ListFunctions = (_s: JsSeries): ListFunctions => {
  const callMethod = (method) => (...args) => {
    const s = seriesWrapper(_s);

    return s.toFrame().select(col(s.name).lst[method](...args)) as any;
  };

  return {
    concat: (...others) => {throw todo();},
    lengths: () => seriesWrapper(pli.series.arr_lengths({_series: _s})),
    max: callMethod("max"),
    mean: callMethod("mean"),
    min: callMethod("min"),
    reverse: callMethod("reverse"),
    sort: callMethod("sort"),
    sum: callMethod("sum"),
    unique: callMethod("unique")
  };

};
