import {jsTypeToPolarsType} from "./internals/construction";
import {Series} from "./series";
import polars_internal from "./internals/polars_internal";

/**
 * _Repeat a single value n times and collect into a Series._
 * @param value - Value to repeat.
 * @param n - Number of repeats
 * @param name - Optional name of the Series
 * @example
 * 
 * ```
 * 
 * > const s = pl.Series("a", [1,2,3])
 * > s.dtype
 * 'Int64Type'
 * 
 * ```
 */
export function repeat<V>(value: V, n: number, name= ""): Series<V>{
  const dtype = jsTypeToPolarsType(value);
  const s = polars_internal.repeat({name, value, dtype, n});

  return new Series(s);
}