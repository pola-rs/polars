import {JsSeries, Series, seriesWrapper} from "../series";
import {col} from "../lazy/lazy_functions";

export interface ListFunctions<T> {
  get(index: number): Series<T>
  first(): Series<T>
  last(): Series<T>
  /** Get the length of the arrays as UInt32. */
  lengths(): Series<number>
  /** Compute the max value of the arrays in the list */
  max(): Series<number>
  /** Compute the mean value of the arrays in the list */
  mean(): Series<number>
  /** Compute the min value of the arrays in the list */
  min(): Series<number>
  /** Reverse the arrays in the list */
  reverse(): Series<Series<T>>
  /** Sort the arrays in the list */
  sort(reverse?: boolean): Series<Series<T>>
  sort(opt: {reverse: boolean}): Series<Series<T>>
  /** Sum all the arrays in the list */
  sum(): Series<number>
  /** Get the unique/distinct values in the list */
  unique(): Series<Series<T>>
}

export const ListFunctions = <T>(_s: JsSeries): ListFunctions<T> => {
  const callExpr = (method) => (...args) => {
    const s = seriesWrapper(_s);

    return s
      .toFrame()
      .select(
        col(s.name)
          .lst[method](...args)
          .as(s.name)
      )
      .getColumn(s.name);
  };

  return {
    get: callExpr("get"),
    first: callExpr("first"),
    last: callExpr("last"),
    lengths: callExpr("lengths"),
    max: callExpr("max"),
    mean: callExpr("mean"),
    min: callExpr("min"),
    reverse: callExpr("reverse"),
    sort: callExpr("sort"),
    sum: callExpr("sum"),
    unique: callExpr("unique")
  } as any;

};
