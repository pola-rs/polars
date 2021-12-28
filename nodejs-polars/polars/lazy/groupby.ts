import {Expr} from "./expr";
import {selectionToExprList} from "../utils";
import pli from "../internals/polars_internal";
import {LazyDataFrame} from "./dataframe";

export interface LazyGroupBy {
  agg(...aggs: Expr[]): LazyDataFrame
  head(n?: number): LazyDataFrame
  tail(n?: number): LazyDataFrame
}


export const LazyGroupBy = (
  _ldf: any,
  by: any[],
  maintainOrder: boolean
): LazyGroupBy => {
  by = selectionToExprList(by, false);

  const baseArgs = {by, _ldf, maintainOrder};
  const unwrap = (args) => LazyDataFrame(pli.ldf.groupby(args));

  return {
    agg(...aggs: Expr[])  {
      return unwrap({
        aggs: aggs.flatMap(a => a._expr),
        aggMethod: "agg",
        ...baseArgs
      });
    },
    head(n=5) {
      return unwrap({
        n,
        aggs: [],
        aggMethod: "head",
        ...baseArgs
      });
    },
    tail(n=5) {
      return  unwrap({
        n,
        aggs: [],
        aggMethod: "tail",
        ...baseArgs
      });
    }
  };
};
