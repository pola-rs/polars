import {Expr} from "./expr";
import {selectionToExprList} from "../utils";
import {_LazyDataFrame, LazyDataFrame} from "./dataframe";

export interface LazyGroupBy {
  agg(...aggs: Expr[]): LazyDataFrame
  head(n?: number): LazyDataFrame
  tail(n?: number): LazyDataFrame
}


export const LazyGroupBy = (_lgb: any): LazyGroupBy => {
  return {
    agg(...aggs: Expr[])  {
      const agg  = selectionToExprList(aggs.flat(), false);

      return _LazyDataFrame(_lgb.agg(agg));
    },
    head(n=5) {
      return _LazyDataFrame(_lgb.head(n));
    },
    tail(n=5) {
      return _LazyDataFrame(_lgb.tail(n));

    }
  };
};
