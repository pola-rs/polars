import {Expr} from "./expr";
import {ColumnsOrExpr, selectionToExprList} from "../utils";
import {_LazyDataFrame, LazyDataFrame} from "./dataframe";
import {DataFrame} from "@polars/dataframe";

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

export interface RollingGroupBy {
  agg(column: ColumnsOrExpr): LazyDataFrame
}

export function RollingGroupBy(
  df: any,
  indexColumn: string,
  period: string,
  offset?: string,
  closed = "none",
  by?: ColumnsOrExpr): RollingGroupBy {
  return {
    agg(column: ColumnsOrExpr) {
      const exprs = selectionToExprList([by]);

      return df
        .groupbyRolling(indexColumn, period, offset, closed, exprs)
        .agg(column)
        .collectSync();
    }
  };
}
