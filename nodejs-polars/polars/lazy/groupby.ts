import {DataFrame, dfWrapper} from "../dataframe";
import {Expr, exprToLitOrExpr} from "./expr";
import {ColumnSelection, ColumnsOrExpr, ExpressionSelection, ValueOrArray} from "../utils";
import pli from "../internals/polars_internal";
import {LazyDataFrame} from "./dataframe";
import {todo} from "../internals/utils";

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
  by = by
    .flat(3)
    .map(e => exprToLitOrExpr(e, false)._expr);

  const baseArgs = {by, _ldf, maintainOrder};
  const unwrap = (args) => LazyDataFrame(pli.ldf.groupby(args));
  const agg = (...aggs: Expr[]) => unwrap({
    aggs: aggs.map(a => a._expr),
    aggMethod: "agg",
    ...baseArgs
  });
  const head = (n=5) => unwrap({
    n,
    aggs: [],
    aggMethod: "head",
    ...baseArgs
  });
  const tail = (n=5) => unwrap({
    n,
    aggs: [],
    aggMethod: "tail",
    ...baseArgs
  });

  return {
    agg,
    head,
    tail
  };
};
