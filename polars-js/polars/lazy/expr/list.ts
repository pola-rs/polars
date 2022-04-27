import {Expr} from "../expr";
import pli from "../../internals/polars_internal";

/**
 * namespace containing expr list functions
 */
export interface ExprList {
  /**
   * Get the value by index in the sublists.
   * So index `0` would return the first item of every sublist
   * and index `-1` would return the last item of every sublist
   * if an index is out of bounds, it will return a `null`.
   */
  get(index: number): Expr
  /** Get the first value of the sublists. */
  first(): Expr
  /**
   * Join all string items in a sublist and place a separator between them.
   * This errors if inner type of list `!= Utf8`.
   * @param separator A string used to separate one element of the list from the next in the resulting string.
   * If omitted, the list elements are separated with a comma.
   */
  join(separator?: string): Expr
  /** Get the last value of the sublists. */
  last(): Expr
  lengths(): Expr;
  max(): Expr;
  mean(): Expr;
  min(): Expr;
  reverse(): Expr;
  sort(reverse?: boolean): Expr;
  sort(opt: {reverse: boolean}): Expr;
  sum(): Expr;
  unique(): Expr;
}

export const ExprListFunctions = (_expr: any): ExprList => {
  const wrap = (method, args?): Expr => {

    return (Expr as any)(pli.expr.lst[method]({_expr, ...args }));
  };

  return {
    get(index: number) {
      return wrap("get", {index});
    },
    first() {
      return wrap("get", {index:0});
    },
    join(separator = ",") {
      return wrap("join", {separator});
    },
    last() {
      return wrap("get", {index:-1});
    },
    lengths() {
      return wrap("lengths");
    },
    max() {
      return wrap("max");
    },
    mean() {
      return wrap("mean");
    },
    min() {
      return wrap("min");
    },
    reverse() {
      return wrap("reverse");
    },
    sort(reverse: any = false) {
      return typeof reverse === "boolean" ?
        wrap("sort", {reverse}) :
        wrap("sort", reverse);
    },
    sum() {
      return wrap("sum");
    },
    unique() {
      return wrap("unique");
    },
  };
};
