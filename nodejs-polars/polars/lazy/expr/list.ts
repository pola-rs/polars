import {Expr, _Expr} from "../expr";
import {ListFunctions} from "../../shared_traits";
/**
 * namespace containing expr list functions
 */
export type ExprList =  ListFunctions<Expr>;
export const ExprListFunctions = (_expr: any): ExprList => {
  const wrap = (method, ...args: any[]): Expr => {
    return _Expr(_expr[method](...args));
  };

  return {
    argMax() {
      return wrap("lstArgMax");
    },
    argMin() {
      return wrap("lstArgMin");
    },
    get(index: number) {
      return wrap("lstGet", index);
    },
    eval(expr, parallel) {
      return wrap("lstEval", expr, parallel);
    },
    first() {
      return wrap("lstGet", 0);
    },
    join(separator = ",") {
      return wrap("lstJoin", separator);
    },
    last() {
      return wrap("lstGet", -1);
    },
    lengths() {
      return wrap("lstLengths");
    },
    max() {
      return wrap("lstMax");
    },
    mean() {
      return wrap("lstMean");
    },
    min() {
      return wrap("lstMin");
    },
    reverse() {
      return wrap("lstReverse");
    },
    shift(n) {
      return wrap("lstShift", n);
    },
    slice(offset, length) {
      return wrap("lstSlice", offset, length);
    },
    sort(reverse: any = false) {
      return typeof reverse === "boolean" ?
        wrap("lstSort", reverse) :
        wrap("lstSort", reverse.reverse);
    },
    sum() {
      return wrap("lstSum");
    },
    unique() {
      return wrap("lstUnique");
    },
  };
};
