import {Series, _Series} from "./series";
import {col} from "../lazy/functions";
import {ListFunctions} from "../shared_traits";

export type SeriesListFunctions =  ListFunctions<Series>;
export const SeriesListFunctions = (_s): ListFunctions<Series> => {
  const wrap = (method, ...args) => {
    const s = _Series(_s);

    return s
      .toFrame()
      .select(
        col(s.name).lst[method](...args)
          .as(s.name)
      )
      .getColumn(s.name);
  };

  return {
    argMax() {

      return wrap("argMax");
    },
    argMin() {
      return wrap("argMin");
    },
    get(index: number) {
      return wrap("get", index);
    },
    eval(expr, parallel) {
      return wrap("eval", expr, parallel);
    },
    first() {
      return wrap("get", 0);
    },
    join(separator = ",") {
      return wrap("join", separator);
    },
    last() {
      return wrap("get", -1);
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
    shift(n) {
      return wrap("shift", n);
    },
    slice(offset, length) {
      return wrap("slice", offset, length);
    },
    sort(reverse: any = false) {
      return typeof reverse === "boolean" ?
        wrap("sort", reverse) :
        wrap("sort", reverse.reverse);
    },
    sum() {
      return wrap("sum");
    },
    unique() {
      return wrap("unique");
    },
  };
};
