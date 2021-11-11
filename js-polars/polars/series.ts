import pl_rs from "./polars_internal";
import util from "util"

export default class Series {
  private series: Series;
  static from = (data: any) => new Series(pl_rs.series_new(data));
  constructor(series: Series) {
    this.series = series
  }

  private of(method: string, args: Object) {
    return new Series(pl_rs[`series_${method}`]({_series: this.series, ...args}))
  }

  private unwrap(method: string) {
    return pl_rs[`series_${method}`]({_series: this.series})
  }

  head = (length = 5) => this.of('head', {length});
  tail = (length = 5) => this.of('tail', {length});
  add = (other: Series) => this.of('add', {other});
  sub = (other: Series) => this.of('sub', {other});
  mul = (other: Series) => this.of('mul', {other});
  div = (other: Series) => this.of('div', {other});

  [util.inspect.custom]() {

  return this.unwrap('get_fmt')
}
}



