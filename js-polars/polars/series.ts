import pl_rs from "./polars_internal";
import util from "util"
export enum DataType {
  Int8 = "Int8",
  Int16 = "Int16",
  Int32 = "Int32",
  Int64 = "Int64",
  UInt8 = "UInt8",
  UInt16 = "UInt16",
  UInt32 = "UInt32",
  UInt64 = "UInt64",
  Float32 = "Float32",
  Float64 = "Float64",
  Bool = "Bool",
  Utf8 = "Utf8",
  List = "List",
  Date = "Date",
  Datetime = "Datetime",
  Time = "Time",
  Object = "Object",
  Categorical = "Categorical",
}
type SeriesInput = {
  name: string, 
  values: any[]
}
type SeriesOptions = {
  type: string, 
}
export default class Series {
  private series: Series;
  static from(data: SeriesInput): Series;
  static from(name: string, values: any[]): Series;
  static from(name: string, values: any[], options: SeriesOptions): Series;

  static from(data: any, values?: any[]) {
    if (values) {
      return new Series(pl_rs.series_read_objects({name: data, values}))
    } else {
      return new Series(pl_rs.series_read_objects(data))
    }
  };

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
  add = (other: Series) => this.of('add', {other: other.series});
  sub = (other: Series) => this.of('sub', {other: other.series});
  mul = (other: Series) => this.of('mul', {other: other.series});
  div = (other: Series) => this.of('div', {other: other.series});

  [util.inspect.custom]() {

    return this.unwrap('get_fmt')
  }
}



