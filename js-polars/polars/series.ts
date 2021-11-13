import pl_rs from "./polars_internal";
import util from "util"
import {PolarsDataType} from './datatypes';

type SeriesInput = {
  name: string,
  values: any[]
}

const typeToInternal = {
  [PolarsDataType.Int8]: "series_new_i8",
  [PolarsDataType.Int16]: "series_new_i16",
  [PolarsDataType.Int32]: "series_new_i32",
  [PolarsDataType.Int64]: "series_new_i64",
  [PolarsDataType.UInt8]: "series_new_u8",
  [PolarsDataType.UInt16]: "series_new_u16",
  [PolarsDataType.UInt32]: "series_new_u32",
  [PolarsDataType.UInt64]: "series_new_u64",
  [PolarsDataType.Float32]: "series_new_f32",
  [PolarsDataType.Float64]: "series_new_f64",
  [PolarsDataType.Bool]: "series_new_bool",
  [PolarsDataType.Utf8]: "series_read_objects",
  [PolarsDataType.List]: "series_read_objects",
  [PolarsDataType.Date]: "series_read_objects",
  [PolarsDataType.Datetime]: "series_read_objects",
  [PolarsDataType.Time]: "series_read_objects",
  [PolarsDataType.Object]: "series_read_objects",
  [PolarsDataType.Categorical]: "series_read_objects",
}
const typeToOptInternal = {
  [PolarsDataType.Int8]: "series_new_opt_i8",
  [PolarsDataType.Int16]: "series_new_opt_i16",
  [PolarsDataType.Int32]: "series_new_opt_i32",
  [PolarsDataType.Int64]: "series_new_opt_i64",
  [PolarsDataType.UInt8]: "series_new_opt_u8",
  [PolarsDataType.UInt16]: "series_new_opt_u16",
  [PolarsDataType.UInt32]: "series_new_opt_u32",
  [PolarsDataType.UInt64]: "series_new_opt_u64",
  [PolarsDataType.Float32]: "series_new_opt_f32",
  [PolarsDataType.Float64]: "series_new_opt_f64",
  [PolarsDataType.Bool]: "series_new_opt_bool",
  [PolarsDataType.Utf8]: "series_new_str",
  [PolarsDataType.List]: "series_read_objects",
  [PolarsDataType.Date]: "series_read_objects",
  [PolarsDataType.Datetime]: "series_read_objects",
  [PolarsDataType.Time]: "series_read_objects",
  [PolarsDataType.Object]: "series_read_objects",
  [PolarsDataType.Categorical]: "series_read_objects",
}

type SeriesOptions = {
  type: PolarsDataType,
  strict?: boolean
}
const defaultOptions = {type: PolarsDataType.Float64, strict: true}
export default class Series {
  private series: Series;
  static from(data: SeriesInput): Series;
  static from(name: string, values: any[]): Series;
  static from(name: string, values: any[], options: SeriesOptions): Series;
  static from(data: string | SeriesInput, values?: any[], options?: SeriesOptions) {
    const optsWithDefaults = {...defaultOptions, ...options}

    const fn = typeToOptInternal[optsWithDefaults.type];
    if (typeof data !== "string") {
      return new Series(pl_rs[fn]({
        ...data,
        ...optsWithDefaults
      }))
    } else {

      return new Series(pl_rs[fn]({name: data, values, ...optsWithDefaults}))
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

  name = () => this.unwrap('name')
  rename = (name: string) => this.of('rename', {name})
  cumsum = () => this.unwrap('cumsum')
  cummin = () => this.unwrap('cummin')
  cummax = () => this.unwrap('cummax')
  cumprod = () => this.unwrap('cumprod')
  dtype = () => PolarsDataType[this.unwrap("dtype")]
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



