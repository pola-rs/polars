import pl_rs from "./polars_internal";
import util from "util"
import {PolarsDataType} from './datatypes';

type SeriesInput = {
  name: string,
  values: any[]
}
const typeToInternalFn = {
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
  [PolarsDataType.Date]: "series_new_opt_date",
  [PolarsDataType.Object]: "series_new_object",

  // TODO:
  [PolarsDataType.List]: "null",
  [PolarsDataType.Datetime]: "null",
  [PolarsDataType.Time]: "null",
  [PolarsDataType.Categorical]: "null",
}

type SeriesOptions = {
  type: PolarsDataType,
  strict?: boolean
}
// look at the first item in array to determine type
const getInferredType = (value: unknown): PolarsDataType => {
  if (Array.isArray(value)) {
    return PolarsDataType.List
  }
  if (value instanceof Date) {
    return PolarsDataType.Date
  }
  switch (typeof value) {
    case "number": return PolarsDataType.Float64;
    case "string": return PolarsDataType.Utf8;
    case "boolean": return PolarsDataType.Bool;
    default: return PolarsDataType.Object;
  }
}
const defaultOptions = {strict: true}
export default class Series {
  _series: Series;
  
  static from(data: SeriesInput): Series;
  static from(name: string, values: any[]): Series;
  static from(name: string, values: any[], options: SeriesOptions): Series;
  static from(data: string | SeriesInput, values?: any[], options?: SeriesOptions) {
    const type = getInferredType(values?.[0]);
    const optsWithDefaults = {...defaultOptions, ...options, type}

    console.log({type: PolarsDataType[type], value: values?.[0]})


    const fn = typeToInternalFn[optsWithDefaults.type];
    console.log(fn)
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

    this._series = series
  }

  private of(method: string, args: Object) {
    return new Series(pl_rs[`series_${method}`]({_series: this._series, ...args}))
  }

  private unwrap(method: string) {
    return pl_rs[`series_${method}`]({_series: this._series})
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
  add = (other: Series) => this.of('add', {other: other._series});
  sub = (other: Series) => this.of('sub', {other: other._series});
  mul = (other: Series) => this.of('mul', {other: other._series});
  div = (other: Series) => this.of('div', {other: other._series});

  [util.inspect.custom]() {

    return this.unwrap('get_fmt')
  }
}



