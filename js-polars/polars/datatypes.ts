import polars_internal from "./polars_internal";

export enum Dtype {
  Int8 = 'Int8',
  Int16 = 'Int16',
  Int32 = 'Int32',
  Int64 = 'Int64',
  UInt8 = 'UInt8',
  UInt16 = 'UInt16',
  UInt32 = 'UInt32',
  UInt64 = 'UInt64',
  Float32 = 'Float32',
  Float64 = 'Float64',
  Boolean = 'Boolean',
  Utf8 = 'Utf8',
  List = 'List',
  Date = 'Date',
  Datetime = 'Datetime',
  Time = 'Time',
  Object = 'Object',
  Categorical = 'Categorical',
}


export type DataType = {
  Int8: number,
  Int16: number,
  Int32: number,
  Int64: number,
  UInt8: number,
  UInt16: number,
  UInt32: number,
  UInt64: number,
  Float32: number,
  Float64: number,
  Boolean: boolean,
  Utf8: string,
  List: Array<any>, // todo add better typing for list types
  Date: Date,
  Datetime: number,
  Time: number,
  Object: object,
  Categorical: string,
}

export const dtypeToU8Value = (dtype: Dtype): number => {
  return Object.keys(Dtype).indexOf(dtype)
}


export const DTYPE_TO_FFINAME: Record<Dtype, string> = {
  [Dtype.Int8]: "i8",
  [Dtype.Int16]: "i16",
  [Dtype.Int32]: "i32",
  [Dtype.Int64]: "i64",
  [Dtype.UInt8]: "u8",
  [Dtype.UInt16]: "u16",
  [Dtype.UInt32]: "u32",
  [Dtype.UInt64]: "u64",
  [Dtype.Float32]: "f32",
  [Dtype.Float64]: "f64",
  [Dtype.Boolean]: "bool",
  [Dtype.Utf8]: "str",
  [Dtype.List]: "list",
  [Dtype.Date]: "date",
  [Dtype.Datetime]: "datetime",
  [Dtype.Time]: "time",
  [Dtype.Object]: "object",
  [Dtype.Categorical]: "categorical",
}

const POLARS_TYPE_TO_CONSTRUCTOR: Record<string, string> = {
  Float32: "new_opt_f32",
  Float64: "new_opt_f64",
  Int8: "new_opt_i8",
  Int16: "new_opt_i16",
  Int32: "new_opt_i32",
  Int64: "new_opt_i64",
  UInt8: "new_opt_u8",
  UInt16: "new_opt_u16",
  UInt32: "new_opt_u32",
  UInt64: "new_opt_u64",
  Date: "new_opt_date",
  Datetime: "new_opt_date",
  Boolean: "new_opt_bool",
  Utf8: "new_str",
  Object: "new_object",
}

export const polarsTypeToConstructor = (dtype: Dtype): CallableFunction => {
  const constructor = POLARS_TYPE_TO_CONSTRUCTOR[Dtype[dtype]];
  if (!constructor) {
    throw new Error(`Cannot construct Series for type ${Dtype[dtype]}.`)
  }
  return polars_internal.series[constructor];
}
