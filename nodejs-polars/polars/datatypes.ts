import pli from "./internals/polars_internal";

export type DtypeToPrimitive<T> = T extends DataType.Bool ? boolean :
 T extends DataType.Utf8 ? string :
 T extends DataType.Categorical ? string :
 T extends DataType.Datetime ? number | Date :
 T extends DataType.Date ? Date :
 T extends DataType.UInt64 ? bigint :
 T extends DataType.Int64 ? bigint : number

export type PrimitiveToDtype<T> = T extends boolean ? DataType.Bool :
 T extends string ? DataType.Utf8 :
 T extends Date ? DataType.Datetime :
 T extends number ? DataType.Float64 :
 T extends bigint ? DataType.Int64 :
 T extends ArrayLike<any> ? DataType.List :
 DataType.Object

export type TypedArray = Int8Array | Int16Array | Int32Array | BigInt64Array | Uint8Array | Uint16Array | Uint32Array | BigInt64Array | Float32Array | Float64Array;

export type DtypeToTypedArray<T> = T extends DataType.Int8 ? Int8Array :
T extends DataType.Int16 ? Int16Array :
T extends DataType.Int32 ? Int32Array :
T extends DataType.Int64 ? BigInt64Array :
T extends DataType.UInt8 ? Uint8Array :
T extends DataType.UInt16 ? Uint16Array :
T extends DataType.UInt32 ? Uint32Array :
T extends DataType.UInt64 ? BigInt64Array :
T extends DataType.Float32 ? Float32Array :
T extends DataType.Float64 ? Float64Array :
never

export type Optional<T> = T | undefined;

export enum DataType {
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float32,
  Float64,
  Bool,
  Utf8,
  List,
  Date,
  Datetime,
  Time,
  Object,
  Categorical,
}


export type JsDataFrame = any;
export type NullValues = string | Array<string> | Record<string, string>;
export type ReadCsvOptions = {
  batchSize?: number;
  columns?: Array<string>;
  commentChar?: string;
  encoding?: "utf8" | "utf8-lossy";
  endRows?: number;
  hasHeader?: boolean;
  ignoreErrors?: boolean;
  inferSchemaLength?: number;
  lowMemory?: boolean;
  nullValues?: NullValues;
  numThreads?: number;
  parseDates?: boolean;
  projection?: Array<number>;
  quoteChar?: string;
  rechunk?: boolean;
  sep?: string;
  startRows?: number;
};
export type ReadJsonOptions = {
  inferSchemaLength?: number;
  batchSize?: number;
};

export type ReadParquetOptions = {
  columns?: string[];
  projection?: number[];
  numRows?: number;
  parallel?: boolean;
  rechunk?: boolean;
}
export type JoinBaseOptions = {
  how?: "left" | "inner" | "outer" | "cross";
  suffix?: string;
}
export type JoinOptions = {
  leftOn?: string | Array<string>;
  rightOn?: string | Array<string>;
  on?: string | Array<string>;
  how?: "left" | "inner" | "outer" | "cross";
  suffix?: string;
};
export type WriteCsvOptions = {
  hasHeader?: boolean;
  sep?: string;
};
export type WriteJsonOptions = {
  orient?: "row" | "col" | "dataframe";
  multiline?: boolean;
};


export const DTYPE_TO_FFINAME: Record<DataType, string> = {
  [DataType.Int8]: "i8",
  [DataType.Int16]: "i16",
  [DataType.Int32]: "i32",
  [DataType.Int64]: "i64",
  [DataType.UInt8]: "u8",
  [DataType.UInt16]: "u16",
  [DataType.UInt32]: "u32",
  [DataType.UInt64]: "u64",
  [DataType.Float32]: "f32",
  [DataType.Float64]: "f64",
  [DataType.Bool]: "bool",
  [DataType.Utf8]: "str",
  [DataType.List]: "list",
  [DataType.Date]: "date",
  [DataType.Datetime]: "datetime",
  [DataType.Time]: "time",
  [DataType.Object]: "object",
  [DataType.Categorical]: "categorical",
};

export const iterToTypedArray = (dtype: DataType, iterator: any): Iterable<any>  => {
  switch (dtype) {
  case DataType.Int8:
    return Int8Array.from(iterator);
  case DataType.Int16:
    return Int16Array.from(iterator);
  case DataType.Int32:
    return Int32Array.from(iterator);
  case DataType.Int64:
    return BigInt64Array.from(iterator);
  case DataType.UInt8:
    return Uint8Array.from(iterator);
  case DataType.UInt16:
    return Uint16Array.from(iterator);
  case DataType.UInt32:
    return Uint32Array.from(iterator);
  case DataType.UInt64:
    return BigUint64Array.from(iterator);
  case DataType.Float32:
    return Float32Array.from(iterator);
  case DataType.Float64:
    return Float64Array.from(iterator);
  default:
    return iterator;
  }
};

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
  Date: "new_opt_u32",
  Datetime: "new_opt_u32",
  Bool: "new_opt_bool",
  Utf8: "new_str",
  Categorical: "new_str",
  Object: "new_object",
  List: "new_list",
};

export const polarsTypeToConstructor = (dtype: DataType): CallableFunction => {
  const constructor = POLARS_TYPE_TO_CONSTRUCTOR[DataType[dtype]];
  if (!constructor) {
    throw new Error(`Cannot construct Series for type ${DataType[dtype]}.`);
  }

  return pli.series[constructor];
};
