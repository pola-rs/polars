import {jsTypeToPolarsType} from "./internals/construction";
import pli from "./internals/polars_internal";

export type DtypeToPrimitive<T> = T extends DataType.Bool ? boolean :
 T extends DataType.Utf8 ? string :
 T extends DataType.Categorical ? string :
 T extends DataType.Datetime ? number | Date :
 T extends DataType.Date ? Date :
 T extends DataType.UInt64 ? bigint : number

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

export type Optional<T> = T | undefined | null;
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
  Struct
}


export type JsDataFrame = any;
export type NullValues = string | Array<string> | Record<string, string>;

export type JoinBaseOptions = {
  how?: "left" | "inner" | "outer" | "semi" | "anti" | "cross"
  suffix?: string;
}

export type JoinOptions = {
  leftOn?: string | Array<string>;
  rightOn?: string | Array<string>;
  on?: string | Array<string>;
  how?: "left" | "inner" | "outer" | "semi" | "anti" | "cross"
  suffix?: string;
};


export const DTYPE_TO_FFINAME: Record<DataType, string> = {
  [DataType.Int8]: "I8",
  [DataType.Int16]: "I16",
  [DataType.Int32]: "I32",
  [DataType.Int64]: "I64",
  [DataType.UInt8]: "U8",
  [DataType.UInt16]: "U16",
  [DataType.UInt32]: "U32",
  [DataType.UInt64]: "U64",
  [DataType.Float32]: "F32",
  [DataType.Float64]: "F64",
  [DataType.Bool]: "Bool",
  [DataType.Utf8]: "Str",
  [DataType.List]: "List",
  [DataType.Date]: "Date",
  [DataType.Datetime]: "Datetime",
  [DataType.Time]: "Time",
  [DataType.Object]: "Object",
  [DataType.Categorical]: "Categorical",
  [DataType.Struct]: "Struct",
};

const POLARS_TYPE_TO_CONSTRUCTOR: Record<string, any> = {
  Float32(name, values, strict?) {
    return pli.JsSeries.newOptF64(name, values, strict);
  },
  Float64(name, values, strict?) {
    return pli.JsSeries.newOptF64(name, values, strict);
  },
  Int8(name, values, strict?) {
    return pli.JsSeries.newOptI32(name, values, strict);
  },
  Int16(name, values, strict?) {
    return pli.JsSeries.newOptI32(name, values, strict);
  },
  Int32(name, values, strict?) {
    return pli.JsSeries.newOptI32(name, values, strict);
  },
  Int64(name, values, strict?) {
    return pli.JsSeries.newOptI64(name, values, strict);
  },
  UInt8(name, values, strict?) {
    return pli.JsSeries.newOptU32(name, values, strict);
  },
  UInt16(name, values, strict?) {
    return pli.JsSeries.newOptU32(name, values, strict);
  },
  UInt32(name, values, strict?) {
    return pli.JsSeries.newOptU32(name, values, strict);
  },
  UInt64(name, values, strict?) {
    return pli.JsSeries.newOptU64(name, values, strict);
  },
  Date(name, values, strict?) {
    return pli.JsSeries.newOptI64(name, values, strict);
  },
  Datetime(name, values, strict?) {
    return pli.JsSeries.newOptI64(name, values, strict);
  },
  Bool(name, values, strict?) {
    return pli.JsSeries.newOptBool(name, values, strict);
  },
  Utf8(name, values, strict?) {
    return (pli.JsSeries.newOptStr as any)(name, values, strict);
  },
  Categorical(name, values, strict?) {
    return (pli.JsSeries.newOptStr as any)(name, values, strict);
  },
  List(name, values, _strict, dtype) {
    return pli.JsSeries.newList(name, values, dtype);
  },
};

export const polarsTypeToConstructor = (dtype: DataType): CallableFunction => {
  const constructor = POLARS_TYPE_TO_CONSTRUCTOR[DataType[dtype]];
  if (!constructor) {
    throw new Error(`Cannot construct Series for type ${DataType[dtype]}.`);
  }


  return constructor;
};
