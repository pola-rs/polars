import {DataType} from "./datatype";
export {DataType};

import pli from "../internals/polars_internal";


export type TypedArray = Int8Array | Int16Array | Int32Array | BigInt64Array | Uint8Array | Uint16Array | Uint32Array | BigInt64Array | Float32Array | Float64Array;


export type Optional<T> = T | undefined | null;
export enum _DataType {
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


export const DTYPE_TO_FFINAME = {
  Int8: "I8",
  Int16: "I16",
  Int32: "I32",
  Int64: "I64",
  UInt8: "U8",
  UInt16: "U16",
  UInt32: "U32",
  UInt64: "U64",
  Float32: "F32",
  Float64: "F64",
  Bool: "Bool",
  Utf8: "Str",
  List: "List",
  Date: "Date",
  Datetime: "Datetime",
  Time: "Time",
  Object: "Object",
  Categorical: "Categorical",
  Struct: "Struct",
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
  const constructor = POLARS_TYPE_TO_CONSTRUCTOR[dtype.variant];
  if (!constructor) {
    throw new Error(`Cannot construct Series for type ${dtype.variant}.`);
  }


  return constructor;
};
