import pli from "./polars_internal";
import { DataType, polarsTypeToConstructor } from "../datatypes";
import { isTypedArray } from "util/types";
import {Series} from "../series/series";
import {_DataFrame} from "../dataframe";
import {TimeUnit} from "../datatypes/datatype";
import {Field} from "../datatypes/field";


export const jsTypeToPolarsType = (value: unknown): DataType => {
  if(value === null) {
    return DataType.Float64;
  }
  if (Array.isArray(value)) {
    return jsTypeToPolarsType(firstNonNull(value));
  }
  if(isTypedArray(value)) {
    switch (value.constructor.name) {
    case Int8Array.name:
      return DataType.Int8;
    case Int16Array.name:
      return DataType.Int16;
    case Int32Array.name:
      return DataType.Int32;
    case BigInt64Array.name:
      return DataType.Int64;
    case Uint8Array.name:
      return DataType.UInt8;
    case Uint16Array.name:
      return DataType.UInt16;
    case Uint32Array.name:
      return DataType.UInt32;
    case BigUint64Array.name:
      return DataType.UInt64;
    case Float32Array.name:
      return DataType.Float32;
    case Float64Array.name:
      return DataType.Float64;
    default:
      throw new Error(`unknown  typed array type: ${value.constructor.name}`);
    }
  }

  if (value instanceof Date) {

    return DataType.Datetime(TimeUnit.Milliseconds);
  }
  if(typeof value === "object" && (value as any).constructor === Object) {
    const flds = Object.entries(value as any).map(([name, value]) => {
      let dtype = jsTypeToPolarsType(value);

      return Field.from(name, dtype);
    });

    return DataType.Struct(flds);
  }

  switch (typeof value) {
  case "bigint":
    return DataType.UInt64;
  case "number":
    return DataType.Float64;
  case "string":
    return DataType.Utf8;
  case "boolean":
    return DataType.Bool;
  default:
    return DataType.Float64;
  }
};

/**
 * __finds the first non null value in the inputs__
 * ___
 * If the first value is an array
 * it will find the first scalar type in the array and return it wrapped into the array
 *
 * @example
 * ```
 * >>> const input = [null, [], [null, "a", "b"]]
 * >>> firstNonNull(input)
 * ["a"]
 * >>> const ints = [null, 1]
 * >>> firstNonNull(ints)
 * 1
 * ```
 */
const firstNonNull = (arr: any[]): any => {
  const first = arr.find(x => x !== null && x !== undefined);
  if(Array.isArray(first)) {
    return [firstNonNull(arr.flat())];
  }

  return first;
};

const fromTypedArray = (name, value) => {
  switch (value.constructor.name) {
  case Int8Array.name:
    return pli.JsSeries.newInt8Array(name, value);
  case Int16Array.name:
    return pli.JsSeries.newInt16Array(name, value);
  case Int32Array.name:
    return pli.JsSeries.newInt32Array(name, value);
  case BigInt64Array.name:
    return pli.JsSeries.newBigint64Array(name, value);
  case Uint8Array.name:
    return pli.JsSeries.newUint8Array(name, value);
  case Uint8ClampedArray.name:
    return pli.JsSeries.newUint8ClampedArray(name, value);
  case Uint16Array.name:
    return pli.JsSeries.newUint16Array(name, value);
  case Uint32Array.name:
    return pli.JsSeries.newUint32Array(name, value);
  case BigUint64Array.name:
    return pli.JsSeries.newBiguint64Array(name, value);
  case Float32Array.name:
    return pli.JsSeries.newFloat32Array(name, value);
  case Float64Array.name:
    return pli.JsSeries.newFloat64Array(name, value);
  default:
    throw new Error(`unknown  typed array type: ${value.constructor.name}`);
  }
};

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string = "", values: any[] = [], dtype?: any, strict = false): any {
  if (isTypedArray(values)) {
    return fromTypedArray(name, values);
  }

  //Empty sequence defaults to Float64 type
  if (!values?.length && !dtype) {
    dtype = DataType.Float64;
  }
  const firstValue = firstNonNull(values);
  if(Array.isArray(firstValue) || isTypedArray(firstValue)) {
    const listDtype = jsTypeToPolarsType(firstValue);

    const constructor = polarsTypeToConstructor(DataType.List(listDtype));

    return constructor(name, values, strict, listDtype);
  }

  dtype = dtype ?? jsTypeToPolarsType(firstValue);
  let series: any;
  if(dtype?.variant === "Struct") {
    const df = pli.fromRows(values, null, 1);

    return df.toStruct(name);
  }
  if(firstValue instanceof Date) {

    series = pli.JsSeries.newOptDate(name, values, strict);
  } else {
    const constructor = polarsTypeToConstructor(dtype);
    series = constructor(name, values, strict);
  }
  if ([
    "Datetime",
    "Date",
    "Categorical",
    "Int8",
    "Int16",
    "UInt8",
    "UInt16",
    "Float32",
  ].includes(dtype.variant)) {
    series = series.cast(dtype, strict);
  }

  return series;
}

export function arrayToJsDataFrame(data: any[], options?): any {
  let columns = options?.columns;
  let orient = options?.orient;


  let dataSeries: any[];

  if(!data.length) {
    dataSeries = [];
  }
  else if (data[0]?._s) {
    dataSeries = [];

    data.forEach((series: any, idx) => {
      if(!series.name) {
        series.rename(`column_${idx}`, true);
      }
      dataSeries.push(series._s);
    });
  }
  else if(data[0].constructor.name === "Object") {

    const df = pli.fromRows( data, options);

    if(columns) {
      df.columns = columns;
    }

    return df;
  }
  else if (Array.isArray(data[0])) {
    if(!orient && columns) {
      orient = columns.length === data.length ? "col" : "row";
    }

    if(orient === "row") {
      const df = pli.fromRows(data);
      columns && (df.columns = columns);

      return df;
    } else {

      dataSeries = data.map((s, idx) => (Series(`column_${idx}`, s) as any)._s);

    }

  }
  else {
    dataSeries = [(Series("column_0", data) as any)._s];
  }
  dataSeries = handleColumnsArg(dataSeries, columns);

  return new pli.JsDataFrame(dataSeries);
}

function handleColumnsArg(data: any[], columns?: string[]) {
  if(!columns) {
    return data;
  } else {
    if(!data) {
      return columns.map(c => (Series.from(c, []) as any)._s);
    } else if(data.length === columns.length) {
      columns.forEach((name, i) => {
        data[i].rename(name);
      });

      return data;
    }
  }
  throw new TypeError("Dimensions of columns arg must match data dimensions.");
}
