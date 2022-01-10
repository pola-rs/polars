import pli from "./polars_internal";
import { DataType, polarsTypeToConstructor } from "../datatypes";
import { isTypedArray } from "util/types";
import {Series} from "../series";
import {isIterator} from "../utils";

export const jsTypeToPolarsType = (value: unknown): DataType => {
  if(value === null) {
    return DataType.Float64;
  }
  if (Array.isArray(value)) {
    return jsTypeToPolarsType(value[0]);
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
    return DataType.Datetime;
  }

  if(typeof value === "object" && (value as any).constructor === Object) {

    return DataType.Object;
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
    break;
  }

  if (isIterator(value)) {
    return jsTypeToPolarsType(value[0]);
  } else {
    return DataType.Object;
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
const firstNonNull = (iter: Iterable<any>, isIterable = false): {firstValue: any, isIterable: boolean} => {
  let nullCount = 0;
  let firstValue: any = null;
  const iterator = iter[Symbol.iterator]();
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const result = iterator.next();
    if (result.done) break;
    if(result.value === null || result.value === undefined) {
      nullCount++;
    } else {
      firstValue = result.value;
      break;
    }
  }

  if(isTypedArray(firstValue) ) {
    return {firstValue, isIterable: true};
  }

  if(isIterator(firstValue)) {
    const f = firstNonNull(firstValue, true);

    return {
      ...firstNonNull(firstValue, true),
      isIterable: true
    };
  }

  return {firstValue, isIterable};

};

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string, values: Iterable<any>, dtype?: any, strict = false): any {
  if (isTypedArray(values)) {
    return pli.series.new_from_typed_array({ name, values, strict });
  }

  //Empty sequence defaults to Float64 type
  if (!values[0] && !dtype) {
    dtype = DataType.Float64;
  }

  const {firstValue, isIterable} = firstNonNull(values);

  if(isIterable) {
    const listDtype = jsTypeToPolarsType(firstValue);
    const constructor = polarsTypeToConstructor(DataType.List);

    return constructor({ name, values, strict, dtype: listDtype});
  }

  dtype = dtype ?? jsTypeToPolarsType(firstValue);
  let series;

  if(firstValue instanceof Date) {
    series =  pli.series.new_opt_date({name, values, strict});
  } else {
    const constructor = polarsTypeToConstructor(dtype);
    series = constructor({ name, values, strict });
  }
  if ([DataType.Datetime, DataType.Date].includes(dtype)) {
    series = pli.series.cast({ _series: series, dtype, strict: false });
  }
  if(dtype === DataType.Categorical) {
    series = pli.series.cast({ _series: series, dtype, strict: false });
  }

  return series;
}

export function arrayToJsDataFrame(data: any[], columns?: string[], orient?: "col"| "row"): any {
  let dataSeries;

  if(!data.length) {
    dataSeries = [];
  }
  else if (data[0]?._series) {
    dataSeries = [];

    data.forEach((series: Series<any>, idx) => {
      if(!series.name) {
        series.rename(`column_${idx}`, true);
      }
      dataSeries.push(series.inner());
    });
  }
  else if(data[0].constructor.name === "Object") {
    const df = pli.df.read_rows({rows: data});

    if(columns) {
      pli.df.set_column_names({_df: df, names: columns});
    }

    return df;
  }
  else if (Array.isArray(data[0])) {
    if(!orient && columns) {
      orient = columns.length === data.length ? "col" : "row";
    }

    if(orient === "row") {
      const df = pli.df.read_array_rows({data});
      columns && pli.df.set_column_names({_df: df, names: columns});

      return df;
    } else {
      dataSeries = data.map((s, idx) => Series(`column_${idx}`, s).inner());
    }

  }
  else {
    dataSeries = [Series("column_0", data).inner()];
  }
  dataSeries = handleColumnsArg(dataSeries, columns);

  return pli.df.read_columns({columns: dataSeries});
}

function handleColumnsArg(data: Series<any>[], columns?: string[]) {
  if(!columns) {
    return data;
  } else {
    if(!data) {
      return columns.map(c => Series(c, []).inner());
    } else if(data.length === columns.length) {
      columns.forEach((name, i) => {
        pli.series.rename({_series: data[i], name});
      });

      return data;
    }
  }
  throw new TypeError("Dimensions of columns arg must match data dimensions.");
}
