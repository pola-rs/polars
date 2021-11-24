import pl from '../';
import polars_internal from './polars_internal';

import { DataType, polarsTypeToConstructor } from '../datatypes';
import { isTypedArray } from 'util/types';

export const jsTypeToPolarsType = (value: unknown): DataType => {
  if (Array.isArray(value)) {
    throw new Error("List type not yet supported");
  }

  if (value instanceof Date) {
    return DataType.Date;
  }

  switch (typeof value) {
  case 'bigint':
    return DataType.UInt64;
  case 'number':
    return DataType.Float64;
  case 'string':
    return DataType.Utf8;
  case 'boolean':
    return DataType.Bool;
  default:
    return DataType.Object;
  }
};

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string, values: any[], dtype?: any, strict = true): any {

  if (isTypedArray(values)) {
    return polars_internal.series.new_from_typed_array({ name, values, strict });
  }

  //Empty sequence defaults to Float32 type
  if (!values.length && !dtype) {
    dtype = pl.Float32;
  }

  dtype = dtype ?? jsTypeToPolarsType(values[0]);

  if (dtype) {
    const constructor = polarsTypeToConstructor(dtype);
    let series = constructor({ name, values, strict });

    if (dtype === pl.Datetime) {
      series = polars_internal.series.cast({ _series: series, dtype, strict: true });
    }

    return series;
  }
}
