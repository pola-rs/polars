import pl from '../';
import polars_internal from '../polars_internal';

import { Dtype, polarsTypeToConstructor } from '../datatypes';
import { isTypedArray } from 'util/types';

const getDtype = (value: unknown): Dtype => {
  if (Array.isArray(value)) {
    throw new Error("List type not yet supported");
  }

  if (value instanceof Date) {
    return Dtype.Date;
  }

  switch (typeof value) {
  case 'bigint':
    return Dtype.UInt64;
  case 'number':
    return Dtype.Float64;
  case 'string':
    return Dtype.Utf8;
  case 'boolean':
    return Dtype.Boolean;
  default:
    return Dtype.Object;
  }
};

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string, values: any[], dtype?: any, strict = true): any {
  const first5TypesMatch = values
    .slice(0, 5)
    .map(getDtype)
    .every((val, _, arr) => val === arr[0]);

  if (!first5TypesMatch) {
    throw new Error('Multi type Series is not supported');
  }

  if (isTypedArray(values)) {
    return polars_internal.series.new_from_typed_array({ name, values, strict });
  }

  //Empty sequence defaults to Float32 type
  if (!values.length && !dtype) {
    dtype = pl.Float32;
  }

  dtype = dtype ?? getDtype(values[0]);

  if (dtype) {
    const constructor = polarsTypeToConstructor(dtype);
    let series = constructor({ name, values, strict });

    if (dtype === pl.Datetime) {
      series = polars_internal.series.cast({ _series: series, dtype, strict: true });
    }

    return series;
  }
}
