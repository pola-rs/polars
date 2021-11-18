import pl from '../';
import polars_internal from "../polars_internal";

import {Dtype, polarsTypeToConstructor} from "../datatypes";
import {JsSeries} from '../series';


const getInferredType = (value: unknown): Dtype => {
  if (Array.isArray(value)) {
    return Dtype.List
  }
  if (value instanceof Date) {
    return Dtype.Date
  }
  switch (typeof value) {
    case "number": return Dtype.Float64;
    case "string": return Dtype.Utf8;
    case "boolean": return Dtype.Boolean;
    default: return Dtype.Object;
  }
}

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(
  name: string,
  values: any[],
  dtype?: any,
  strict = true
): JsSeries {
  //Empty sequence defaults to Float32 type
  if (!values.length && !dtype) {
    dtype = pl.Float32
  }
  dtype = dtype ?? getInferredType(values[0])


  if (dtype) {
    const constructor = polarsTypeToConstructor(dtype);
    let series = constructor({name, values, strict})
    if (dtype === pl.Datetime) {
      series = polars_internal.series.cast({_series: series, dtype, strict: true})
    }

    return series
  }

}