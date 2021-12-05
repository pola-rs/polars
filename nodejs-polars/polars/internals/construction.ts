import pli from "./polars_internal";
import { DataType, polarsTypeToConstructor } from "../datatypes";
import { isTypedArray } from "util/types";
import {Series, seriesWrapper} from "../series";

export const jsTypeToPolarsType = (value: unknown): DataType => {
  if (Array.isArray(value)) {
    return jsTypeToPolarsType(value[0]);
  }

  if (value instanceof Date) {
    return DataType.Datetime;
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

const firstNonNull = (arr: any[]): any => arr.find(x => x !== null && x !== undefined);
/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string, values: any[], dtype?: any, strict = false): any {
  if (isTypedArray(values)) {
    return pli.series.new_from_typed_array({ name, values, strict });
  }

  //Empty sequence defaults to Float32 type
  if (!values.length && !dtype) {
    dtype = DataType.Float32;
  }
  const firstValue = firstNonNull(values);
  if(Array.isArray(firstValue)) {
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