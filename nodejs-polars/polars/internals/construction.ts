import polars_internal from "./polars_internal";
import { DataType, polarsTypeToConstructor } from "../datatypes";
import { isTypedArray } from "util/types";
import {Series, _wrapSeries} from "../series";

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
    return DataType.Object;
  }
};

/**
 * Construct an internal `JsSeries` from an array
 */
export function arrayToJsSeries(name: string, values: any[], dtype?: any, strict = false): any {
  if (isTypedArray(values)) {
    return polars_internal.series.new_from_typed_array({ name, values, strict });
  }

  //Empty sequence defaults to Float32 type
  if (!values.length && !dtype) {
    dtype = DataType.Float32;
  }
  if(Array.isArray(values[0])) {
    const listDtype = jsTypeToPolarsType(values[0]);
    const constructor = polarsTypeToConstructor(DataType.List);

    return constructor({ name, values, strict, dtype: listDtype});
  }

  dtype = dtype ?? jsTypeToPolarsType(values[0]);
  let series;

  if(values[0] instanceof Date) {
    series =  polars_internal.series.new_opt_date({name, values, strict});
  } else {

    const constructor = polarsTypeToConstructor(dtype);
    series = constructor({ name, values, strict });
  }

  if ([DataType.Datetime, DataType.Date].includes(dtype)) {
    series = polars_internal.series.cast({ _series: series, dtype, strict: true });
  }


  return series;
}

export function arrayToJsDataFrame(data: any[], columns?: string[], orient?: "col"| "row"): any {
  let dataSeries;

  if(!data.length) {
    dataSeries = [];
  }
  else if (data[0] instanceof Series) {
    dataSeries = [];

    data.forEach((series: Series<any>, idx) => {
      if(!series.name) {
        series.rename(`column_${idx}`, true);
      }
      dataSeries.push(series.inner());
    });
  }
  else if(data[0].constructor.name === "Object") {
    const df = polars_internal.df.read_rows({rows: data});

    if(columns) {
      polars_internal.df.set_column_names({_df: df, names: columns});
    }

    return df;
  }
  else if (Array.isArray(data[0])) {
    if(!orient && columns) {
      orient = columns.length === data.length ? "col" : "row";
    }

    if(orient === "row") {
      const df = polars_internal.df.read_array_rows({data});
      columns && polars_internal.df.set_column_names({_df: df, names: columns});

      return df;
    } else {
      dataSeries = data.map((s,idx) => Series(`column_${idx}`, s).inner());
    }

  }
  else {
    dataSeries = [Series("column_0", data).inner()];
  }
  dataSeries = handleColumnsArg(dataSeries, columns);

  return polars_internal.df.read_columns({columns: dataSeries});
}

function handleColumnsArg(data: Series<any>[], columns?: string[]) {
  if(!columns) {
    return data;
  } else {
    if(!data) {
      return columns.map(c => Series(c, []).inner());
    } else if(data.length === columns.length) {
      columns.forEach((name, i) => {
        polars_internal.series.rename({_series: data[i], name});
      });

      return data;
    }
  }
  throw new TypeError("Dimensions of columns arg must match data dimensions.");
}