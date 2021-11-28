import pl from '../';
import polars_internal from './polars_internal';

import { DataType, polarsTypeToConstructor } from '../datatypes';
import { isTypedArray } from 'util/types';
import {Series} from '../series';

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

export function arrayToJsDataFrame(data: any[], columns?: string[], orient?: 'col'| 'row'): any {
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
  else if(data[0].constructor.name === 'Object') {
    const df = polars_internal.df.read_rows({rows: data});

    if(columns) {
      polars_internal.df.set_column_names({_df: df, names: columns});
    }

    return df;
  }
  else if (Array.isArray(data[0])) {
    if(!orient && columns) {
      orient = columns.length === data.length ? 'col' : 'row';
    }

    if(orient === 'row') {
      const df = polars_internal.df.read_array_rows({data});
      columns && polars_internal.df.set_column_names({_df: df, names: columns});

      return df; 
    } else {
      dataSeries = data.map((s,idx) => Series.of(`column_${idx}`, s).inner());
    }
    
  }
  else {
    dataSeries = [Series.of("column_0", data).inner()];
  }
  dataSeries = handleColumnsArg(dataSeries, columns);

  return polars_internal.df.read_columns({columns: dataSeries});
}

function handleColumnsArg(data: Series<any>[], columns?: string[]) {
  if(!columns) {
    return data;
  } else {
    if(!data) {
      return columns.map(c => Series.of(c, []).inner());
    } else if(data.length === columns.length) {
      columns.forEach((name, i) => {
        polars_internal.series.rename({_series: data[i], name});
      });

      return data;
    }
  }
  throw new TypeError("Dimensions of columns arg must match data dimensions.");
}