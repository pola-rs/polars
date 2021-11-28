import {Series} from './series';
import {DataFrame} from './dataframe';
import { DataType } from './datatypes';
import * as func from './functions';

export default {
  Int8:  DataType.Int8,
  Int16:  DataType.Int16,
  Int32:  DataType.Int32,
  Int64:  DataType.Int64,
  UInt8:  DataType.UInt8,
  UInt16:  DataType.UInt16,
  UInt32:  DataType.UInt32,
  UInt64:  DataType.UInt64,
  Float32:  DataType.Float32,
  Float64:  DataType.Float64,
  Bool:  DataType.Bool,
  Utf8:  DataType.Utf8,
  List:  DataType.List,
  Date:  DataType.Date,
  Datetime:  DataType.Datetime,
  Time:  DataType.Time,
  Object:  DataType.Object,
  Categorical:  DataType.Categorical,
  repeat: func.repeat,
  readCSV: DataFrame.readCSV,
  readJSON: DataFrame.readJSON,
  concat: func.concat,
  Series,
  NewDataFrame: DataFrame,
  DataFrame: DataFrame.of,
};
