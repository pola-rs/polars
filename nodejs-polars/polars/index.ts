import {Series} from "./series";
import {DataFrame, readCSV, readJSON} from "./dataframe";
import { DataType } from "./datatypes";
import * as func from "./functions";

export default {
  Int8:  DataType.Int8 as DataType.Int8,
  Int16:  DataType.Int16 as DataType.Int16,
  Int32:  DataType.Int32 as DataType.Int32,
  Int64:  DataType.Int64 as DataType.Int64,
  UInt8:  DataType.UInt8 as DataType.UInt8,
  UInt16:  DataType.UInt16 as DataType.UInt16,
  UInt32:  DataType.UInt32 as DataType.UInt32,
  UInt64:  DataType.UInt64 as DataType.UInt64,
  Float32:  DataType.Float32 as DataType.Float32,
  Float64:  DataType.Float64 as DataType.Float64,
  Bool:  DataType.Bool as DataType.Bool,
  Utf8:  DataType.Utf8 as DataType.Utf8,
  List:  DataType.List as DataType.List,
  Date:  DataType.Date as DataType.Date,
  Datetime:  DataType.Datetime as DataType.Datetime,
  Time:  DataType.Time as DataType.Time,
  Object:  DataType.Object as DataType.Object,
  Categorical:  DataType.Categorical as DataType.Categorical,
  repeat: func.repeat,
  readCSV,
  readJSON,
  concat: func.concat,
  Series,
  DataFrame,
};
