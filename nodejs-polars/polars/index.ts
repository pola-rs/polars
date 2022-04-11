import * as series from "./series/series";
import * as df from "./dataframe";
import { DataType } from "./datatypes";
import * as func from "./functions";
import * as io from "./io";
import * as cfg from "./cfg";
import {version as _version} from "../package.json";

import type { FillNullStrategy as _FillNullStrategy } from "./utils";
import  {
  funcs as lazy,
  Expr as lazyExpr,
  GroupBy as lazyGroupBy,
  when as _when
} from "./lazy";


namespace pl {
  export import Expr = lazyExpr.Expr
  export import DataFrame = df.DataFrame
  export import Series = series.Series;
  export type LazyGroupBy = lazyGroupBy;
  export type When = _when.When;
  export type WhenThen = _when.WhenThen;
  export type WhenThenThen = _when.WhenThenThen;
  export type FillNullStrategy = _FillNullStrategy;
  export import Config = cfg.Config;
  export import Int8 = DataType.Int8
  export import Int16 = DataType.Int16
  export import Int32 =  DataType.Int32;
  export import Int64 =  DataType.Int64;
  export import UInt8 =  DataType.UInt8;
  export import UInt16 =  DataType.UInt16;
  export import UInt32 =  DataType.UInt32;
  export import UInt64 =  DataType.UInt64;
  export import Float32 =  DataType.Float32;
  export import Float64 =  DataType.Float64;
  export import Bool =  DataType.Bool;
  export import Utf8 =  DataType.Utf8;
  export import List =  DataType.List;
  export import Date = DataType.Date;
  export import Datetime = DataType.Datetime;
  export import Time = DataType.Time;
  export import Object = DataType.Object;
  export import Categorical = DataType.Categorical;
  export import repeat =  func.repeat;
  export import concat =  func.concat;

  // IO
  export import scanCSV = io.scanCSV;
  export import scanIPC = io.scanIPC;
  export import scanParquet = io.scanParquet;

  export import readCSV = io.readCSV;
  export import readIPC = io.readIPC;
  export import readJSON = io.readJSON;
  export import readParquet = io.readParquet;
  export import readAvro = io.readAvro;

  export import readCSVStream = io.readCSVStream;
  export import readJSONStream = io.readJSONStream;

  // lazy
  export import col = lazy.col
  export import cols = lazy.cols
  export import lit = lazy.lit
  export import arange = lazy.arange
  export import argSortBy = lazy.argSortBy
  export import avg = lazy.avg
  export import concatList = lazy.concatList
  export import concatString = lazy.concatString
  export import count = lazy.count
  export import cov = lazy.cov
  export import exclude = lazy.exclude
  export import first = lazy.first
  export import format = lazy.format
  export import groups = lazy.groups
  export import head = lazy.head
  export import last = lazy.last
  export import mean = lazy.mean
  export import median = lazy.median
  export import nUnique = lazy.nUnique
  export import pearsonCorr = lazy.pearsonCorr
  export import quantile = lazy.quantile
  export import select = lazy.select
  export import spearmanRankCorr = lazy.spearmanRankCorr
  export import tail = lazy.tail
  export import list = lazy.list
  export import when = _when.when;
  export const version = _version;
}

export = pl;
