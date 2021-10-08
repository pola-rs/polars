# Changelog polars (Python bindings)

The Rust crate `polars` has its own changelog.
# Changelog

## [py-polars-v0.10.0](https://github.com/pola-rs/polars/tree/py-polars-v0.10.0) (2021-10-08)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.12...py-polars-v0.10.0)

**Merged pull requests:**

- python collect\_all function [\#1503](https://github.com/pola-rs/polars/pull/1503) ([ritchie46](https://github.com/ritchie46))
- python: remove deprecated code/funcitonallity [\#1502](https://github.com/pola-rs/polars/pull/1502) ([ritchie46](https://github.com/ritchie46))
- Refactor Date/Datetime dtypes [\#1501](https://github.com/pola-rs/polars/pull/1501) ([ritchie46](https://github.com/ritchie46))
- add vertical string concat; closes \#1490 [\#1500](https://github.com/pola-rs/polars/pull/1500) ([ritchie46](https://github.com/ritchie46))
- fix bug in outer\_join functions, add tests [\#1498](https://github.com/pola-rs/polars/pull/1498) ([marcvanheerden](https://github.com/marcvanheerden))
- refactor cast [\#1493](https://github.com/pola-rs/polars/pull/1493) ([ritchie46](https://github.com/ritchie46))
- Redesign logical types [\#1489](https://github.com/pola-rs/polars/pull/1489) ([ritchie46](https://github.com/ritchie46))
- python: concat also Series [\#1484](https://github.com/pola-rs/polars/pull/1484) ([ritchie46](https://github.com/ritchie46))
- fix outer join on floats and remove dtype-u64 flag [\#1483](https://github.com/pola-rs/polars/pull/1483) ([ritchie46](https://github.com/ritchie46))
- fill\_nan expression and fix \#1478 [\#1480](https://github.com/pola-rs/polars/pull/1480) ([ritchie46](https://github.com/ritchie46))
- Allowed non-vec in DF expressions. [\#1477](https://github.com/pola-rs/polars/pull/1477) ([jorgecarleitao](https://github.com/jorgecarleitao))
- fix rank on data with nulls: closes \#1473 [\#1475](https://github.com/pola-rs/polars/pull/1475) ([ritchie46](https://github.com/ritchie46))
- Python rename drop\_column\(s\) to drop \(union eager/lazy\) and pl.exclud… [\#1472](https://github.com/pola-rs/polars/pull/1472) ([ritchie46](https://github.com/ritchie46))
- change memmap2 version from 0.2.0 -\> 0.5.0 to support wasm [\#1470](https://github.com/pola-rs/polars/pull/1470) ([LemonPy29](https://github.com/LemonPy29))
- add categorical anyvalue [\#1464](https://github.com/pola-rs/polars/pull/1464) ([ritchie46](https://github.com/ritchie46))
- Fix cat append [\#1462](https://github.com/pola-rs/polars/pull/1462) ([ritchie46](https://github.com/ritchie46))
- Optional pyarrow [\#1459](https://github.com/pola-rs/polars/pull/1459) ([ghuls](https://github.com/ghuls))
- remove time64 dtype [\#1458](https://github.com/pola-rs/polars/pull/1458) ([ritchie46](https://github.com/ritchie46))
- fix agg\_std/agg\_var for Float32 [\#1449](https://github.com/pola-rs/polars/pull/1449) ([ritchie46](https://github.com/ritchie46))
- sort multiple with date64; closes \#1437 [\#1438](https://github.com/pola-rs/polars/pull/1438) ([ritchie46](https://github.com/ritchie46))
- Lazy: rename columns + Join suffix [\#1433](https://github.com/pola-rs/polars/pull/1433) ([ritchie46](https://github.com/ritchie46))
- add reverse collect for numeric ChunkedArray [\#1430](https://github.com/pola-rs/polars/pull/1430) ([ritchie46](https://github.com/ritchie46))
- fix and test boolean Series indexing  [\#1425](https://github.com/pola-rs/polars/pull/1425) ([ritchie46](https://github.com/ritchie46))
- elide bound checks on validity in take kernels [\#1419](https://github.com/pola-rs/polars/pull/1419) ([ritchie46](https://github.com/ritchie46))
- Use "sep" instead of "delimiter" in pl.Dataframe\(\).to\_csv\(\). [\#1418](https://github.com/pola-rs/polars/pull/1418) ([ghuls](https://github.com/ghuls))
- fix rolling\_mean on integers: closes \#1411 [\#1413](https://github.com/pola-rs/polars/pull/1413) ([ritchie46](https://github.com/ritchie46))
- expose numeric bitwise on Series [\#1412](https://github.com/pola-rs/polars/pull/1412) ([ritchie46](https://github.com/ritchie46))
- improve hashing performance w/ specialized hashers  [\#1405](https://github.com/pola-rs/polars/pull/1405) ([ritchie46](https://github.com/ritchie46))
- categorical aggregation output consistency [\#1403](https://github.com/pola-rs/polars/pull/1403) ([ritchie46](https://github.com/ritchie46))
- fix apply with list output type [\#1402](https://github.com/pola-rs/polars/pull/1402) ([ritchie46](https://github.com/ritchie46))
- expose categorical round trip to python and add more dictonary types:… [\#1399](https://github.com/pola-rs/polars/pull/1399) ([ritchie46](https://github.com/ritchie46))
- fix invalid ReplaceDropNulls optimization [\#1398](https://github.com/pola-rs/polars/pull/1398) ([ritchie46](https://github.com/ritchie46))
- fix memcpy of multiple chunks; closes 1396 [\#1397](https://github.com/pola-rs/polars/pull/1397) ([ritchie46](https://github.com/ritchie46))
- Workflow to generate m1 wheels [\#1394](https://github.com/pola-rs/polars/pull/1394) ([tiphaineruy](https://github.com/tiphaineruy))
- Add nan\_to\_none flag for converting NaN to None for from\_pandas [\#1393](https://github.com/pola-rs/polars/pull/1393) ([mahadeveaswar](https://github.com/mahadeveaswar))
- fix explode [\#1392](https://github.com/pola-rs/polars/pull/1392) ([ritchie46](https://github.com/ritchie46))
- improve performance of rolling\_var/ rolling\_std [\#1390](https://github.com/pola-rs/polars/pull/1390) ([ritchie46](https://github.com/ritchie46))
- python: More rolling\_window functions [\#1389](https://github.com/pola-rs/polars/pull/1389) ([ritchie46](https://github.com/ritchie46))
- improve rolling\_window performance [\#1387](https://github.com/pola-rs/polars/pull/1387) ([ritchie46](https://github.com/ritchie46))
- csv-parsing automatically parse dates [\#1386](https://github.com/pola-rs/polars/pull/1386) ([ritchie46](https://github.com/ritchie46))
- Rust; add parquet compression [\#1385](https://github.com/pola-rs/polars/pull/1385) ([ritchie46](https://github.com/ritchie46))
- python improve various ergonomics; Series.shift\(expr\), Series.shift\_a… [\#1384](https://github.com/pola-rs/polars/pull/1384) ([ritchie46](https://github.com/ritchie46))
- add conversion between categorical \<-\> arrow dictionary [\#1380](https://github.com/pola-rs/polars/pull/1380) ([ritchie46](https://github.com/ritchie46))
- feature gate categorical dtype [\#1379](https://github.com/pola-rs/polars/pull/1379) ([ritchie46](https://github.com/ritchie46))
- Ipc: read\_schema and projection option in read\_ipc [\#1378](https://github.com/pola-rs/polars/pull/1378) ([ritchie46](https://github.com/ritchie46))
- keep col name in fill\_null [\#1377](https://github.com/pola-rs/polars/pull/1377) ([ritchie46](https://github.com/ritchie46))
- python DataFrame.with\_column -\> pl.Series [\#1373](https://github.com/pola-rs/polars/pull/1373) ([ritchie46](https://github.com/ritchie46))
- proposal: implement \_\_bool\_\_ for pl.Expr [\#1368](https://github.com/pola-rs/polars/pull/1368) ([wseaton](https://github.com/wseaton))
- fix bug in Lazy::apply [\#1367](https://github.com/pola-rs/polars/pull/1367) ([ritchie46](https://github.com/ritchie46))
- Lazy: fix inconsistency in apply\_flat [\#1366](https://github.com/pola-rs/polars/pull/1366) ([ritchie46](https://github.com/ritchie46))
- Flate2 zlib ng [\#1365](https://github.com/pola-rs/polars/pull/1365) ([ritchie46](https://github.com/ritchie46))
- python make division consistent [\#1364](https://github.com/pola-rs/polars/pull/1364) ([ritchie46](https://github.com/ritchie46))
- remove non par sort branches. reduces code bloat [\#1363](https://github.com/pola-rs/polars/pull/1363) ([ritchie46](https://github.com/ritchie46))

## [py-polars-v0.9.12](https://github.com/pola-rs/polars/tree/py-polars-v0.9.12) (2021-09-27)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.11...py-polars-v0.9.12)

**Closed issues:**

- ShapeMisMatch error when vstacking two dataframes with equal amount columns, but different number of rows [\#1452](https://github.com/pola-rs/polars/issues/1452)
- split `agg_std` and `agg_mean` for floats  [\#1447](https://github.com/pola-rs/polars/issues/1447)
- `.std().over('groups')` raises a `PanicException` when one of the columns has dtype `Float32`. [\#1446](https://github.com/pola-rs/polars/issues/1446)
- A numeric type is converted to a string type [\#1444](https://github.com/pola-rs/polars/issues/1444)
- Adding a column without stating df length [\#1440](https://github.com/pola-rs/polars/issues/1440)
- Sorting by multiple columns doesn't work when one of the columns is `Date64` [\#1437](https://github.com/pola-rs/polars/issues/1437)
- Include a center option in the rolling function \(like pandas\) [\#1436](https://github.com/pola-rs/polars/issues/1436)
- Parquet File Size Larger than CSV File Size [\#1381](https://github.com/pola-rs/polars/issues/1381)
- Dead links in source [\#1370](https://github.com/pola-rs/polars/issues/1370)

## [py-polars-v0.9.11](https://github.com/pola-rs/polars/tree/py-polars-v0.9.11) (2021-09-24)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.10...py-polars-v0.9.11)

**Closed issues:**

- Adding an optional suffix to overlaping columns when joining dataframes [\#1432](https://github.com/pola-rs/polars/issues/1432)
- collect reverse [\#1429](https://github.com/pola-rs/polars/issues/1429)
- indexing bool column gives NotImplemented error [\#1422](https://github.com/pola-rs/polars/issues/1422)
- use `sep` instead of `delimiter` in DataFrame.to\_csv [\#1415](https://github.com/pola-rs/polars/issues/1415)

## [py-polars-v0.9.10](https://github.com/pola-rs/polars/tree/py-polars-v0.9.10) (2021-09-22)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.9...py-polars-v0.9.10)

**Closed issues:**

- Rolling\_mean only working with floats [\#1411](https://github.com/pola-rs/polars/issues/1411)
- Add numeric bitwise operations on Series/ Exprs [\#1410](https://github.com/pola-rs/polars/issues/1410)
- Polars date arithmetic is not correct [\#1404](https://github.com/pola-rs/polars/issues/1404)
- mean, median, mode, ... probably should not work for pl.Categorical [\#1401](https://github.com/pola-rs/polars/issues/1401)
- Initially empty series causes memcpy\_values to index OOB [\#1396](https://github.com/pola-rs/polars/issues/1396)
- Filter gets mis-optimized and doesn't filter out false values [\#1395](https://github.com/pola-rs/polars/issues/1395)
- pl.DataFrame.explode\(\) gives an empty name name now for the exploded column. [\#1391](https://github.com/pola-rs/polars/issues/1391)
- m1 wheels generation [\#1345](https://github.com/pola-rs/polars/issues/1345)
- Arrow dictionaries -\> Polars Categorical [\#1308](https://github.com/pola-rs/polars/issues/1308)
- RuntimeError: Other\("Could not determine output type"\) [\#1307](https://github.com/pola-rs/polars/issues/1307)
- Add from\_pandas flag for converting NaN to None [\#1164](https://github.com/pola-rs/polars/issues/1164)

## [py-polars-v0.9.9](https://github.com/pola-rs/polars/tree/py-polars-v0.9.9) (2021-09-19)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.8...py-polars-v0.9.9)

**Closed issues:**

- add rolling\_std [\#1388](https://github.com/pola-rs/polars/issues/1388)
- Allow fill\_null to accept Expr for Series [\#1383](https://github.com/pola-rs/polars/issues/1383)
- getting AttributeError when using `.is_in` on polars 0.9.7, but works on polars 0.8.\* [\#1382](https://github.com/pola-rs/polars/issues/1382)

## [py-polars-v0.9.8](https://github.com/pola-rs/polars/tree/py-polars-v0.9.8) (2021-09-18)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.7...py-polars-v0.9.8)

**Closed issues:**

- Faster gzip decompression [\#1359](https://github.com/pola-rs/polars/issues/1359)

## [py-polars-v0.9.7](https://github.com/pola-rs/polars/tree/py-polars-v0.9.7) (2021-09-16)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.6...py-polars-v0.9.7)

**Closed issues:**

- Series with large ints lose precision when divided [\#1369](https://github.com/pola-rs/polars/issues/1369)
- // integer division in python not working  [\#1362](https://github.com/pola-rs/polars/issues/1362)
- Add a method to extract a column as a series [\#1346](https://github.com/pola-rs/polars/issues/1346)

## [py-polars-v0.9.6](https://github.com/pola-rs/polars/tree/py-polars-v0.9.6) (2021-09-15)

[Full Changelog](https://github.com/pola-rs/polars/compare/0.16.0...py-polars-v0.9.6)

## [py-polars-v0.9.6-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.9.6-beta.1) (2021-09-13)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.5...py-polars-v0.9.6-beta.1)

**Closed issues:**

- Allow rounding within `.agg()` [\#1336](https://github.com/pola-rs/polars/issues/1336)
- Filtering of DataFrame based on the column with dates in Date32 format is not working [\#1332](https://github.com/pola-rs/polars/issues/1332)
- Filters sometime fill null string value with "" [\#1322](https://github.com/pola-rs/polars/issues/1322)
- Allow conversion from/to list of dicts [\#1300](https://github.com/pola-rs/polars/issues/1300)

## [py-polars-v0.9.5](https://github.com/pola-rs/polars/tree/py-polars-v0.9.5) (2021-09-10)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.5-beta.1...py-polars-v0.9.5)

## [py-polars-v0.9.5-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.9.5-beta.1) (2021-09-10)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.4...py-polars-v0.9.5-beta.1)

**Closed issues:**

- read\_csv: dtypes date32 and date64 are not inferred correctly anymore [\#1330](https://github.com/pola-rs/polars/issues/1330)
- add clip function [\#1326](https://github.com/pola-rs/polars/issues/1326)
- Allow `.agg()` to work on ungrouped data frames [\#1324](https://github.com/pola-rs/polars/issues/1324)
- Add `.with_column_renamed()` eager method [\#1323](https://github.com/pola-rs/polars/issues/1323)
- add two list array. [\#1316](https://github.com/pola-rs/polars/issues/1316)
- TypeError: large\_list\(\) takes exactly one argument \(0 given\) [\#1303](https://github.com/pola-rs/polars/issues/1303)
- parquet dict encoded panic [\#1281](https://github.com/pola-rs/polars/issues/1281)
- fix explode on lists with empty values [\#1177](https://github.com/pola-rs/polars/issues/1177)

## [py-polars-v0.9.4](https://github.com/pola-rs/polars/tree/py-polars-v0.9.4) (2021-09-10)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.3...py-polars-v0.9.4)

**Closed issues:**

- Add standard error as an aggregation expression \(standard deviation / sqrt\(number of measurements\) \) [\#1315](https://github.com/pola-rs/polars/issues/1315)
- Make impossible datatime aggregations consistent with primitives [\#1311](https://github.com/pola-rs/polars/issues/1311)
- Add an argument `strict` to constructors, cast, etc to ensure the integer range etc on python polars \(and otherwise\) instead of silently converting to null. [\#1293](https://github.com/pola-rs/polars/issues/1293)

## [py-polars-v0.9.3](https://github.com/pola-rs/polars/tree/py-polars-v0.9.3) (2021-09-04)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.3-beta.1...py-polars-v0.9.3)

**Closed issues:**

- data = data\[range\(N\)\] isn't very performant [\#1283](https://github.com/pola-rs/polars/issues/1283)

## [py-polars-v0.9.3-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.9.3-beta.1) (2021-09-03)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.2...py-polars-v0.9.3-beta.1)

## [py-polars-v0.9.2](https://github.com/pola-rs/polars/tree/py-polars-v0.9.2) (2021-09-03)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.1...py-polars-v0.9.2)

**Closed issues:**

- cannot explode dataframe with single list column [\#1288](https://github.com/pola-rs/polars/issues/1288)
- Sort dataframe using Date32 column is causing nulls to appear [\#1277](https://github.com/pola-rs/polars/issues/1277)
- Drop duplicates [\#1260](https://github.com/pola-rs/polars/issues/1260)
- rolling functions in lazy [\#1185](https://github.com/pola-rs/polars/issues/1185)

## [py-polars-v0.9.1](https://github.com/pola-rs/polars/tree/py-polars-v0.9.1) (2021-08-31)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.0...py-polars-v0.9.1)

## [py-polars-v0.9.0](https://github.com/pola-rs/polars/tree/py-polars-v0.9.0) (2021-08-31)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.0-beta.2...py-polars-v0.9.0)

## [py-polars-v0.9.0-beta.2](https://github.com/pola-rs/polars/tree/py-polars-v0.9.0-beta.2) (2021-08-30)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.9.0-beta.1...py-polars-v0.9.0-beta.2)

## [py-polars-v0.9.0-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.9.0-beta.1) (2021-08-30)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.29...py-polars-v0.9.0-beta.1)

**Closed issues:**

- Aggregation fails when grouped by multiple columns with one column of datatype Int8 or Int16 [\#1255](https://github.com/pola-rs/polars/issues/1255)
- \[Rust\] cross\_join coalesce left to right [\#1254](https://github.com/pola-rs/polars/issues/1254)
- rename all `null` to `none` [\#1247](https://github.com/pola-rs/polars/issues/1247)
- expr filter.count\(\) gives wrong result [\#1242](https://github.com/pola-rs/polars/issues/1242)
- Lazy: hard error on duplicate names [\#1241](https://github.com/pola-rs/polars/issues/1241)
- horizontal\_sum define null behavior [\#1173](https://github.com/pola-rs/polars/issues/1173)
- the new types with udf in apply function [\#1165](https://github.com/pola-rs/polars/issues/1165)

## [py-polars-v0.8.29](https://github.com/pola-rs/polars/tree/py-polars-v0.8.29) (2021-08-27)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.28...py-polars-v0.8.29)

**Closed issues:**

- pyo3\_runtime.PanicException: should already be coerced to u64 when joining two DataFrames [\#1231](https://github.com/pola-rs/polars/issues/1231)
- Error when performing modulo operation '%" [\#1230](https://github.com/pola-rs/polars/issues/1230)
- CsvReader ignores the last charactor if there's no newline at EOF [\#1229](https://github.com/pola-rs/polars/issues/1229)
- Bad link in doc [\#1227](https://github.com/pola-rs/polars/issues/1227)

## [py-polars-v0.8.28](https://github.com/pola-rs/polars/tree/py-polars-v0.8.28) (2021-08-26)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.27...py-polars-v0.8.28)

## [py-polars-v0.8.27](https://github.com/pola-rs/polars/tree/py-polars-v0.8.27) (2021-08-26)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.27-beta.2...py-polars-v0.8.27)

**Closed issues:**

- Remove HSTACK node  [\#1218](https://github.com/pola-rs/polars/issues/1218)
- Groupby Rank [\#1209](https://github.com/pola-rs/polars/issues/1209)
- Alias dense\_rank to argsort\_by + 1 [\#1207](https://github.com/pola-rs/polars/issues/1207)

## [py-polars-v0.8.27-beta.2](https://github.com/pola-rs/polars/tree/py-polars-v0.8.27-beta.2) (2021-08-25)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.27-beta.1...py-polars-v0.8.27-beta.2)

**Closed issues:**

- Predicate pushdown using a filter with `null` values doesn't correctly trigger the filter [\#1217](https://github.com/pola-rs/polars/issues/1217)
- Lazy: use a lazy function to determine type of apply/map [\#1203](https://github.com/pola-rs/polars/issues/1203)
- \[python\] missing dependency when using Anaconda [\#1200](https://github.com/pola-rs/polars/issues/1200)
- Add transpose method to dataframe [\#1176](https://github.com/pola-rs/polars/issues/1176)

## [py-polars-v0.8.27-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.8.27-beta.1) (2021-08-23)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.26...py-polars-v0.8.27-beta.1)

**Closed issues:**

- Python: add rolling apply [\#1194](https://github.com/pola-rs/polars/issues/1194)
- Slow concat with large list of dataframes [\#1183](https://github.com/pola-rs/polars/issues/1183)

## [py-polars-v0.8.26](https://github.com/pola-rs/polars/tree/py-polars-v0.8.26) (2021-08-21)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.25...py-polars-v0.8.26)

## [py-polars-v0.8.25](https://github.com/pola-rs/polars/tree/py-polars-v0.8.25) (2021-08-20)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.24...py-polars-v0.8.25)

## [py-polars-v0.8.24](https://github.com/pola-rs/polars/tree/py-polars-v0.8.24) (2021-08-20)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.23...py-polars-v0.8.24)

**Closed issues:**

- Expose interpolate on DataFrame in core [\#1161](https://github.com/pola-rs/polars/issues/1161)
- \[Python\] df.filter fails with "out of range" error [\#1157](https://github.com/pola-rs/polars/issues/1157)

## [py-polars-v0.8.23](https://github.com/pola-rs/polars/tree/py-polars-v0.8.23) (2021-08-18)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.22...py-polars-v0.8.23)

**Closed issues:**

- shift\_and\_fill by groups + other operations by grouping variables [\#1124](https://github.com/pola-rs/polars/issues/1124)

## [py-polars-v0.8.22](https://github.com/pola-rs/polars/tree/py-polars-v0.8.22) (2021-08-17)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.22-beta.1...py-polars-v0.8.22)

**Closed issues:**

- inspect/debug expr [\#1146](https://github.com/pola-rs/polars/issues/1146)

## [py-polars-v0.8.22-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.8.22-beta.1) (2021-08-16)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.21...py-polars-v0.8.22-beta.1)

**Closed issues:**

- Filter expression should call eval\_on\_groups [\#1148](https://github.com/pola-rs/polars/issues/1148)
- the trait `FromIterator<&&str>` is not implemented for `polars::prelude::Series` [\#1147](https://github.com/pola-rs/polars/issues/1147)
- Shift in aggregation context [\#1141](https://github.com/pola-rs/polars/issues/1141)
- Get/set categorical levels? [\#1115](https://github.com/pola-rs/polars/issues/1115)
- Is it possible to create a dataframe from row-like \(Vec\<Struct\>\) data? [\#1111](https://github.com/pola-rs/polars/issues/1111)
- PanicException when filter on sorted columns [\#1110](https://github.com/pola-rs/polars/issues/1110)
- Pickle support for Polars dataframes [\#1109](https://github.com/pola-rs/polars/issues/1109)
- Rosetta stone for groupby between pandas and polars? [\#1083](https://github.com/pola-rs/polars/issues/1083)

## [py-polars-v0.8.21](https://github.com/pola-rs/polars/tree/py-polars-v0.8.21) (2021-08-13)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.21-beta.2...py-polars-v0.8.21)

**Closed issues:**

- csv schema inference scientific float notation [\#1134](https://github.com/pola-rs/polars/issues/1134)
- python: csv read\_cvs dtypes accept list [\#1133](https://github.com/pola-rs/polars/issues/1133)

## [py-polars-v0.8.21-beta.2](https://github.com/pola-rs/polars/tree/py-polars-v0.8.21-beta.2) (2021-08-12)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.21-beta.1...py-polars-v0.8.21-beta.2)

## [py-polars-v0.8.21-beta.1](https://github.com/pola-rs/polars/tree/py-polars-v0.8.21-beta.1) (2021-08-12)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.20...py-polars-v0.8.21-beta.1)

**Closed issues:**

- Extremely slow `pivot.count` \(compared with `pivot.sum`, `pivot.max`, `pivot.first`, ...\) [\#1129](https://github.com/pola-rs/polars/issues/1129)
- Only first gzip stream of gzipped CSV/TSV files with multiple gzip streams is read. [\#1126](https://github.com/pola-rs/polars/issues/1126)
- implement `__copy__, __deepcopy__` [\#1120](https://github.com/pola-rs/polars/issues/1120)
- pretty print failure output of `frame_equal` assertions [\#1112](https://github.com/pola-rs/polars/issues/1112)
- PanicException when converting non-Utf8 column to Catergorical. [\#1107](https://github.com/pola-rs/polars/issues/1107)
- read\_csv of a compressed file fails when selecting a subset of columns [\#1026](https://github.com/pola-rs/polars/issues/1026)
- csv-parser: remove dependency on csv crate. [\#956](https://github.com/pola-rs/polars/issues/956)

## [py-polars-v0.8.20](https://github.com/pola-rs/polars/tree/py-polars-v0.8.20) (2021-08-07)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.19...py-polars-v0.8.20)

## [py-polars-v0.8.19](https://github.com/pola-rs/polars/tree/py-polars-v0.8.19) (2021-08-06)

[Full Changelog](https://github.com/pola-rs/polars/compare/py-polars-v0.8.18...py-polars-v0.8.19)

**Closed issues:**

- numpy boolean vectors get converted as "objects" instead of cast to bool [\#1105](https://github.com/pola-rs/polars/issues/1105)
- PanicException: assertion failed: i \< \(self.bits.len\(\) \<\< 3\) [\#1098](https://github.com/pola-rs/polars/issues/1098)

### Polars 0.8.18
* feature
  - select columns by regex
  - support >1M columns in IPC reader
  - make DataFrame.sort arguments equal to LazyFrame.sort
  - pl.all() == pl.col("*")

* bug fix
  - fix bugs due to filtering in aggregations: #1101
  - fix bug in wildcard in functions `3163ee5`

### Polars 0.8.17
* feature
  - keep_name expr
  - exclude expr
  - drop_nulls expre
  - explode accepts expression (thus wildcard)
  - groupby: head + tail added
  - `df[::2]` slicing added
  
### Polars 0.8.16
patch release to fix panic #1077

### Polars 0.8.15
* feature
  - extract jsonpath
  - more object support
* performance
  - improve list take performance
* bug fix
  - don't panic in out of bounds take, but error
  - update offsets in case of utf8lossy
  - fix bug in pyarrow round trip with list types

### Polars 0.8.13/(14 patch)
* feature
  - concat_str function
  - more object support
  - hash and row_hash function/ expr
  - reinterpret function/ expr
  - Series.mode expr/function
  - csv file decompression
  - read_sql support
* performance
  - divide and conquer binary expressions

### Polars 0.8.12
* feature
  - cross join added
  - dot-product
* performance
  - improve csv-parser performance by ~25%
* bug fix
  - various minor

### Polars 0.8.11
* feature
  - cross join added
  - dot-product
* performance
  - improve csv-parser performance by ~25%
* bug fix
  - various minor

### Polars 0.8.10
* feature
  - is_first expr/method
  - asof join added
  - eager io can open multiple sources with ffspec
  - resolve `~` to homedir
  - python arange add step and run eager
* performance
  - use fast csv-parser for more python memory buffers/streams
* bug fix
  - kleene or and and operations
  - maybe fix rayon deadlock
  - concat is a pure function
  - string addition lhs broadcast

### Polars 0.8.9
* feature
  - correct type hints for python 3.6
  - csv-parser option to ignore comment lines
* performance
  - improve take on DataFrame
  - remove bound checks in buffer creation
  - improve performance of sorting by multiple columns
  - improve argsort performance
* bug fix
  - fix backward/forward fill
  - window groupby context
  - fix is_duplicated dispatch

### Polars 0.8.8
* bug fix
  - fix UB due to slice in take kernel
  - fix join for dates
  
### Polars 0.8.7
* feature
  - from_pandas accept series and date range #875
  - expr: forward_fill, backward_fill #874
  - gzipped file support in csv parser
* performance
  - reduce memory usage of multi-key groupby
  - improve variance and std-dev aggregation
* bug fix
  - cast to large-utf8 before collecting chunks #870
  - various

### Polars 0.8.6
* performance
  - improve hashing performance for grouping on two keys for 64 bit and 32 and 64 bit data.
  - improve cache coherence take operation of multiple chunks
* bug fix
  - fix replaxing string with None #802

### Polars 0.8.5
* feature
  - improve compatibility with pyarrow csv parser
* performance
  - improve hashing performance for grouping on two keys for 64 bit and 32 and 64 bit data.
  - improve cache coherence take operation of multiple chunks
  - fast path for categorical unique
  - decrease memory fragmentation and usage of csv-parser
* bug fix
  - split utf8 data only at valid char boundaries #789
  - fix bug in outer join due to new partitioning algorithm

### Polars 0.8.4
* feature
  - Series.round
  - head/ limit aliases
* performance
  - partitioned hashing
  
### Polars 0.8.0
* breaking change
  - `str` namespace Series.str_* methods to Series.str.<method>
  - `dt` namespace Series datetime related methods to Series.dt.<method>
    
* feature
  - DataFrame.rows method
  - apply on object types
  - `Series.dt.to_python_datetime`
  - `Series.dt.timestamp`
  
* bug fix
  - preserve date64 in round trip to parquet #723
  - during arrow conversion coerce categorical to utf8 (this preserves string data) #725
  - fix bug in csv skip rows

* performance
  - improve hashing of string data in groupby and join
  - improve numeric hashing in join
  - fast path for filtering no data and all date (upstream)

### polars 0.7.19
* feature
  - window function by multiple group columns

* bug fix
  - fix bug in argsort multiple
  - fix bug in filter with nulls (upstream)

* performance
  - improve numeric hashing in groupby
  - fast paths for filters (upstream)

### polars 0.7.18
* feature
  - argsort multiple columns

### polars 0.7.17
* feature
  - support more indexing
  - scan_csv low memory argument
  - Series.filter accept list of expressions
  - object type:
      - zip
      - take -> join / groupby agg
      - agg first/ last

* performance
  - change memory usage of csv-parser
  - binary aggregation in parallel
  - determine groupby keys in threadpool

### polars 0.7.16
* feature
  - Series literal may have any length
  - change globaly string cache behavior
  - Add Expr.arg_sort
  - Make literals typed

* bug fix
  - Fix Expr.fill_null
  - set offset in null buffers (fixes aggregation with null values)

* performance
  - sample cardinality in groupby and choose algorithm

### polars 0.7.15
* feature
  - join allows expression syntax
  - use pyarrow as default ipc backend
  
* bug fix
  - fix deadlock in window expressions

### polars 0.7.13 / 0.7.14 (patch) 2021-05-08

* bug fix
  - fix bug in cumsum #604
  
* feature
  - DataFrame.describe method #606
  - Multi-level sorting of a DataFrame #607
  - Expand functionality of Expr.is_in #614
  - Csv-parser low_memory option #615
  - Allow expressions in `pl.arange` #611
    
* performance
  - sort().reverse() optimization #605

### polars 0.7.12
* bug fix
  - null handling in mean, std, var, and cov aggregations. #595
  - rev-mapping of categorical stored duplicates. #595
  - fix memory surge after csv-parsing #593

### polars 0.7.11 
* bug fix
  - Throw error on join from different string cache #584
  - fix covariance of array with null values #585

* feature
  - Series describe method #569
  - dsl: take, arg_unique, unique
  - allow lazy expressions in Eager API # 588
  - describe Series

* performance
  - fix accidental expensive appends #592
  - remove chunk_id from ChunkedArray #593


### polars 0.7.8 -> 0.7.9 (patched)
* bug fix
  - ensure column name persist after pyarrow cast #563
  - make sure that `agg_list` maintains dtype #567
  - fix panic in physical dispatch of Date dtypes

* feature
  - Implicitly Cast dtypes to temporal types in csv parser #560
  - Series describe method #569

* performance
  - Cache and improve window functions performance #570

### polars 0.7.7
* bug fix
  - fix bug with pyarrow chunkedarray: #545

* feature
  - DataFrame.apply method
  - Make a Series a Literal
  - Make None a Literal
    
* performance
  - Update arrow
    * faster iterators
    * faster kernels

### polars 0.7.6
* bug fix
  - fix bug in downsample: #537

* feature
  - cast categorical in csv parser: #533
  - add many groupby-context aware operations: #534
  - dowcast by month: #537

* performance
  - improve iterator in no null case: #538
  - remove indirection: #536

### polars 0.7.5
* bug fix
  - fix bug in vectorized hashing algorithm that affected groupbys with null values: #523
  - fix bug in downsample: 528
  - change median algorithm: #527

* feature
  - use lazy groupby API/DSL in eager API: #522
  - make sort groupby-context aware: #522

* performance
  - improve sort algorithms for sort and argsort: #526

### polars 0.7.4
* performance
  - \[python | rust\] multi-threaded outer join
  - \[python | rust\] better performance in groupby on multiple keys (faster hashmap comparisons)
  - \[python | rust\] better performance in multi column joins

* bug fix
  - \[python\] make horizontal aggregations null aware

* feature
  - \[python | rust\] Downsample by week
  - \[python | rust\] join by unlimited columns
  - \[python\] ~Create a list Series directly.~
  - \[python\] Create DataFrame from np.ndarray
  

### polars 0.7.3
* bug fix
  - \[python\] pandas to polars date64, maintain time information
  - \[python\] fix bug in Date64 Series.year
  - \[python\] fix bug Series.mean (did not correct for null values) #484
  - \[python | rust \] fix bug in rolling windows #484
  - \[python | rust \] fix bug lazy csv parser #459

* feature
  - \[python | rust\] Series methods
    * Series.week
    * Series.weekday
    * Series.arg_min
    * Series.arg_max
    * Series.shape

### polars 0.7.2
* bug fix
  - \[python\] More pyarrow -> polars conversions.
    
* feature
  - \[python\] DataFrame methods: \[ shift_and_fill\].
  - \[python\] eager: sum, min, max, mean horizontal aggregation.

### polars 0.7.1
* performance
  - \[python | rust\] arrow arrays have a layer of indirection less; 10/20% performance improvement

### polars 0.7.0
* name change: Python bindings module renamed from pypolars to polars
* name change: Python bindings package renamed from py-polars to polars

* feature
  - \[python\] lazy: DataFrame methods: \[ tail, first, last \].
  - \[python\] eager: DataFrame fold for horizontal aggregation.
  - \[python\] eager: Series methods: \[median, quantile, is_in, to_frame\]
  - \[python\] eager: iterate over groupby and yield groups' DataFrames
  - \[python\] eager: groupby.get_group('value')
  - \[python\] add parquet compression
  - \[python\] shift_and_fill expression
  - \[python\] implicitly download raw files from the web in `read_parquet`, `read_csv`.
  - \[python | rust\] methods for local peak finding in numerical series
  - \[python | rust\] faster query optimization due to local memory arena's.
  - \[rust\] reduce default compile time by making less features default.
  - \[python | rust\] Series zip_with implicitly cast to supertype.
  - \[python | rust\] window functions have a `min_periods` argument to control when to compute a result
  
* bug fix
  - \[python\] support file buffers for reading and writing csv and parquet
  - \[python | rust\] fix csv-parser: allow new-line character in a string field
  - \[python | rust\] don't let predicate-pushdown pass shift | sort operation to maintain correctness.

### py-polars 0.6.7
* performance
  - \[python | rust\] use mimalloc global allocator
  - \[python | rust\] undo performance regression on large number of threads
* bug fix
  - \[python | rust\] fix accidental over-allocation in csv-parser
  - \[python\] support agg (dictionary aggregation) for downsample

### py-polars 0.6.6
* performance
  - \[python | rust\] categorical type groupby keys (use size hint)
  - \[python | rust\] remove indirection layer in vector hasher
  - \[python | rust\] improve performance of null array creation
* bug fix
  - \[python\] implement set_with_mask for Boolean type
  - \[python | rust\] don't panic (instead return null) in dataframe aggregation `std` and `var`
* other
  - \[rust\] internal refactors

### py-polars 0.6.5
* bug fix
  - \[python\] fix various pyarrow related bugs
  
### py-polars 0.6.4
* feature
  - \[python\] render html tables
* performance
  - \[python\] default to pyarrow for parquet reading
  - \[python | rust\] use u32 instead of usize in groupby and join to increase cache coherence and reduce memory pressure.
