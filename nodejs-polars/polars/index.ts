import Series from './series';
export default {
  Int8: 'Int8',
  Int16: 'Int16',
  Int32: 'Int32',
  Int64: 'Int64',
  UInt8: 'UInt8',
  UInt16: 'UInt16',
  UInt32: 'UInt32',
  UInt64: 'UInt64',
  Float32: 'Float32',
  Float64: 'Float64',
  Boolean: 'Boolean',
  Utf8: 'Utf8',
  List: 'List',
  Date: 'Date',
  Datetime: 'Datetime',
  Time: 'Time',
  Object: 'Object',
  Categorical: 'Categorical',

  /**
   * A Series represents a single column in a polars DataFrame.
   * @param name - Name of the series. Will be used as a column name when used in a DataFrame.
   * @param {PolarsArrayLike} values - One-dimensional data in various forms. Supported are: Array, Series,
   * Set
   * @param {Dtype} [dtype] - Polars dtype of the Series data. If not specified, the dtype is inferred.
   * @param [strict] - Throw error on numeric overflow
   *
   * @example
   * > const s = pl.Series('a', [1,2,3]);
   * > s
   * shape: (3,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   * ]
   * // Notice that the dtype is automatically inferred as a polars Int64:
   * > s.dtype()
   * "Int64"
   *
   * // Constructing a Series with a specific dtype:
   * > const s2 = pl.Series('a', [1, 2, 3], dtype=pl.Float32);
   * > s2
   * shape: (3,)
   * Series: 'a' [f32]
   * [
   *         1
   *         2
   *         3
   * ]
   */
  Series,
};
