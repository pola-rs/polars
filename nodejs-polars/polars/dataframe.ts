/* eslint-disable no-unused-vars */
import polarsInternal from './internals/polars_internal';
import { arrayToJsSeries } from './internals/construction';
import util from 'util';
import { DataType, DtypeToPrimitive, DTYPE_TO_FFINAME, Optional } from './datatypes';
import {Series} from './series';


/**
 *  
  A DataFrame is a two-dimensional data structure that represents data as a table
  with rows and columns.

  Parameters
  ----------
  @param data -  Object, Array, or Series
      Two-dimensional data in various forms. object must contain Arrays.
      Array may contain Series or other Arrays.
  @param columns - Array of str, default undefined
      Column labels to use for resulting DataFrame. If specified, overrides any
      labels already present in the data. Must match data dimensions.
  @param orient - 'col' | 'row' default undefined
      Whether to interpret two-dimensional data as columns or as rows. If None,
      the orientation is inferred by matching the columns and data dimensions. If
      this does not yield conclusive results, column orientation is used.

  Examples
  --------
  Constructing a DataFrame from an object :
  ```
  >>> data = {'a': [1, 2], 'b': [3, 4]}
  >>> df = pl.DataFrame(data)
  >>> df
  shape: (2, 2)
  ╭─────┬─────╮
  │ a   ┆ b   │
  │ --- ┆ --- │
  │ i64 ┆ i64 │
  ╞═════╪═════╡
  │ 1   ┆ 3   │
  ├╌╌╌╌╌┼╌╌╌╌╌┤
  │ 2   ┆ 4   │
  ╰─────┴─────╯
  ```
  Notice that the dtype is automatically inferred as a polars Int64:
  ```
  >>> df.dtypes
  ['Int64', `Int64']
  ```
  In order to specify dtypes for your columns, initialize the DataFrame with a list
  of Series instead:
  ```
  >>> data = [pl.Series('col1', [1, 2], pl.Float32),
  ...         pl.Series('col2', [3, 4], pl.Int64)]
  >>> df2 = pl.DataFrame(series)
  >>> df2
  shape: (2, 2)
  ╭──────┬──────╮
  │ col1 ┆ col2 │
  │ ---  ┆ ---  │
  │ f32  ┆ i64  │
  ╞══════╪══════╡
  │ 1    ┆ 3    │
  ├╌╌╌╌╌╌┼╌╌╌╌╌╌┤
  │ 2    ┆ 4    │
  ╰──────┴──────╯
  ```

  Constructing a DataFrame from a list of lists, row orientation inferred:
  ```
  >>> data = [[1, 2, 3], [4, 5, 6]]
  >>> df4 = pl.DataFrame(data, ['a', 'b', 'c'])
  >>> df4
  shape: (2, 3)
  ╭─────┬─────┬─────╮
  │ a   ┆ b   ┆ c   │
  │ --- ┆ --- ┆ --- │
  │ i64 ┆ i64 ┆ i64 │
  ╞═════╪═════╪═════╡
  │ 1   ┆ 2   ┆ 3   │
  ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
  │ 4   ┆ 5   ┆ 6   │
  ╰─────┴─────┴─────╯
  ```
 */

export class DataFrame {

  constructor(
    data: Record<string, Array<any>> | Array<Array<any>> | Array<Series<any>>, 
    columns?: Array<string> ,
    orient?: 'col' | 'row'
  ) {
    const isArray = Array.isArray(data);

    if(!data) {
      return obj_to_df({});
    } else if (isArray && orient === 'col') {
      if (data[0] instanceof Series) {
        const items = data.map(s => s._series);

        return polarsInternal.df.new_from_columns({columns: items});
      } else {
        return arraysToDf((data as any), columns); 
      }
    } else if (isArray && orient === 'row') {
      

    }

    return;
  }


}

function obj_to_df(obj: Record<string, Array<any>>, columns?: Array<string>): any {
  const data =  Object.entries(obj).map(([key, value], idx) => {
    return Series.of(columns?.[idx] ?? key, value)._series;
  });

  return polarsInternal.df.new_from_columns({columns: data});
}

function arraysToDf(items: Array<Array<any>>, columns?: Array<string>) {
  const data = items.map((item, idx) => Series.of(columns?.[idx] ?? `${idx}`, item)._series);
  
  return polarsInternal.df.new_from_columns({columns: data});
}