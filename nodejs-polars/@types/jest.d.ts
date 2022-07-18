import {DataFrame} from "@polars/dataframe";
import {Series} from "@polars/series/series";

declare global {
  namespace jest {
    interface Matchers<R> {
      toSeriesEqual(b: Series<any>): R;
      toSeriesStrictEqual(b: Series<any>): R;
      toFrameEqual(b: DataFrame, nullEqual?: boolean): R;
      /**
       * Compares two DataFrames, including the dtypes
       *
       * @example
       * ```
       * >>> df = pl.Dataframe([pl.Series("int32": [1,2], pl.Int32)])
       * >>> other = pl.Dataframe([pl.Series("int32": [1,2], pl.UInt32)])
       *
       * >>> expect(df).toFrameEqual(other) // passes
       * >>> expect(df).toFrameStrictEqual(other) // fails
       * ```
       */
      toFrameStrictEqual(b: DataFrame): R;
      toFrameEqualIgnoringOrder(b: DataFrame): R;
    }
  }
}
