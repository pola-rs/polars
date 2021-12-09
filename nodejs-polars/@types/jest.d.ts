import {DataFrame} from "@polars/dataframe";
import {Series} from "@polars/series";

declare global {
  namespace jest {
    interface Matchers<R> {
      toSeriesEqual(b: Series<any>): R;
      toFrameEqual(b: DataFrame): R;
      toFrameEqualIgnoringOrder(b: DataFrame): R;
    }
  }
}