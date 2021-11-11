import Dataframe from "./dataframe"
import pl from "./polars_internal";

export const fromArray = (js_array: Array<any>): Dataframe => {
  const df = pl.from_js_array({js_array})
  return Dataframe.from(df)
};

export const fromObjects = (js_objects: Record<string, any>): Dataframe => {
  const df = pl.from_js_object({js_objects})
  return Dataframe.from(df)
};
