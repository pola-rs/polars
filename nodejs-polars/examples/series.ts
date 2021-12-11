import pl, {col, lit} from "../polars";
import pli from "../polars/internals/polars_internal";
const df = pl.DataFrame({
  "foo": [1, 2, 3],
  "bar": [4, 5, 6],
});
const fn = () => df.select(col("foo"), lit(pl.Series([1, 2])));
console.log("fn", fn());