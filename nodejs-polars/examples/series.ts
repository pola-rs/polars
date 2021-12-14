import pl, {col, lit} from "../polars";
import pli from "../polars/internals/polars_internal";
const expected = [[1, 2], [3], [null], []];
const actual = pl.Series(expected).toArray();
console.log(actual);