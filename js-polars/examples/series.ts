import pl from '../polars'



// Creating series
console.log(pl.Series('str', ["foo", "bar", "baz"]))
console.log(pl.Series('num', [1,2,3]))
console.log(pl.Series('bool', [true, false]))
const s = pl.repeat("foo", 10, pl.Utf8)
console.log(s.getIdx(1));

