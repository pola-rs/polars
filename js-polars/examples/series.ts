import pl from '../polars'


console.log(pl.Series('str', ["foo", "bar", "baz"]))
console.log(pl.Series('num', [1,2,3]))
console.log(pl.Series('bool', [true, false]))

// let s2 = pl.Series(
//   'series 2',
//   [{foo: "bar"}, 1],
//   {type: pl.Object, strict: true}
// )

// s2 = s2.rename('renamed series 2')
// console.log(s2) ""