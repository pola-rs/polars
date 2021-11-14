import pl from '../polars'

const s = pl.Series('a', [1, 2, 3])
console.log(s)

let s2 = pl.Series(
  'series 2',
  [{foo: "bar"}, 1],
  {type: pl.Object, strict: true}
)

s2 = s2.rename('renamed series 2')
console.log(s2)