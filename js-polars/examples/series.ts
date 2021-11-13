import pl from '../polars'

const s = pl.Series('a', [1, 2, 3])
console.log(s)

let s2 = pl.Series(
  'series 2',
  ["apples", "oranges", "bananas"],
  {type: pl.Utf8, strict: true}
)

s2 = s2.rename('renamed series 2')
console.log(s2.dtype())