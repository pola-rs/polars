import pl from '../polars';

// Creating series
// console.log(pl.Series('str', ['foo', 'bar', 'baz']))
// console.log(pl.Series('num', [1, 2, 3]))
let s = pl.Series('a', ['foo', 'bar'])
  .rename('new name');
console.log(s.toJS());
