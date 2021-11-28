import pl from '../polars';

const  df = pl.DataFrame({
  "a": ["a", "b", "a", "b", "b", "c"],
  "b": [1, 2, 3, 4, 5, 6],
  "c": [6, 5, 4, 3, 2, 1],
});
// const s = df.groupBy('a')
//   .first()
//   .getColumn('c_first')
//   .toArray()
//   .sort();
// console.log(df.groupBy('a'));
// const gb = df.groupBy('a')(['b', 'c']); 
const restParamGroupBy  = df.groupBy('a', 'b', 'c');
const singleParamGroupBy  = df.groupBy('a')('a', 'b');
const arrayParamGroupBy = df.groupBy(['a', 'b']);

df
  .select('a', 'b', 'c')
  .peek()
  .select(['a'])
  .groupBy('a')
  .first()
  .peek();
