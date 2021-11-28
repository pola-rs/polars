import pl from '@polars';

describe('concat', () => {
  it('can concat multiple dataframes vertically', () => {
    const df1 = pl.DataFrame({
      "a": [1,2,3],
      "b": ['a', 'b', 'c']
    });
    const df2 = pl.DataFrame({
      "a": [4,5,6],
      "b": ['d', 'e', 'f']
    });
    const df3 = pl.concat([df1,df2]);
    expect(df3.shape).toStrictEqual({height: 6, width: 2});
  });

  it('can concat multiple series vertically', () => {
    const s1 = pl.Series("a", [1,2,3]);
    const s2 = pl.Series("a", [4,5,6]);
    const s3 = pl.concat([s1,s2]);
    expect(s3.length).toStrictEqual(6);
    expect([...s3]).toStrictEqual([1,2,3,4,5,6]);
  });
});