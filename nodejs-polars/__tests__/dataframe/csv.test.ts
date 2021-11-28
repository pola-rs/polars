import pl from '@polars';
import path from 'path';

describe('csv', () => {
  // eslint-disable-next-line no-undef
  const csvpath = path.resolve(__dirname, "../../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv");

  describe('read', () => {
    it('can read from a csv file', () => {
      const df = pl.readCSV(csvpath);
      expect(df.shape).toStrictEqual({height: 27, width: 4});
    });

    it('can read from a csv string', () => {
      const csvString = 'foo,bar,baz\n1,2,3\n4,5,6\n';
      const df = pl.readCSV(csvString);
      expect(df.toCSV()).toEqual(csvString);
    });
  });

  describe('write', () => {
    it('allows no headers on write', () => {
      const csvString = 'foo,bar,baz\n1,2,3\n4,5,6\n';
      const df = pl.readCSV(csvString);
      expect(df.toCSV({hasHeader:false})).toEqual('1,2,3\n4,5,6\n');
    });

    it('allows a custom separator', () => {
      const csvString = 'foo,bar,baz\n1,2,3\n4,5,6\n';
      const df = pl.readCSV(csvString);
      expect(df.toCSV({hasHeader:false, sep: 'X'})).toEqual('1X2X3\n4X5X6\n');
    });
  });
});
