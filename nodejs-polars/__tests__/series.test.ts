import pl from '@polars';
import Chance from 'chance';

describe('series', () => {
  const chance = new Chance();

  describe('create series', () => {
    it.each`
      values
      ${[1, 1n]}
      ${['foo', 2]}
      ${[false, 'false']}
    `('does not allow multiple types', ({ values }) => {
      try {
        pl.Series('', values);
      } catch (err) {
        expect((err as Error).message).toStrictEqual('Multi type Series is not supported');
      }
    });

    it.each`
      values                   | dtype        | type
      ${['foo', 'bar', 'baz']} | ${'Utf8'}    | ${'string'}
      ${[1, 2, 3]}             | ${'Float64'} | ${'number'}
      ${[1n, 2n, 3n]}          | ${'UInt64'}  | ${'bigint'}
      ${[true, false]}         | ${'Bool'}    | ${'boolean'}
    `('defaults to $dtype for "$type"', ({ values, dtype}) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });

    it.each`
      values                   | dtype
      ${['foo', 'bar', 'baz']} | ${'Utf8'}
      ${[1, 2, 3]}             | ${'Float64'}
      ${[1n, 2n, 3n]}          | ${'UInt64'}
    `('defaults to $dtype for $input', ({ values, dtype }) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });
  });
  
  describe('math', () => {

    it('can add', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      let s = pl.Series("", [item]);
      s = s.add(other);
      expect(s[0]).toStrictEqual(item + other);
    });

    it('can subtract', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.sub(other);
      expect(s[0]).toStrictEqual(item - other);
    });

    it('can multiply', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.mul(other);
      expect(s[0]).toStrictEqual(item * other);
    });

    it('can divide', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.div(other);
      expect(s[0]).toStrictEqual(item / other);
    });

    it('can add two series', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      let s = pl.Series("", [item]);
      s = s.add(pl.Series('', [other]));
      expect(s[0]).toStrictEqual(item + other);
    });

    it('can subtract two series', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.sub(pl.Series('', [other]));
      expect(s[0]).toStrictEqual(item - other);
    });

    it('can multiply two series', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.mul(pl.Series('', [other]));
      expect(s[0]).toStrictEqual(item * other);
    });

    it('can divide two series', () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      
      let s = pl.Series("", [item]);
      s = s.div(pl.Series('', [other]));
      expect(s[0]).toStrictEqual(item / other);
    });
  });

  describe('comparator', () => {
    it("can perform 'eq", () => {
      const s =  pl.Series("", [1,2,3]).eq(1);
      expect([...s]).toEqual([true, false, false]);
    });
  });
});
