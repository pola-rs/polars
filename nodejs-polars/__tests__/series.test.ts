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
        expect((err as Error).message).toBeDefined();
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

    it.each`
    values | type
    ${[1,2,3]} | ${'number'}
    ${['1','2','3']} | ${'string'}
    ${[1n,2n,3n]} | ${'bigint'}
    ${[true, false, null]} | ${'Option<bool>'}
    ${[1,2,null]} | ${'Option<number>'}
    ${[1n,2n,null]} |  ${'Option<bigint>'}
    ${[1.11,2.22,3.33, null]} |  ${'Option<float>'}
    ${new Int8Array([9,10,11])} | ${'Int8Array'}
    ${new Int16Array([12321,2456,22])} | ${'Int16Array'}
    ${new Int32Array([515121,32411322,32423])} | ${'Int32Array'}
    ${new Uint8Array([1,2,3,4,5,6,11])} | ${'Uint8Array'}
    ${new Uint16Array([1,2,3,55,11])} | ${'Uint16Array'}
    ${new Uint32Array([1123,2,3000,12801,99,43242])} | ${'Uint32Array'}
    `('can be created from $type', ({values}) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect([...s]).toEqual([...values]);
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
