import pl from '@polars'
import Chance from 'chance'
describe('series', () => {
  const chance = new Chance()

  describe('create series', () => {
    it.each`
      values
      ${[1, 1n]}
      ${['foo', 2]}
      ${[false, 'false']}
    `('does not allow multiple types', ({ values }) => {
      try {
        pl.Series('', values)
      } catch (err) {
        expect((err as Error).message).toStrictEqual('Multi type Series is not supported')
      }
    })

    it.each`
      values                   | dtype
      ${['foo', 'bar', 'baz']} | ${'Utf8'}
      ${[1, 2, 3]}             | ${'Float64'}
      ${[1n, 2n, 3n]}          | ${'UInt64'}
      ${[true, false]}         | ${'Bool'}
    `('defaults to $dtype for $input', ({ values, dtype }) => {
      const name = chance.string()
      const s = pl.Series(name, values)
      expect(s.name).toStrictEqual(name)
      expect(s.length).toStrictEqual(values.length)
      expect(s.dtype).toStrictEqual(dtype)
    })

    it.each`
      values                   | dtype
      ${['foo', 'bar', 'baz']} | ${'Utf8'}
      ${[1, 2, 3]}             | ${'Float64'}
      ${[1n, 2n, 3n]}          | ${'UInt64'}
    `('defaults to $dtype for $input', ({ values, dtype }) => {
      const name = chance.string()
      const s = pl.Series(name, values)
      expect(s.name).toStrictEqual(name)
      expect(s.length).toStrictEqual(values.length)
      expect(s.dtype).toStrictEqual(dtype)
    })
  })

  


})
