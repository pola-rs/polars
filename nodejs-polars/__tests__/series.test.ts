/* eslint-disable newline-per-chained-call */
import pl from "@polars";
import {InvalidOperationError} from "../polars/error";
import Chance from "chance";


describe("from lists", () => {
  test("bool", () => {
    const expected = [[true, false], [true], [null], []];
    const actual = pl.Series(expected).toArray();
    expect(actual).toEqual(expected);
  });
  test("number", () => {
    const expected = [[1, 2], [3], [null], []];
    const actual = pl.Series(expected).toArray();
    expect(actual).toEqual(expected);
  });
  test("bigint", () => {
    const expected = [[1n, 2n], [3n], [null], []];
    const actual = pl.Series(expected).toArray();
    expect(actual).toEqual(expected);
  });
  test("string", () => {
    const expected = [[], ["a"], [null], ["b", "c"]];
    const actual = pl.Series(expected).toArray();
    expect(actual).toEqual(expected);
  });
});
describe("typedArrays", () => {
  test("int8", () => {
    const int8Array = new Int8Array([1, 2, 3]);
    const actual = pl.Series(int8Array).toArray();
    const expected = [...int8Array];
    expect(actual).toEqual(expected);
  });
  test("int8:list", () => {
    const int8Arrays = [
      new Int8Array([1, 2, 3]),
      new Int8Array([33, 44, 55]),
    ];
    const expected = int8Arrays.map(i => [...i]);
    const actual = pl.Series(int8Arrays).toArray();
    expect(actual).toEqual(expected);
  });
  test("int16", () => {
    const int16Array = new Int16Array([1, 2, 3]);
    const actual = pl.Series(int16Array).toArray();
    const expected = Array.from(int16Array);
    expect(actual).toEqual(expected);
  });
  test("int16:list", () => {
    const int16Arrays = [
      new Int16Array([1, 2, 3]),
      new Int16Array([33, 44, 55]),
    ];
    const actual = pl.Series(int16Arrays).toArray();
    const expected = int16Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("int32", () => {
    const int32Array = new Int32Array([1, 2, 3]);
    const actual = pl.Series(int32Array).toArray();
    expect(actual).toEqual([...int32Array]);
  });
  test("int32:list", () => {
    const int32Arrays = [
      new Int32Array([1, 2, 3]),
      new Int32Array([33, 44, 55]),
    ];
    const actual = pl.Series(int32Arrays).toArray();
    const expected = int32Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("int64", () => {
    const int64Array = new BigInt64Array([1n, 2n, 3n]);
    const actual = pl.Series(int64Array).toArray();
    const expected = [...int64Array];
    expect(actual).toEqual(expected);
  });
  test("int64:list", () => {
    const int64Arrays = [
      new BigInt64Array([1n, 2n, 3n]),
      new BigInt64Array([33n, 44n, 55n]),
    ];
    const actual = pl.Series(int64Arrays).toArray();
    const expected = int64Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("uint8", () => {
    const uint8Array = new Uint8Array([1, 2, 3]);
    const actual = pl.Series(uint8Array).toArray();
    const expected = [...uint8Array];
    expect(actual).toEqual(expected);
  });
  test("uint8:list", () => {
    const uint8Arrays = [
      new Uint8Array([1, 2, 3]),
      new Uint8Array([33, 44, 55]),
    ];
    const actual = pl.Series(uint8Arrays).toArray();
    const expected = uint8Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("uint16", () => {
    const uint16Array = new Uint16Array([1, 2, 3]);
    const actual = pl.Series(uint16Array).toArray();
    const expected = [...uint16Array];
    expect(actual).toEqual(expected);
  });
  test("uint16:list", () => {
    const uint16Arrays = [
      new Uint16Array([1, 2, 3]),
      new Uint16Array([33, 44, 55]),
    ];
    const actual = pl.Series(uint16Arrays).toArray();
    const expected = uint16Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("uint32", () => {
    const uint32Array = new Uint32Array([1, 2, 3]);
    const actual = pl.Series(uint32Array).toArray();
    const expected = [...uint32Array];
    expect(actual).toEqual(expected);
  });
  test("uint32:list", () => {
    const uint32Arrays = [
      new Uint32Array([1, 2, 3]),
      new Uint32Array([33, 44, 55]),
    ];
    const actual = pl.Series(uint32Arrays).toArray();
    const expected = uint32Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("uint64", () => {
    const uint64Array = new BigUint64Array([1n, 2n, 3n]);
    const actual = pl.Series(uint64Array).toArray();
    const expected = [...uint64Array];
    expect(actual).toEqual(expected);
  });
  test("uint64:list", () => {
    const uint64Arrays = [
      new BigUint64Array([1n, 2n, 3n]),
      new BigUint64Array([33n, 44n, 55n]),
    ];
    const actual = pl.Series(uint64Arrays).toArray();
    const expected = uint64Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("float32", () => {
    const float32Array = new Float32Array([1, 2, 3]);
    const actual = pl.Series(float32Array).toArray();
    const expected = [...float32Array];
    expect(actual).toEqual(expected);
  });
  test("float32:list", () => {
    const float32Arrays = [
      new Float32Array([1, 2, 3]),
      new Float32Array([33, 44, 55]),
    ];
    const actual = pl.Series(float32Arrays).toArray();
    const expected = float32Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("float64", () => {
    const float64Array = new Float64Array([1, 2, 3]);
    const actual = pl.Series(float64Array).toArray();
    const expected = [...float64Array];
    expect(actual).toEqual(expected);
  });
  test("float64:list", () => {
    const float64Arrays = [
      new Float64Array([1, 2, 3]),
      new Float64Array([33, 44, 55]),
    ];
    const actual = pl.Series(float64Arrays).toArray();
    const expected = float64Arrays.map(i => [...i]);
    expect(actual).toEqual(expected);
  });
  test("invalid:list", () => {
    const float64Arrays = [
      new Float64Array([33, 44, 55]),
      new BigUint64Array([1n, 2n, 3n]),
    ];
    const fn = () =>  pl.Series(float64Arrays).toArray();
    expect(fn).toThrow();
  });
});
describe("series", () => {
  const chance = new Chance();

  describe("create series", () => {
    it.each`
      values
      ${[1, 1n]}
      ${["foo", 2]}
      ${[false, "false"]}
    `("does not allow multiple types", ({ values }) => {
      try {
        pl.Series("", values);
      } catch (err) {
        expect((err as Error).message).toBeDefined();
      }
    });

    it.each`
      values                    | dtype         | type
      ${["foo", "bar", "baz"]}  | ${"Utf8"}     | ${"string"}
      ${[1, 2, 3]}              | ${"Float64"}  | ${"number"}
      ${[1n, 2n, 3n]}           | ${"UInt64"}   | ${"bigint"}
      ${[true, false]}          | ${"Bool"}     | ${"boolean"}
      ${[]}                     | ${"Float64"}  | ${"empty"}
      ${[new Date(Date.now())]} | ${"Datetime"} | ${"Date"}
    `("defaults to $dtype for \"$type\"", ({ values, dtype}) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });

    it.each`
      values                   | dtype
      ${["foo", "bar", "baz"]} | ${"Utf8"}
      ${[1, 2, 3]}             | ${"Float64"}
      ${[1n, 2n, 3n]}          | ${"UInt64"}
    `("defaults to $dtype for $input", ({ values, dtype }) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });

    it.each`
    values | type
    ${[1, 2, 3]} | ${"number"}
    ${["1", "2", "3"]} | ${"string"}
    ${[1n, 2n, 3n]} | ${"bigint"}
    ${[true, false, null]} | ${"Option<bool>"}
    ${[1, 2, null]} | ${"Option<number>"}
    ${[1n, 2n, null]} |  ${"Option<bigint>"}
    ${[1.11, 2.22, 3.33, null]} |  ${"Option<float>"}
    ${new Int8Array([9, 10, 11])} | ${"Int8Array"}
    ${new Int16Array([12321, 2456, 22])} | ${"Int16Array"}
    ${new Int32Array([515121, 32411322, 32423])} | ${"Int32Array"}
    ${new Uint8Array([1, 2, 3, 4, 5, 6, 11])} | ${"Uint8Array"}
    ${new Uint16Array([1, 2, 3, 55, 11])} | ${"Uint16Array"}
    ${new Uint32Array([1123, 2, 3000, 12801, 99, 43242])} | ${"Uint32Array"}
    ${new BigInt64Array([1123n, 2n, 3000n, 12801n, 99n, 43242n])} | ${"BigInt64Array"}
    ${new BigUint64Array([1123n, 2n, 3000n, 12801n, 99n, 43242n])} | ${"BigUint64Array"}
    `("can be created from $type", ({values}) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect([...s]).toEqual([...values]);
    });
  });

  describe("math", () => {

    it("can add", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      let s = pl.Series("", [item]);
      s = s.add(other);
      expect(s[0]).toStrictEqual(item + other);
    });

    it("can subtract", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.sub(other);
      expect(s[0]).toStrictEqual(item - other);
    });

    it("can multiply", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.mul(other);
      expect(s[0]).toStrictEqual(item * other);
    });

    it("can divide", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.div(other);
      expect(s[0]).toStrictEqual(item / other);
    });

    it("can add two series", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});
      let s = pl.Series("", [item]);
      s = s.add(pl.Series("", [other]));
      expect(s[0]).toStrictEqual(item + other);
    });

    it("can subtract two series", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.sub(pl.Series("", [other]));
      expect(s[0]).toStrictEqual(item - other);
    });

    it("can multiply two series", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.mul(pl.Series("", [other]));
      expect(s[0]).toStrictEqual(item * other);
    });

    it("can divide two series", () => {
      const item = chance.natural({max: 100});
      const other = chance.natural({max: 100});

      let s = pl.Series("", [item]);
      s = s.div(pl.Series("", [other]));
      expect(s[0]).toStrictEqual(item / other);
    });
  });

  describe("comparator", () => {
    it("can perform 'eq", () => {
      const s =  pl.Series("", [1, 2, 3]).eq(1);
      expect([...s]).toEqual([true, false, false]);
    });
  });
});

describe("series", () => {
  const numSeries = () => pl.Series("foo", [1, 2, 3], pl.Int32);
  const fltSeries = () => pl.Series("float", [1, 2, 3], pl.Float64);
  const boolSeries = () =>  pl.Series("bool", [true, false, false]);
  const other = () => pl.Series("bar", [3, 4, 5], pl.Int32);

  const chance = new Chance();

  it.each`
  series        | getter
  ${numSeries()}  | ${"dtype"}
  ${numSeries()}  | ${"name"}
  ${numSeries()}  | ${"length"}
  `("$# $getter does not error", ({series, getter}) => {
    try {
      series[getter];
    } catch (err) {
      expect(err).not.toBeDefined();
    }
  });
  it.each`
  series          | method              | args
  ${numSeries()}  | ${"abs"}          | ${[]}
  ${numSeries()}  | ${"alias"}        | ${[chance.string()]}
  ${numSeries()}  | ${"append"}       | ${[other()]}
  ${numSeries()}  | ${"argMax"}       | ${[]}
  ${numSeries()}  | ${"argMin"}       | ${[]}
  ${numSeries()}  | ${"argSort"}      | ${[]}
  ${boolSeries()} | ${"argTrue"}      | ${[]}
  ${numSeries()}  | ${"argUnique"}    | ${[]}
  ${numSeries()}  | ${"cast"}         | ${[pl.UInt32]}
  ${numSeries()}  | ${"chunkLengths"} | ${[]}
  ${numSeries()}  | ${"clone"}        | ${[]}
  ${numSeries()}  | ${"cumMax"}       | ${[]}
  ${numSeries()}  | ${"cumMin"}       | ${[]}
  ${numSeries()}  | ${"cumProd"}      | ${[]}
  ${numSeries()}  | ${"cumSum"}       | ${[]}
  ${numSeries()}  | ${"describe"}| ${[]}
  ${numSeries()}  | ${"diff"}         | ${[]}
  ${numSeries()}  | ${"diff"}         | ${[{n: 1, nullBehavior: "drop"}]}
  ${numSeries()}  | ${"diff"}         | ${[{nullBehavior: "drop"}]}
  ${numSeries()}  | ${"diff"}         | ${[1, "drop"]}
  ${numSeries()}  | ${"dot"}          | ${[other()]}
  ${numSeries()}  | ${"dropNulls"}    | ${[]}
  ${numSeries()}  | ${"fillNull"}     | ${["zero"]}
  ${numSeries()}  | ${"fillNull"}     | ${[{strategy: "zero"}]}
  ${numSeries()}  | ${"filter"}       | ${[boolSeries()]}
  ${fltSeries()}  | ${"floor"}        | ${[]}
  ${numSeries()}  | ${"hasValidity"}  | ${[]}
  ${numSeries()}  | ${"hash"}         | ${[]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29, k2: 3}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29, k3: 1, k2: 3}]}
  ${numSeries()}  | ${"hash"}         | ${[1]}
  ${numSeries()}  | ${"hash"}         | ${[1, 2]}
  ${numSeries()}  | ${"hash"}         | ${[1, 2, 3]}
  ${numSeries()}  | ${"hash"}         | ${[1, 2, 3, 4]}
  ${numSeries()}  | ${"head"}         | ${[]}
  ${numSeries()}  | ${"head"}         | ${[1]}
  ${numSeries()}  | ${"inner"}        | ${[]}
  ${numSeries()}  | ${"interpolate"}  | ${[]}
  ${numSeries()}  | ${"isBoolean"}    | ${[]}
  ${numSeries()}  | ${"isDateTime"}   | ${[]}
  ${numSeries()}  | ${"isDuplicated"} | ${[]}
  ${fltSeries()}  | ${"isFinite"}     | ${[]}
  ${numSeries()}  | ${"isFirst"}      | ${[]}
  ${numSeries()}  | ${"isFloat"}      | ${[]}
  ${numSeries()}  | ${"isIn"}         | ${[other()]}
  ${numSeries()}  | ${"isIn"}         | ${[[1, 2, 3]]}
  ${fltSeries()}  | ${"isInfinite"}   | ${[]}
  ${numSeries()}  | ${"isNotNull"}    | ${[]}
  ${numSeries()}  | ${"isNull"}       | ${[]}
  ${numSeries()}  | ${"isNumeric"}    | ${[]}
  ${numSeries()}  | ${"isUnique"}     | ${[]}
  ${numSeries()}  | ${"isUtf8"}       | ${[]}
  ${numSeries()}  | ${"kurtosis"}     | ${[]}
  ${numSeries()}  | ${"kurtosis"}     | ${[{fisher: true, bias: true}]}
  ${numSeries()}  | ${"kurtosis"}     | ${[{bias: false}]}
  ${numSeries()}  | ${"kurtosis"}     | ${[{fisher: false}]}
  ${numSeries()}  | ${"kurtosis"}     | ${[false, false]}
  ${numSeries()}  | ${"kurtosis"}     | ${[false]}
  ${numSeries()}  | ${"len"}          | ${[]}
  ${numSeries()}  | ${"limit"}        | ${[]}
  ${numSeries()}  | ${"limit"}        | ${[2]}
  ${numSeries()}  | ${"max"}          | ${[]}
  ${numSeries()}  | ${"mean"}         | ${[]}
  ${numSeries()}  | ${"median"}       | ${[]}
  ${numSeries()}  | ${"min"}          | ${[]}
  ${numSeries()}  | ${"mode"}         | ${[]}
  ${numSeries()}  | ${"nChunks"}      | ${[]}
  ${numSeries()}  | ${"nUnique"}      | ${[]}
  ${numSeries()}  | ${"nullCount"}    | ${[]}
  ${numSeries()}  | ${"peakMax"}      | ${[]}
  ${numSeries()}  | ${"peakMin"}      | ${[]}
  ${numSeries()}  | ${"quantile"}     | ${[0.4]}
  ${numSeries()}  | ${"rank"}         | ${[]}
  ${numSeries()}  | ${"rank"}         | ${["average"]}
  ${numSeries()}  | ${"rechunk"}      | ${[]}
  ${numSeries()}  | ${"rechunk"}      | ${[true]}
  ${numSeries()}  | ${"rechunk"}      | ${[{inPlace: true}]}
  ${numSeries()}  | ${"rename"}       | ${["new name"]}
  ${numSeries()}  | ${"rename"}       | ${["new name", true]}
  ${numSeries()}  | ${"rename"}       | ${[{name: "new name"}]}
  ${numSeries()}  | ${"rename"}       | ${[{name: "new name", inPlace: true}]}
  ${numSeries()}  | ${"rename"}       | ${[{name: "new name"}]}
  ${numSeries()}  | ${"rollingMax"}   | ${[{windowSize: 1}]}
  ${numSeries()}  | ${"rollingMax"}   | ${[{windowSize: 1, weights: [.33]}]}
  ${numSeries()}  | ${"rollingMax"}   | ${[{windowSize: 1, weights: [.11], minPeriods: 1}]}
  ${numSeries()}  | ${"rollingMax"}   | ${[{windowSize: 1, weights: [.44], minPeriods: 1, center: false}]}
  ${numSeries()}  | ${"rollingMax"}   | ${[1]}
  ${numSeries()}  | ${"rollingMax"}   | ${[1, [.11]]}
  ${numSeries()}  | ${"rollingMax"}   | ${[1, [.11], 1]}
  ${numSeries()}  | ${"rollingMax"}   | ${[1, [.23], 1, true]}
  ${numSeries()}  | ${"rollingMean"}  | ${[{windowSize: 1}]}
  ${numSeries()}  | ${"rollingMean"}  | ${[{windowSize: 1, weights: [.33]}]}
  ${numSeries()}  | ${"rollingMean"}  | ${[{windowSize: 1, weights: [.11], minPeriods: 1}]}
  ${numSeries()}  | ${"rollingMean"}  | ${[{windowSize: 1, weights: [.44], minPeriods: 1, center: false}]}
  ${numSeries()}  | ${"rollingMean"}  | ${[1]}
  ${numSeries()}  | ${"rollingMean"}  | ${[1, [.11]]}
  ${numSeries()}  | ${"rollingMean"}  | ${[1, [.11], 1]}
  ${numSeries()}  | ${"rollingMean"}  | ${[1, [.23], 1, true]}
  ${numSeries()}  | ${"rollingMin"}   | ${[{windowSize: 1}]}
  ${numSeries()}  | ${"rollingMin"}   | ${[{windowSize: 1, weights: [.33]}]}
  ${numSeries()}  | ${"rollingMin"}   | ${[{windowSize: 1, weights: [.11], minPeriods: 1}]}
  ${numSeries()}  | ${"rollingMin"}   | ${[{windowSize: 1, weights: [.44], minPeriods: 1, center: false}]}
  ${numSeries()}  | ${"rollingMin"}   | ${[1]}
  ${numSeries()}  | ${"rollingMin"}   | ${[1, [.11]]}
  ${numSeries()}  | ${"rollingMin"}   | ${[1, [.11], 1]}
  ${numSeries()}  | ${"rollingMin"}   | ${[1, [.23], 1, true]}
  ${numSeries()}  | ${"rollingSum"}   | ${[{windowSize: 1}]}
  ${numSeries()}  | ${"rollingSum"}   | ${[{windowSize: 1, weights: [.33]}]}
  ${numSeries()}  | ${"rollingSum"}   | ${[{windowSize: 1, weights: [.11], minPeriods: 1}]}
  ${numSeries()}  | ${"rollingSum"}   | ${[{windowSize: 1, weights: [.44], minPeriods: 1, center: false}]}
  ${numSeries()}  | ${"rollingSum"}   | ${[1]}
  ${numSeries()}  | ${"rollingSum"}   | ${[1, [.11]]}
  ${numSeries()}  | ${"rollingSum"}   | ${[1, [.11], 1]}
  ${numSeries()}  | ${"rollingSum"}   | ${[1, [.23], 1, true]}
  ${numSeries()}  | ${"rollingVar"}   | ${[{windowSize: 1}]}
  ${numSeries()}  | ${"rollingVar"}   | ${[{windowSize: 1, weights: [.33]}]}
  ${numSeries()}  | ${"rollingVar"}   | ${[{windowSize: 1, weights: [.11], minPeriods: 1}]}
  ${numSeries()}  | ${"rollingVar"}   | ${[{windowSize: 1, weights: [.44], minPeriods: 1, center: false}]}
  ${numSeries()}  | ${"rollingVar"}   | ${[1]}
  ${numSeries()}  | ${"rollingVar"}   | ${[1, [.11]]}
  ${numSeries()}  | ${"rollingVar"}   | ${[1, [.11], 1]}
  ${numSeries()}  | ${"rollingVar"}   | ${[1, [.23], 1, true]}
  ${fltSeries()}  | ${"round"}        | ${[1]}
  ${numSeries()}  | ${"sample"}       | ${[1, null, true]}
  ${numSeries()}  | ${"sample"}       | ${[null, 1]}
  ${numSeries()}  | ${"sample"}       | ${[{n: 1}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.5}]}
  ${numSeries()}  | ${"sample"}       | ${[{n: 1, withReplacement: true}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.1, withReplacement: true}]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other()]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), true]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), false]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), {nullEqual:true}]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), {nullEqual:false}]}
  ${numSeries()}  | ${"set"}          | ${[boolSeries(), 2]}
  ${numSeries()}  | ${"setAtIdx"}     | ${[[0, 1], 1]}
  ${numSeries()}  | ${"shift"}        | ${[]}
  ${numSeries()}  | ${"shift"}        | ${[1]}
  ${numSeries()}  | ${"shift"}        | ${[{periods: 1}]}
  ${numSeries()}  | ${"shiftAndFill"} | ${[1, 2]}
  ${numSeries()}  | ${"shiftAndFill"} | ${[{periods: 1, fillValue: 2}]}
  ${numSeries()}  | ${"skew"}         | ${[]}
  ${numSeries()}  | ${"skew"}         | ${[true]}
  ${numSeries()}  | ${"skew"}         | ${[false]}
  ${numSeries()}  | ${"skew"}         | ${[{bias: true}]}
  ${numSeries()}  | ${"skew"}         | ${[{bias: false}]}
  ${numSeries()}  | ${"slice"}        | ${[1, 2]}
  ${numSeries()}  | ${"slice"}        | ${[{offset: 1, length: 2}]}
  ${numSeries()}  | ${"sort"}         | ${[]}
  ${numSeries()}  | ${"sort"}         | ${[false]}
  ${numSeries()}  | ${"sort"}         | ${[true]}
  ${numSeries()}  | ${"sort"}         | ${[{reverse: true}]}
  ${numSeries()}  | ${"sort"}         | ${[{reverse: false}]}
  ${numSeries()}  | ${"sum"}          | ${[]}
  ${numSeries()}  | ${"tail"}         | ${[]}
  ${numSeries()}  | ${"take"}         | ${[[1, 2]]}
  ${numSeries()}  | ${"takeEvery"}    | ${[1]}
  ${numSeries()}  | ${"toArray"}      | ${[]}
  ${numSeries()}  | ${"unique"}       | ${[]}
  ${numSeries()}  | ${"valueCounts"}  | ${[]}
  ${numSeries()}  | ${"zipWith"}      | ${[boolSeries(), other()]}
  `("$# $method is callable", ({series, method, args}) => {
    try {
      series[method](...args);
    } catch (err) {
      expect(err).not.toBeDefined();
    }
  });

  it.each`
  name               | actual                                               |  expected
  ${"dtype:Utf8"}    | ${pl.Series(["foo"]).dtype}                          | ${"Utf8"}
  ${"dtype:UInt64"}  | ${pl.Series([1n]).dtype}                             | ${"UInt64"}
  ${"dtype:Float64"} | ${pl.Series([1]).dtype}                              | ${"Float64"}
  ${"dtype"}         | ${pl.Series(["foo"]).dtype}                          | ${"Utf8"}
  ${"name"}          | ${pl.Series("a", ["foo"]).name}                      | ${"a"}
  ${"length"}        | ${pl.Series([1, 2, 3, 4]).length}                    | ${4}
  ${"abs"}           | ${pl.Series([1, 2, -3]).abs()}                       | ${pl.Series([1, 2, 3])}
  ${"alias"}         | ${pl.Series([1, 2, 3]).alias("foo")}                 | ${pl.Series("foo", [1, 2, 3])}
  ${"argMax"}        | ${pl.Series([1, 2, 3]).argMax()}                     | ${2}
  ${"argMin"}        | ${pl.Series([1, 2, 3]).argMin()}                     | ${0}
  ${"argSort"}       | ${pl.Series([3, 2, 1]).argSort()}                    | ${pl.Series([2, 1, 0])}
  ${"argTrue"}       | ${pl.Series([true, false]).argTrue()}                | ${pl.Series([0])}
  ${"argUnique"}     | ${pl.Series([1, 1, 2]).argUnique()}                  | ${pl.Series([0, 2])}
  ${"cast-Int16"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int16)}           | ${pl.Series("", [1, 1, 2], pl.Int16)}
  ${"cast-Int32"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int32)}           | ${pl.Series("", [1, 1, 2], pl.Int32)}
  ${"cast-Int64"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int64)}           | ${pl.Series("", [1n, 1n, 2n], pl.Int64)}
  ${"cast-UInt16"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt16)}          | ${pl.Series("", [1, 1, 2], pl.UInt16)}
  ${"cast-UInt32"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt32)}          | ${pl.Series("", [1, 1, 2], pl.UInt32)}
  ${"cast-UInt64"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt64)}          | ${pl.Series("", [1n, 1n, 2n])}
  ${"cast-Utf8"}     | ${pl.Series("", [1, 1, 2]).cast(pl.Utf8)}            | ${pl.Series("", ["1.0", "1.0", "2.0"])}
  ${"chunkLengths"}  | ${pl.Series([1, 2, 3]).chunkLengths()[0]}            | ${3}
  ${"clone"}         | ${pl.Series([1, 2, 3]).clone()}                      | ${pl.Series([1, 2, 3])}
  ${"concat"}        | ${pl.Series([1]).concat(pl.Series([2, 3]))}          | ${pl.Series([1, 2, 3])}
  ${"cumMax"}        | ${pl.Series([3, 2, 4]).cumMax()}                     | ${pl.Series([3, 3, 4])}
  ${"cumMin"}        | ${pl.Series([3, 2, 4]).cumMin()}                     | ${pl.Series([3, 2, 2])}
  ${"cumProd"}       | ${pl.Series("", [1, 2, 3], pl.Int32).cumProd()}      | ${pl.Series("", [1, 2, 6], pl.Int32)}
  ${"cumSum"}        | ${pl.Series("", [1, 2, 3], pl.Int32).cumSum()}       | ${pl.Series("", [1, 3, 6], pl.Int32)}
  ${"diff"}          | ${pl.Series([1, 2, 12]).diff(1, "drop").toJS()}      | ${pl.Series([1, 10]).toJS()}
  ${"diff"}          | ${pl.Series([1, 11]).diff(1, "ignore")}              | ${pl.Series("", [null, 10], pl.Float64, false)}
  ${"dropNulls"}     | ${pl.Series([1, null, 2]).dropNulls()}               | ${pl.Series([1, 2])}
  ${"dropNulls"}     | ${pl.Series([1, undefined, 2]).dropNulls()}          | ${pl.Series([1, 2])}
  ${"dropNulls"}     | ${pl.Series(["a", null, "f"]).dropNulls()}           | ${pl.Series(["a", "f"])}
  ${"fillNull:zero"} | ${pl.Series([1, null, 2]).fillNull("zero")}          | ${pl.Series([1, 0, 2])}
  ${"fillNull:one"}  | ${pl.Series([1, null, 2]).fillNull("one")}           | ${pl.Series([1, 1, 2])}
  ${"fillNull:max"}  | ${pl.Series([1, null, 5]).fillNull("max")}           | ${pl.Series([1, 5, 5])}
  ${"fillNull:min"}  | ${pl.Series([1, null, 5]).fillNull("min")}           | ${pl.Series([1, 1, 5])}
  ${"fillNull:mean"} | ${pl.Series([1, 1, null, 10]).fillNull("mean")}      | ${pl.Series([1, 1, 4, 10])}
  ${"fillNull:back"} | ${pl.Series([1, 1, null, 10]).fillNull("backward")}  | ${pl.Series([1, 1, 10, 10])}
  ${"fillNull:fwd"}  | ${pl.Series([1, 1, null, 10]).fillNull("forward")}   | ${pl.Series([1, 1, 1, 10])}
  ${"floor"}         | ${pl.Series([1.1, 2.2]).floor()}                     | ${pl.Series([1, 2])}
  ${"get"}           | ${pl.Series(["foo"]).get(0)}                         | ${"foo"}
  ${"get"}           | ${pl.Series([1, 2, 3]).get(2)}                       | ${3}
  ${"getIndex"}      | ${pl.Series(["a", "b", "c"]).getIndex(0)}            | ${"a"}
  ${"hasValidity"}   | ${pl.Series([1, null, 2]).hasValidity()}             | ${true}
  ${"hasValidity"}   | ${pl.Series([1, 1, 2]).hasValidity()}                | ${false}
  ${"hash"}          | ${pl.Series([1]).hash()}                             | ${pl.Series([6340063056640878722n])}
  ${"head"}          | ${pl.Series([1, 2, 3, 4, 5, 5, 5]).head()}           | ${pl.Series([1, 2, 3, 4, 5])}
  ${"head"}          | ${pl.Series([1, 2, 3, 4, 5, 5, 5]).head(2)}          | ${pl.Series([1, 2])}
  ${"interpolate"}   | ${pl.Series([1, 2, null, null, 5]).interpolate()}    | ${pl.Series([1, 2, 3, 4, 5])}
  ${"isBoolean"}     | ${pl.Series([1, 2, 3]).isBoolean()}                  | ${false}
  ${"isBoolean"}     | ${pl.Series([true, false]).isBoolean()}              | ${true}
  ${"isDateTime"}    | ${pl.Series([new Date(Date.now())]).isDateTime()}    | ${true}
  ${"isDuplicated"}  | ${pl.Series([1, 3, 3]).isDuplicated()}               | ${pl.Series([false, true, true])}
  ${"isFinite"}      | ${pl.Series([1.0, 3.1]).isFinite()}                  | ${pl.Series([true, true])}
  ${"isInfinite"}    | ${pl.Series([1.0, 2]).isInfinite()}                  | ${pl.Series([false, false])}
  ${"isNotNull"}     | ${pl.Series([1, null, undefined, 2]).isNotNull()}    | ${pl.Series([true, false, false, true])}
  ${"isNull"}        | ${pl.Series([1, null, undefined, 2]).isNull()}       | ${pl.Series([false, true, true, false])}
  ${"isNumeric"}     | ${pl.Series([1, 2, 3]).isNumeric()}                  | ${true}
  ${"isUnique"}      | ${pl.Series([1, 2, 3, 1]).isUnique()}                | ${pl.Series([false, true, true, false])}
  ${"isUtf8"}        | ${pl.Series([1, 2, 3, 1]).isUtf8()}                  | ${false}
  ${"kurtosis"}      | ${pl.Series([1, 2, 3, 3, 4]).kurtosis()?.toFixed(6)} | ${"-1.044379"}
  ${"isUtf8"}        | ${pl.Series(["foo"]).isUtf8()}                       | ${true}
  ${"len"}           | ${pl.Series([1, 2, 3, 4, 5]).len()}                  | ${5}
  ${"limit"}         | ${pl.Series([1, 2, 3, 4, 5, 5, 5]).limit(2)}         | ${pl.Series([1, 2])}
  ${"max"}           | ${pl.Series([-1, 10, 3]).max()}                      | ${10}
  ${"mean"}          | ${pl.Series([1, 1, 10]).mean()}                      | ${4}
  ${"median"}        | ${pl.Series([1, 1, 10]).median()}                    | ${1}
  ${"min"}           | ${pl.Series([-1, 10, 3]).min()}                      | ${-1}
  ${"nChunks"}       | ${pl.Series([1, 2, 3, 4, 4]).nChunks()}              | ${1}
  ${"nullCount"}     | ${pl.Series([1, null, null, 4, 4]).nullCount()}      | ${2}
  ${"peakMax"}       | ${pl.Series([9, 4, 5]).peakMax()}                    | ${pl.Series([true, false, true])}
  ${"peakMin"}       | ${pl.Series([4, 1, 3, 2, 5]).peakMin()}              | ${pl.Series([false, true, false, true, false])}
  ${"quantile"}      | ${pl.Series([1, 2, 3]).quantile(0.5)}                | ${2}
  ${"rank"}          | ${pl.Series([1, 2, 3, 2, 2, 3, 0]).rank("dense")}    | ${pl.Series("", [2, 3, 4, 3, 3, 4, 1], pl.UInt32)}
  ${"rename"}        | ${pl.Series([1, 3, 0]).rename("b")}                  | ${pl.Series("b", [1, 3, 0])}
  ${"rename"}        | ${pl.Series([1, 3, 0]).rename({name: "b"})}          | ${pl.Series("b", [1, 3, 0])}
  ${"rollingMax"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingMax(2)}          | ${pl.Series("", [null, 2, 3, 3, 2], pl.Float64)}
  ${"rollingMin"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingMin(2)}          | ${pl.Series("", [null, 1, 2, 2, 1], pl.Float64)}
  ${"rollingSum"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingSum(2)}          | ${pl.Series("", [null, 3, 5, 5, 3], pl.Float64)}
  ${"rollingMean"}   | ${pl.Series([1, 2, 3, 2, 1]).rollingMean(2)}         | ${pl.Series("", [null, 1.5, 2.5, 2.5, 1.5], pl.Float64)}
  ${"rollingVar"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingVar(2)[1]}       | ${0.5}
  ${"sample:n"}      | ${pl.Series([1, 2, 3, 4, 5]).sample(2).len()}        | ${2}
  ${"sample:frac"}   | ${pl.Series([1, 2, 3, 4, 5]).sample({frac:.4}).len()}| ${2}
  ${"shift"}         | ${pl.Series([1, 2, 3]).shift(1)}                     | ${pl.Series([null, 1, 2])}
  ${"shift"}         | ${pl.Series([1, 2, 3]).shift(-1)}                    | ${pl.Series([2, 3, null])}
  ${"skew"}          | ${pl.Series([1, 2, 3, 3, 0]).skew()?.toPrecision(6)} | ${"-0.363173"}
  ${"slice"}         | ${pl.Series([1, 2, 3, 3, 0]).slice(-3, 3)}           | ${pl.Series([3, 3, 0])}
  ${"slice"}         | ${pl.Series([1, 2, 3, 3, 0]).slice(1, 3)}            | ${pl.Series([2, 3, 3])}
  ${"sort"}          | ${pl.Series([4, 2, 5, 1, 2, 3, 3, 0]).sort()}        | ${pl.Series([0, 1, 2, 2, 3, 3, 4, 5])}
  ${"sort"}          | ${pl.Series([4, 2, 5, 0]).sort({reverse:true})}      | ${pl.Series([5, 4, 2, 0])}
  ${"sort"}          | ${pl.Series([4, 2, 5, 0]).sort({reverse:false})}     | ${pl.Series([0, 2, 4, 5])}
  ${"sum"}           | ${pl.Series([1, 2, 2, 1]).sum()}                     | ${6}
  ${"tail"}          | ${pl.Series([1, 2, 2, 1]).tail(2)}                   | ${pl.Series([2, 1])}
  ${"takeEvery"}     | ${pl.Series([1, 3, 2, 9, 1]).takeEvery(2)}           | ${pl.Series([1, 2, 1])}
  ${"take"}          | ${pl.Series([1, 3, 2, 9, 1]).take([0, 1, 3])}        | ${pl.Series([1, 3, 9])}
  ${"toArray"}       | ${pl.Series([1, 2, 3]).toArray()}                    | ${[1, 2, 3]}
  ${"unique"}        | ${pl.Series([1, 2, 3, 3]).unique().sort()}           | ${pl.Series([1, 2, 3])}
  ${"toFrame"}       | ${pl.Series("foo", [1, 2, 3]).toFrame().toJSON()}    | ${pl.DataFrame([pl.Series("foo", [1, 2, 3])]).toJSON()}
  ${"shiftAndFill"}  | ${pl.Series("foo", [1, 2, 3]).shiftAndFill(1, 99)}   | ${pl.Series("foo", [99, 1, 2])}
  `("$# $name: expected matches actual ", ({expected, actual}) => {
    expect(expected).toStrictEqual(actual);
  });
  it("set: expected matches actual", () => {
    const expected = pl.Series([99, 2, 3]);
    const mask = pl.Series([true, false, false]);
    const actual = pl.Series([1, 2, 3]).set(mask, 99);
    expect(actual).toSeriesEqual(expected);
  });
  it("set: throws error", () => {
    const mask = pl.Series([true]);
    expect(() => pl.Series([1, 2, 3]).set(mask, 99)).toThrow();
  });
  it("setAtIdx:array expected matches actual", () => {
    const expected = pl.Series([99, 2, 99]);
    const actual = pl.Series([1, 2, 3]).setAtIdx([0, 2], 99);
    expect(actual).toSeriesEqual(expected);
  });
  it("setAtIdx:series expected matches actual", () => {
    const expected = pl.Series([99, 2, 99]);
    const indices = pl.Series([0, 2]);
    const actual = pl.Series([1, 2, 3])
      .setAtIdx(indices, 99);
    expect(actual).toSeriesEqual(expected);
  });
  it("setAtIdx: throws error", () => {
    const mask = pl.Series([true]);
    expect(() => pl.Series([1, 2, 3]).set(mask, 99)).toThrow();
  });
  it.each`
  name | fn | errorType
  ${"isFinite"} | ${pl.Series(["foo"]).isFinite} | ${InvalidOperationError}
  ${"isInfinite"} | ${pl.Series(["foo"]).isInfinite} | ${InvalidOperationError}
  ${"rollingMax"} | ${() => pl.Series(["foo"]).rollingMax(null as any)} | ${Error}
  ${"sample"} | ${() => pl.Series(["foo"]).sample(null as any)} | ${Error}
  `("$# $name throws an error ", ({fn, errorType}) => {
    expect(fn).toThrow(errorType);
  });
});

describe("StringFunctions", () => {
  it.each`
  name               | actual                                           |  expected
  ${"toUpperCase"}   | ${pl.Series(["foo"]).str.toUpperCase()}          | ${pl.Series(["FOO"])}
  ${"lstrip"}        | ${pl.Series(["  foo"]).str.lstrip()}             | ${pl.Series(["foo"])}
  ${"rstrip"}        | ${pl.Series(["foo   "]).str.rstrip()}            | ${pl.Series(["foo"])}
  ${"toLowerCase"}   | ${pl.Series(["FOO"]).str.toLowerCase()}          | ${pl.Series(["foo"])}
  ${"contains"}      | ${pl.Series(["f1", "f0"]).str.contains(/[0]/)}   | ${pl.Series([false, true])}
  ${"lengths"}       | ${pl.Series(["apple", "ham"]).str.lengths()}     | ${pl.Series([5, 3])}
  ${"slice"}         | ${pl.Series(["apple", "ham"]).str.slice(1)}      | ${pl.Series(["pple", "am"])}
  `("$# $name expected matches actual", ({expected, actual}) => {

    expect(expected).toStrictEqual(actual);
  });
});
