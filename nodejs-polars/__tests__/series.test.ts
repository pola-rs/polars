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
    const expected = [[], [null], ["a"], [null], ["b", "c"]];
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

  // serde downcasts int64 to 'number'
  test("int64", () => {
    const int64Array = new BigInt64Array([1n, 2n, 3n]);
    const actual = pl.Series(int64Array).toArray();

    const expected = Array.from(int64Array).map((v: any) => parseInt(v));

    expect(actual).toEqual(expected);
  });
  // serde downcasts int64 to 'number'
  test("int64:list", () => {
    const int64Arrays = [
      new BigInt64Array([1n, 2n, 3n]),
      new BigInt64Array([33n, 44n, 55n]),
    ] as any;

    const actual = pl.Series(int64Arrays).toArray();
    const expected = [
      [1, 2, 3],
      [33, 44, 55]
    ];
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

});
describe("series", () => {
  const chance = new Chance();

  describe("create series", () => {


    it.each`
      values                    | dtype         | type
      ${["foo", "bar", "baz"]}  | ${pl.Utf8}     | ${"string"}
      ${[1, 2, 3]}              | ${pl.Float64}  | ${"number"}
      ${[1n, 2n, 3n]}           | ${pl.UInt64}   | ${"bigint"}
      ${[true, false]}          | ${pl.Bool}     | ${"boolean"}
      ${[]}                     | ${pl.Float64}  | ${"empty"}
      ${[new Date(Date.now())]} | ${pl.Datetime("ms")} | ${"Date"}
    `("defaults to $dtype for \"$type\"", ({ values, dtype}) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });

    it.each`
      values                   | dtype
      ${["foo", "bar", "baz"]} | ${pl.Utf8}
      ${[1, 2, 3]}             | ${pl.Float64}
      ${[1n, 2n, 3n]}          | ${pl.UInt64}
    `("defaults to $dtype for $input", ({ values, dtype }) => {
      const name = chance.string();
      const s = pl.Series(name, values);
      expect(s.name).toStrictEqual(name);
      expect(s.length).toStrictEqual(values.length);
      expect(s.dtype).toStrictEqual(dtype);
    });
  });

});
describe("series", () => {
  const numSeries = () => pl.Series("foo", [1, 2, 3], pl.Int32);
  const fltSeries = () => pl.Series("float", [1, 2, 3], pl.Float64);
  const boolSeries = () =>  pl.Series("bool", [true, false, false]);
  const other = () => pl.Series("bar", [3, 4, 5], pl.Int32);

  const chance = new Chance();

  // test("to/fromBinary round trip", () => {
  //   const s = pl.Series("serde", [1, 2, 3, 4, 5, 2]);
  //   const buf = s.toBinary();
  //   const actual = pl.Series.fromBinary(buf);
  //   expect(s).toStrictEqual(actual);
  // });
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
  ${numSeries()}  | ${"as"}           | ${[chance.string()]}
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
  ${numSeries()}  | ${"describe"}     | ${[]}
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
  ${numSeries()}  | ${"sample"}       | ${[]}
  ${numSeries()}  | ${"sample"}       | ${[1, null, true]}
  ${numSeries()}  | ${"sample"}       | ${[null, 1]}
  ${numSeries()}  | ${"sample"}       | ${[{n: 1}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.5}]}
  ${numSeries()}  | ${"sample"}       | ${[{n: 1, withReplacement: true}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.1, withReplacement: true}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.1, withReplacement: true, seed: 1n}]}
  ${numSeries()}  | ${"sample"}       | ${[{frac: 0.1, withReplacement: true, seed: 1}]}
  ${numSeries()}  | ${"sample"}       | ${[{n: 1, withReplacement: true, seed: 1}]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other()]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), true]}
  ${numSeries()}  | ${"seriesEqual"}  | ${[other(), false]}
  ${numSeries()}  | ${"set"}          | ${[boolSeries(), 2]}
  ${fltSeries()}  | ${"setAtIdx"}     | ${[[0, 1], 1]}
  ${numSeries()}  | ${"shift"}        | ${[]}
  ${numSeries()}  | ${"shift"}        | ${[1]}
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
  ${"dtype:Utf8"}    | ${pl.Series(["foo"]).dtype}                          | ${pl.Utf8}
  ${"dtype:UInt64"}  | ${pl.Series([1n]).dtype}                             | ${pl.UInt64}
  ${"dtype:Float64"} | ${pl.Series([1]).dtype}                              | ${pl.Float64}
  ${"dtype"}         | ${pl.Series(["foo"]).dtype}                          | ${pl.Utf8}
  ${"name"}          | ${pl.Series("a", ["foo"]).name}                      | ${"a"}
  ${"length"}        | ${pl.Series([1, 2, 3, 4]).length}                    | ${4}
  ${"abs"}           | ${pl.Series([1, 2, -3]).abs()}                       | ${pl.Series([1, 2, 3])}
  ${"alias"}         | ${pl.Series([1, 2, 3]).as("foo")}                    | ${pl.Series("foo", [1, 2, 3])}
  ${"alias"}         | ${pl.Series([1, 2, 3]).alias("foo")}                 | ${pl.Series("foo", [1, 2, 3])}
  ${"argMax"}        | ${pl.Series([1, 2, 3]).argMax()}                     | ${2}
  ${"argMin"}        | ${pl.Series([1, 2, 3]).argMin()}                     | ${0}
  ${"argSort"}       | ${pl.Series([3, 2, 1]).argSort()}                    | ${pl.Series([2, 1, 0])}
  ${"argTrue"}       | ${pl.Series([true, false]).argTrue()}                | ${pl.Series([0])}
  ${"argUnique"}     | ${pl.Series([1, 1, 2]).argUnique()}                  | ${pl.Series([0, 2])}
  ${"cast-Int16"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int16)}           | ${pl.Series("", [1, 1, 2], pl.Int16)}
  ${"cast-Int32"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int32)}           | ${pl.Series("", [1, 1, 2], pl.Int32)}
  ${"cast-Int64"}    | ${pl.Series("", [1, 1, 2]).cast(pl.Int64)}           | ${pl.Series("", [1, 1, 2], pl.Int64)}
  ${"cast-UInt16"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt16)}          | ${pl.Series("", [1, 1, 2], pl.UInt16)}
  ${"cast-UInt32"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt32)}          | ${pl.Series("", [1, 1, 2], pl.UInt32)}
  ${"cast-UInt64"}   | ${pl.Series("", [1, 1, 2]).cast(pl.UInt64)}          | ${pl.Series("", [1n, 1n, 2n])}
  ${"cast-Utf8"}     | ${pl.Series("", [1, 1, 2]).cast(pl.Utf8)}            | ${pl.Series("", ["1.0", "1.0", "2.0"])}
  ${"chunkLengths"}  | ${pl.Series([1, 2, 3]).chunkLengths()[0]}            | ${3}
  ${"clone"}         | ${pl.Series([1, 2, 3]).clone()}                      | ${pl.Series([1, 2, 3])}
  ${"concat"}        | ${pl.Series([1]).concat(pl.Series([2, 3]))}          | ${pl.Series([1, 2, 3])}
  ${"cumMax"}        | ${pl.Series([3, 2, 4]).cumMax()}                     | ${pl.Series([3, 3, 4])}
  ${"cumMin"}        | ${pl.Series([3, 2, 4]).cumMin()}                     | ${pl.Series([3, 2, 2])}
  ${"cumProd"}       | ${pl.Series("", [1, 2, 3], pl.Int32).cumProd()}      | ${pl.Series("", [1, 2, 6], pl.Int64)}
  ${"cumSum"}        | ${pl.Series("", [1, 2, 3], pl.Int32).cumSum()}       | ${pl.Series("", [1, 3, 6], pl.Int32)}
  ${"diff"}          | ${pl.Series([1, 2, 12]).diff(1, "drop").toObject()}  | ${pl.Series([1, 10]).toObject()}
  ${"diff"}          | ${pl.Series([1, 11]).diff(1, "ignore")}              | ${pl.Series("", [null, 10], pl.Float64)}
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
  ${"rollingMax"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingMax(2)}          | ${pl.Series("", [null, 2, 3, 3, 2], pl.Float64)}
  ${"rollingMin"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingMin(2)}          | ${pl.Series("", [null, 1, 2, 2, 1], pl.Float64)}
  ${"rollingSum"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingSum(2)}          | ${pl.Series("", [null, 3, 5, 5, 3], pl.Float64)}
  ${"rollingMean"}   | ${pl.Series([1, 2, 3, 2, 1]).rollingMean(2)}         | ${pl.Series("", [null, 1.5, 2.5, 2.5, 1.5], pl.Float64)}
  ${"rollingVar"}    | ${pl.Series([1, 2, 3, 2, 1]).rollingVar(2)[1]}       | ${0.5}
  ${"sample:n"}      | ${pl.Series([1, 2, 3, 4, 5]).sample(2).len()}        | ${2}
  ${"sample:frac"}   | ${pl.Series([1, 2, 3, 4, 5]).sample({frac:.4, seed:0}).len()}| ${2}
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
  ${"shiftAndFill"}  | ${pl.Series("foo", [1, 2, 3]).shiftAndFill(1, 99)}   | ${pl.Series("foo", [99, 1, 2])}
  `("$# $name: expected matches actual ", ({expected, actual}) => {
    if(pl.Series.isSeries(expected) && pl.Series.isSeries(actual)) {
      expect(actual).toSeriesEqual(expected);
    } else {
      expect(actual).toEqual(expected);
    }
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
  ${"isFinite"} | ${pl.Series(["foo"]).isFinite} | ${TypeError}
  ${"isInfinite"} | ${pl.Series(["foo"]).isInfinite} | ${TypeError}
  ${"rollingMax"} | ${() => pl.Series(["foo"]).rollingMax(null as any)} | ${Error}
  ${"sample"} | ${() => pl.Series(["foo"]).sample(null as any)} | ${Error}
  `("$# $name throws an error ", ({fn, errorType}) => {
    expect(fn).toThrow(errorType);
  });
  test("reinterpret", () => {
    const s = pl.Series("reinterpret", [1, 2], pl.Int64);
    const unsignedExpected = pl.Series("reinterpret", [1n, 2n], pl.UInt64);
    const signedExpected = pl.Series("reinterpret", [1, 2], pl.Int64);
    const unsigned = s.reinterpret(false);
    const signed = unsigned.reinterpret(true);

    expect(unsigned).toSeriesStrictEqual(unsignedExpected);
    expect(signed).toSeriesStrictEqual(signedExpected);
  });
  test("reinterpret:invalid", () => {
    const s = pl.Series("reinterpret", [1, 2]);
    const fn = () => s.reinterpret();
    expect(fn).toThrow();
  });
  test("extend", () => {
    const s = pl.Series("extended", [1], pl.UInt16);
    const expected = pl.Series("extended", [1, null, null], pl.UInt16);
    const actual = s.extend(null, 2);
    expect(actual).toSeriesStrictEqual(expected);
  });
  test("round invalid", () => {
    const s = pl.Series([true, false]);
    const fn = () => s.round(2);
    expect(fn).toThrow();
  });
  test("round:positional", () => {
    const s = pl.Series([1.1111, 2.2222]);
    const expected = pl.Series([1.11, 2.22]);
    const actual = s.round(2);
    expect(actual).toSeriesEqual(expected);
  });
  test("round:named", () => {
    const s = pl.Series([1.1111, 2.2222]);
    const expected = pl.Series([1.11, 2.22]);
    const actual = s.round({decimals: 2});
    expect(actual).toSeriesEqual(expected);
  });
});
describe("comparators & math", () => {
  test("add/plus", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([2, 3]);
    expect(s.add(1)).toSeriesEqual(expected);
    expect(s.plus(1)).toSeriesEqual(expected);
  });
  test("sub/minus", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([0, 1]);
    expect(s.sub(1)).toSeriesEqual(expected);
    expect(s.minus(1)).toSeriesEqual(expected);
  });
  test("mul/multiplyBy", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([10, 20]);
    expect(s.mul(10)).toSeriesEqual(expected);
    expect(s.multiplyBy(10)).toSeriesEqual(expected);
  });
  test("div/divideBy", () => {
    const s = pl.Series([2, 4]);
    const expected = pl.Series([1, 2]);
    expect(s.div(2)).toSeriesEqual(expected);
    expect(s.divideBy(2)).toSeriesEqual(expected);
  });
  test("div/divideBy", () => {
    const s = pl.Series([2, 4]);
    const expected = pl.Series([1, 2]);
    expect(s.div(2)).toSeriesEqual(expected);
    expect(s.divideBy(2)).toSeriesEqual(expected);
  });
  test("rem/modulo", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([1, 0]);
    expect(s.rem(2)).toSeriesEqual(expected);
    expect(s.modulo(2)).toSeriesEqual(expected);
  });
  test("eq/equals", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([true, false]);
    expect(s.eq(1)).toSeriesEqual(expected);
    expect(s.equals(1)).toSeriesEqual(expected);
  });
  test("neq/notEquals", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([false, true]);
    expect(s.neq(1)).toSeriesEqual(expected);
    expect(s.notEquals(1)).toSeriesEqual(expected);
  });
  test("gt/greaterThan", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([false, true]);
    expect(s.gt(1)).toSeriesEqual(expected);
    expect(s.greaterThan(1)).toSeriesEqual(expected);
  });
  test("gtEq/equals", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([true, true]);
    expect(s.gtEq(1)).toSeriesEqual(expected);
    expect(s.greaterThanEquals(1)).toSeriesEqual(expected);
  });
  test("lt/lessThan", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([false, false]);
    expect(s.lt(1)).toSeriesEqual(expected);
    expect(s.lessThan(1)).toSeriesEqual(expected);
  });
  test("ltEq/lessThanEquals", () => {
    const s = pl.Series([1, 2]);
    const expected = pl.Series([true, false]);
    expect(s.ltEq(1)).toSeriesEqual(expected);
    expect(s.lessThanEquals(1)).toSeriesEqual(expected);
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

  test("hex encode", () => {
    const s = pl.Series("strings", ["foo", "bar", null]);
    const expected = pl.Series("encoded", ["666f6f", "626172", null]);
    const encoded = s.str.encode("hex").alias("encoded");
    expect(encoded).toSeriesEqual(expected);
  });
  test("hex decode", () => {
    const s = pl.Series("encoded", ["666f6f", "626172", "invalid", null]);
    const expected = pl.Series("decoded", ["foo", "bar", null, null]);
    const decoded = s.str.decode("hex").alias("decoded");
    expect(decoded).toSeriesEqual(expected);
  });
  test("hex decode strict", () => {
    const s = pl.Series("encoded", ["666f6f", "626172", "invalid", null]);
    const fn0  = () => s.str.decode("hex", true).alias("decoded");
    const fn1  = () => s.str.decode({encoding: "hex", strict: true}).alias("decoded");
    expect(fn0).toThrow();
    expect(fn1).toThrow();
  });
  test("encode base64", () => {
    const s = pl.Series("strings", ["foo", "bar"]);
    const expected = pl.Series("encoded", ["Zm9v", "YmFy"]);
    const encoded = s.str.encode("base64").alias("encoded");
    expect(encoded).toSeriesEqual(expected);
  });
  test("base64 decode strict", () => {
    const s = pl.Series("encoded", ["Zm9v", "YmFy", "not base64 encoded", null]);
    const fn0  = () => s.str.decode("base64", true).alias("decoded");
    const fn1  = () => s.str.decode({encoding: "base64", strict: true}).alias("decoded");
    expect(fn0).toThrow();
    expect(fn1).toThrow();
  });
  test("base64 decode", () => {
    const s = pl.Series("encoded", ["Zm9v", "YmFy", "invalid", null]);
    const decoded = pl.Series("decoded", ["foo", "bar", null, null]);

    const actual =  s.str.decode("base64").alias("decoded");
    expect(actual).toSeriesEqual(decoded);
  });
});
