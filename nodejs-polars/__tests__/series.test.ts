/* eslint-disable newline-per-chained-call */
import pl from "@polars";
import Chance from "chance";

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
      ${[]}                     | ${"Float32"}  | ${"empty"}
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
    ${[1,2,3]} | ${"number"}
    ${["1","2","3"]} | ${"string"}
    ${[1n,2n,3n]} | ${"bigint"}
    ${[true, false, null]} | ${"Option<bool>"}
    ${[1,2,null]} | ${"Option<number>"}
    ${[1n,2n,null]} |  ${"Option<bigint>"}
    ${[1.11,2.22,3.33, null]} |  ${"Option<float>"}
    ${new Int8Array([9,10,11])} | ${"Int8Array"}
    ${new Int16Array([12321,2456,22])} | ${"Int16Array"}
    ${new Int32Array([515121,32411322,32423])} | ${"Int32Array"}
    ${new Uint8Array([1,2,3,4,5,6,11])} | ${"Uint8Array"}
    ${new Uint16Array([1,2,3,55,11])} | ${"Uint16Array"}
    ${new Uint32Array([1123,2,3000,12801,99,43242])} | ${"Uint32Array"}
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
      const s =  pl.Series("", [1,2,3]).eq(1);
      expect([...s]).toEqual([true, false, false]);
    });
  });
});

describe("series", () => {
  const numSeries = () => pl.Series("foo", [1,2,3], pl.Int64);
  const fltSeries = () => pl.Series("float", [1,2,3], pl.Float64);
  const boolSeries = () =>  pl.Series("bool", [true, false, false]);
  const other = () => pl.Series("bar", [3,4,5], pl.Int64);

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
  ${numSeries()}  | ${"clip"}         | ${[{min:1, max: 2}]}
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
  ${numSeries()}  | ${"explode"}      | ${[]}
  ${numSeries()}  | ${"fillNull"}     | ${["zero"]}
  ${numSeries()}  | ${"fillNull"}     | ${[{strategy: "zero"}]}
  ${numSeries()}  | ${"filter"}       | ${[]}
  ${numSeries()}  | ${"floor"}        | ${[]}
  ${numSeries()}  | ${"hasValidity"}  | ${[]}
  ${numSeries()}  | ${"hash"}         | ${[]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29, k2: 3}]}
  ${numSeries()}  | ${"hash"}         | ${[{k0: 10, k1: 29, k3: 1, k2: 3}]}
  ${numSeries()}  | ${"hash"}         | ${[1]}
  ${numSeries()}  | ${"hash"}         | ${[1,2]}
  ${numSeries()}  | ${"hash"}         | ${[1,2,3]}
  ${numSeries()}  | ${"hash"}         | ${[1,2,3,4]}
  ${numSeries()}  | ${"head"}         | ${[]}
  ${numSeries()}  | ${"head"}         | ${[1]}
  ${numSeries()}  | ${"inner"}        | ${[]}
  ${numSeries()}  | ${"interpolate"}  | ${[]}
  ${numSeries()}  | ${"isBoolean"}    | ${[]}
  ${numSeries()}  | ${"isDatetime"}   | ${[]}
  ${numSeries()}  | ${"isDuplicated"} | ${[]}
  ${fltSeries()}  | ${"isFinite"}     | ${[]}
  ${numSeries()}  | ${"isFirst"}      | ${[]}
  ${numSeries()}  | ${"isFloat"}      | ${[]}
  ${numSeries()}  | ${"isIn"}         | ${[]}
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
  ${numSeries()}  | ${"reinterpret"}  | ${[]}
  ${numSeries()}  | ${"reinterpret"}  | ${[true]}
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
  ${numSeries()}  | ${"round"}        | ${[1]}
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
  ${numSeries()}  | ${"setAtIdx"}     | ${[0, 1]}
  ${numSeries()}  | ${"setAtIdx"}     | ${[1, 2]}
  ${numSeries()}  | ${"shift"}        | ${[]}
  ${numSeries()}  | ${"shift"}        | ${[1]}
  ${numSeries()}  | ${"shift"}        | ${[{periods: 1}]}
  ${numSeries()}  | ${"shiftAndFill"} | ${[1, 2]}
  ${numSeries()}  | ${"shiftAndFill"} | ${[{periods: 1, fillValue: 2}]}
  ${numSeries()}  | ${"shiftAndFill"} | ${[{periods: 1, fillValue: 2}]}
  ${numSeries()}  | ${"skew"}         | ${[]}
  ${numSeries()}  | ${"skew"}         | ${[true]}
  ${numSeries()}  | ${"skew"}         | ${[false]}
  ${numSeries()}  | ${"skew"}         | ${[{bias: true}]}
  ${numSeries()}  | ${"skew"}         | ${[{bias: false}]}
  ${numSeries()}  | ${"slice"}        | ${[1, 2]}
  ${numSeries()}  | ${"slice"}        | ${[{start: 1, length: 2}]}
  ${numSeries()}  | ${"sort"}         | ${[]}
  ${numSeries()}  | ${"sort"}         | ${[false]}
  ${numSeries()}  | ${"sort"}         | ${[true]}
  ${numSeries()}  | ${"sort"}         | ${[{reverse: true}]}
  ${numSeries()}  | ${"sort"}         | ${[{reverse: false}]}
  ${numSeries()}  | ${"sqrt"}         | ${[]}
  ${numSeries()}  | ${"sum"}          | ${[]}
  ${numSeries()}  | ${"tail"}         | ${[]}
  ${numSeries()}  | ${"take"}         | ${[[1,2]]}
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
  name              | expected                                      |  actual
  ${"dtype"}        | ${pl.Series(["foo"]).dtype}                   | ${"Utf8"}
  ${"name"}         | ${pl.Series("a", ["foo"]).name}               | ${"a"}
  ${"length"}       | ${pl.Series([1,2,3,4]).length}                | ${4}
  ${"abs"}          | ${pl.Series([1,2,-3]).abs()}                  | ${pl.Series([1,2,3])}
  ${"alias"}        | ${pl.Series([1,2,3]).alias("foo")}            | ${pl.Series("foo", [1,2,3])}
  ${"argMax"}       | ${pl.Series([1,2,3]).argMax()}                | ${2}
  ${"argMin"}       | ${pl.Series([1,2,3]).argMin()}                | ${0}
  ${"argSort"}      | ${pl.Series([3,2,1]).argSort()}               | ${pl.Series([2,1,0])}
  ${"argTrue"}      | ${pl.Series([true, false]).argTrue()}         | ${pl.Series([0])}
  ${"argUnique"}    | ${pl.Series([1,1,2]).argUnique()}             | ${pl.Series([0, 2])}
  ${"cast-Int16"}   | ${pl.Series("",[1,1,2]).cast(pl.Int16)}       | ${pl.Series("", [1,1,2], pl.Int16)}
  ${"cast-Int32"}   | ${pl.Series("",[1,1,2]).cast(pl.Int32)}       | ${pl.Series("", [1,1,2], pl.Int32)}
  ${"cast-Int64"}   | ${pl.Series("",[1,1,2]).cast(pl.Int64)}       | ${pl.Series("", [1,1,2], pl.Int64)}
  ${"cast-UInt16"}  | ${pl.Series("",[1,1,2]).cast(pl.UInt16)}      | ${pl.Series("", [1,1,2], pl.UInt16)}
  ${"cast-UInt32"}  | ${pl.Series("",[1,1,2]).cast(pl.UInt32)}      | ${pl.Series("", [1,1,2], pl.UInt32)}
  ${"cast-UInt64"}  | ${pl.Series("",[1,1,2]).cast(pl.UInt64)}      | ${pl.Series("", [1n,1n,2n])}
  ${"cast-Utf8"}    | ${pl.Series("",[1,1,2]).cast(pl.Utf8)}        | ${pl.Series("", ["1.0","1.0","2.0"])}
  ${"chunkLengths"} | ${pl.Series([1,2,3]).chunkLengths()[0]}       | ${3}
  ${"clone"}        | ${pl.Series([1,2,3]).clone()}                 | ${pl.Series([1,2,3])}
  ${"concat"}       | ${pl.Series([1]).concat(pl.Series([2,3]))}    | ${pl.Series([1,2,3])}
  ${"cumMax"}       | ${pl.Series([3,2,4]).cumMax()}                | ${pl.Series([3,3,4])}
  ${"cumMin"}       | ${pl.Series([3,2,4]).cumMin()}                | ${pl.Series([3,2,2])}
  ${"cumProd"}      | ${pl.Series("", [1,2,3], pl.Int32).cumProd()} | ${pl.Series("", [1,2,6], pl.Int32)}
  ${"cumSum"}       | ${pl.Series("", [1,2,3], pl.Int32).cumSum()}  | ${pl.Series("", [1,3,6], pl.Int32)}
  ${"diff"}         | ${pl.Series([1,2,12]).diff(1, "drop").toJS()} | ${pl.Series([1,10]).toJS()}
  ${"diff"}         | ${pl.Series([1,11]).diff(1, "ignore")}        | ${pl.Series("", [null, 10], pl.Float64, false)}
  ${"dropNulls"}    | ${pl.Series([1, null,2]).dropNulls()}         | ${pl.Series([1,2])}
  ${"dropNulls"}    | ${pl.Series([1, undefined,2]).dropNulls()}    | ${pl.Series([1,2])}
  ${"dropNulls"}    | ${pl.Series(["a", null,"f"]).dropNulls()}     | ${pl.Series(["a", "f"])}
  ${"dropNulls"}    | ${pl.Series([[1, 2], [3, 4]]).explode()}      | ${pl.Series([1,2,3,4])}
  `("$# $name: expected matches actual ", ({expected, actual}) => {
    expect(expected).toStrictEqual(actual);
  });

});