/* eslint-disable newline-per-chained-call */
import pl from "@polars";
import {Stream} from "stream";
import fs from "fs";
describe("dataframe", () => {
  const df = pl.DataFrame([
    pl.Series("foo", [1, 2, 9], pl.Int16),
    pl.Series("bar", [6, 2, 8], pl.Int16),
  ]);

  test("dtypes", () => {
    const expected = [pl.Float64, pl.Utf8];
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).dtypes;
    expect(actual).toEqual(expected);
  });
  test("height", () => {
    const expected = 3;
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).height;
    expect(actual).toEqual(expected);
  });
  test("width", () => {
    const expected = 2;
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).width;
    expect(actual).toEqual(expected);
  });
  test("shape", () => {
    const expected = {height: 3, width: 2};
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).shape;
    expect(actual).toEqual(expected);
  });
  test("get columns", () => {
    const expected = ["a", "b"];
    const actual = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}).columns;
    expect(actual).toEqual(expected);
  });
  test("set columns", () => {
    const expected = ["d", "e"];
    const df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]});
    df.columns = expected;

    expect(df.columns).toEqual(expected);
  });
  test("clone", () => {
    const expected = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]});
    const actual = expected.clone();
    expect(actual).toFrameEqual(expected);
  });
  test.skip("describe", () => {
    const actual = pl.DataFrame({
      "a": [1, 2, 3],
      "b": ["a", "b", "c"],
      "c": [true, true, false]
    }).describe();
    const expected = pl.DataFrame({
      "describe": ["mean", "std", "min", "max", "median"],
      "a": [2, 1, 1, 3, 2],
      "b": [null, null, null, null, null],
      "c": [null, null, 0, 1, null]
    });

    expect(actual).toFrameEqual(expected);
  });
  test("drop", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    const actual = df.drop("apple");
    expect(actual).toFrameEqual(expected);
  });
  test("drop: array", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
    });
    const actual = df.drop(["apple", "ham"]);
    expect(actual).toFrameEqual(expected);
  });
  test("drop: ...rest", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
      "apple": ["a", "b", "c"]
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
    });
    const actual = df.drop("apple", "ham");
    expect(actual).toFrameEqual(expected);
  });
  test("unique", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 3],
      "bar": [1, 2, 2, 4],
      "ham": ["a", "d", "d", "c"],
    }).unique();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [1, 2, 4],
      "ham": ["a", "d", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  test("unique:subset", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 2, 2],
      "bar": [1, 2, 2, 2],
      "ham": ["a", "b", "c", "c"],
    }).unique({subset: ["foo", "ham"]});
    const expected = pl.DataFrame({
      "foo": [1, 2, 2],
      "bar": [1, 2, 2],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqualIgnoringOrder(expected);
  });
  // run this test 100 times to make sure it is deterministic.
  test("unique:maintainOrder", () => {
    Array.from({length:100}).forEach(() => {
      const actual = pl.DataFrame({
        "foo": [0, 1, 2, 2, 2],
        "bar": [0, 1, 2, 2, 2],
        "ham": ["0", "a", "b", "b", "b"],
      }).unique({maintainOrder: true});

      const expected = pl.DataFrame({
        "foo": [0, 1, 2],
        "bar": [0, 1, 2],
        "ham": ["0", "a", "b"],
      });
      expect(actual).toFrameEqual(expected);
    });
  });
  // run this test 100 times to make sure it is deterministic.
  test("unique:maintainOrder:single subset", () => {
    Array.from({length:100}).forEach(() => {
      const actual = pl.DataFrame({
        "foo": [0, 1, 2, 2, 2],
        "bar": [0, 1, 2, 2, 2],
        "ham": ["0", "a", "b", "c", "d"],
      }).unique({maintainOrder: true, subset: "foo"});

      const expected = pl.DataFrame({
        "foo": [0, 1, 2],
        "bar": [0, 1, 2],
        "ham": ["0", "a", "b"],
      });
      expect(actual).toFrameEqual(expected);
    });
  });
  // run this test 100 times to make sure it is deterministic.
  test("unique:maintainOrder:multi subset", () => {
    Array.from({length:100}).forEach(() => {
      const actual = pl.DataFrame({
        "foo": [0, 1, 2, 2, 2],
        "bar": [0, 1, 2, 2, 2],
        "ham": ["0", "a", "b", "c", "c"],
      }).unique({maintainOrder: true, subset: ["foo", "ham"]});

      const expected = pl.DataFrame({
        "foo": [0, 1, 2, 2],
        "bar": [0, 1, 2, 2],
        "ham": ["0", "a", "b", "c"],
      });
      expect(actual).toFrameEqual(expected);
    });
  });
  test("dropNulls", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).dropNulls();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("dropNulls subset", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).dropNulls("foo");
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("explode", () => {
    const actual = pl.DataFrame({
      "letters": ["c", "a"],
      "nrs": [[1, 2], [1, 3]]
    }).explode("nrs");

    const expected = pl.DataFrame({
      "letters": ["c", "c", "a", "a"],
      "nrs": [1, 2, 1, 3]
    });

    expect(actual).toFrameEqual(expected);
  });
  test("fillNull:zero", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).fillNull("zero");
    const expected = pl.DataFrame({
      "foo": [1, 0, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("fillNull:one", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    }).fillNull("one");
    const expected = pl.DataFrame({
      "foo": [1, 1, 2, 3],
      "bar": [6.0, .5, 7.0, 8.0],
      "ham": ["a", "d", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  // test.todo("filter");
  test("findIdxByName", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).findIdxByName("ham");
    const expected = 2;
    expect(actual).toEqual(expected);
  });
  // test("fold:single column", () => {
  //   const expected = pl.Series([1, 2, 3]);
  //   const df = pl.DataFrame([expected]);
  //   const actual = df.fold((a, b) => a.concat(b));
  //   expect(actual).toSeriesEqual(expected);
  // });
  // test("fold", () => {
  //   const s1 = pl.Series([1, 2, 3]);
  //   const s2 = pl.Series([4, 5, 6]);
  //   const s3 = pl.Series([7, 8, 1]);
  //   const expected = pl.Series("foo", [true, true, false]);
  //   const df = pl.DataFrame([s1, s2, s3]);
  //   const actual = df.fold((a, b) => a.lessThan(b)).alias("foo");
  //   expect(actual).toSeriesEqual(expected);
  // });
  // test("fold-again", () => {
  //   const s1 = pl.Series([1, 2, 3]);
  //   const s2 = pl.Series([4, 5, 6]);
  //   const s3 = pl.Series([7, 8, 1]);
  //   const expected = pl.Series("foo", [12, 15, 10]);
  //   const df = pl.DataFrame([s1, s2, s3]);
  //   const actual = df.fold((a, b) => a.plus(b)).alias("foo");
  //   expect(actual).toSeriesEqual(expected);
  // });
  test("frameEqual:true", () => {
    const df  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
    });
    const other  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
    });
    const actual = df.frameEqual(other);
    expect(actual).toStrictEqual(true);
  });
  test("frameEqual:false", () => {
    const df  = pl.DataFrame({
      "foo": [3, 2, 22],
      "baz": [0, 7, 8],
    });
    const other  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
    });
    const actual = df.frameEqual(other);
    expect(actual).toStrictEqual(false);
  });
  test("frameEqual:nullEq:false", () => {
    const df  = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 7, 8],
    });
    const other  = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 7, 8],
    });
    const actual = df.frameEqual(other, false);
    expect(actual).toStrictEqual(false);
  });
  test("frameEqual:nullEq:true", () => {
    const df  = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 7, 8],
    });
    const other  = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 7, 8],
    });
    const actual = df.frameEqual(other, true);
    expect(actual).toStrictEqual(true);
  });
  test("getColumn", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).getColumn("ham");
    const expected = pl.Series("ham", ["a", "b", "c"]);
    expect(actual).toSeriesEqual(expected);
  });
  test("getColumns", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).getColumns();
    const expected = [
      pl.Series("foo", [1, 2, 3]),
      pl.Series("ham", ["a", "b", "c"])
    ];
    actual.forEach((a, idx) => {
      expect(a).toSeriesEqual(expected[idx]);
    });

  });
  test("groupBy", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).groupBy("foo");
    expect(actual.toString()).toEqual("GroupBy");
  });
  test("hashRows", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).hashRows();
    expect(actual.dtype).toEqual(pl.UInt64);
  });
  test.each([
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
  ])("hashRows:positional", (...args: any[]) => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).hashRows(...args);
    expect(actual.dtype).toEqual(pl.UInt64);
  });
  test.each([
    [{k0: 1}],
    [{k0: 1, k1: 2}],
    [{k0: 1, k1: 2, k2:3}],
    [{k0: 1, k1: 2, k2:3, k3:4}],
  ])("hashRows:named", (opts) => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).hashRows(opts);
    expect(actual.dtype).toEqual(pl.UInt64);
  });
  test("head", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).head(1);
    const expected  = pl.DataFrame({
      "foo": [1],
      "ham": ["a"]
    }).head(1);
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).hstack([pl.Series("apple", [10, 20, 30])]);
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": [10, 20, 30]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:df", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).hstack(pl.DataFrame([pl.Series("apple", [10, 20, 30])]));
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": [10, 20, 30]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("hstack:df", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    });
    actual.insertAtIdx(0, pl.Series("apple", [10, 20, 30]));
    const expected = pl.DataFrame({
      "apple": [10, 20, 30],
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("interpolate", () => {
    const df = pl.DataFrame({
      a: [1, null, 3]
    });
    const expected = pl.DataFrame({
      a: [1, 2, 3]
    });
    const actual = df.interpolate();
    expect(actual).toFrameEqual(expected);
  });
  test("isDuplicated", () => {
    const df = pl.DataFrame({
      a: [1, 2, 2],
      b: [1, 2, 2]
    });
    const expected = pl.Series([false, true, true]);
    const actual = df.isDuplicated();
    expect(actual).toSeriesEqual(expected);
  });
  test("isEmpty", () => {
    const df = pl.DataFrame({});
    expect(df.isEmpty()).toEqual(true);
  });
  test("isUnique", () => {
    const df = pl.DataFrame({
      a: [1, 2, 2],
      b: [1, 2, 2]
    });
    const expected = pl.Series([true, false, false]);
    const actual = df.isUnique();
    expect(actual).toSeriesEqual(expected);
  });

  test("lazy", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    }).lazy().collectSync();

    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("limit", () => {
    const actual  = pl.DataFrame({
      "foo": [1, 2, 3],
      "ham": ["a", "b", "c"]
    }).limit(1);
    const expected  = pl.DataFrame({
      "foo": [1],
      "ham": ["a"]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("max:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).max();
    expect(actual.row(0)).toEqual([3, 8, null]);
  });
  test("max:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).max(1);
    const expected  = pl.Series("foo", [6, 2, 9]);
    expect(actual).toSeriesEqual(expected);
  });
  test("mean:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [4, 4, 4],
      "bar": [1, 1, 10],
      "ham": ["a", "b", "a"]
    }).mean();
    expect(actual.row(0)).toEqual([4, 4, null]);
  });
  test("mean:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, null, 6],
      "bar": [6, 2, 8],
    }).mean(1, "ignore");
    const expected  = pl.Series("foo", [3.5, 2, 7]);
    expect(actual).toSeriesEqual(expected);
  });
  test("median", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).median();

    expect(actual.row(0)).toEqual([2, 7, null]);
  });
  test.todo("melt");
  test("min:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).min();
    expect(actual.row(0)).toEqual([1, 6, null]);
  });
  test("min:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).min(1);
    const expected  = pl.Series("foo", [1, 2, 8]);
    expect(actual).toSeriesEqual(expected);
  });
  test("nChunks", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    });
    expect(actual.nChunks()).toEqual(1);
  });
  test("nullCount", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, null],
      "bar": [6, 2, 8],
      "apple": [6, 2, 8],
      "pizza": [null, null, 8],
    }).nullCount();
    expect(actual.row(0)).toEqual([1, 0, 0, 2]);
  });
  test.todo("pipe");
  test("quantile", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).quantile(0.5);
    expect(actual.row(0)).toEqual([2, 7, null]);
  });
  test("rename", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).rename({
      "foo": "foo_new",
      "bar": "bar_new",
      "ham": "ham_new"
    });
    expect(actual.columns).toEqual(["foo_new", "bar_new", "ham_new"]);
  });
  test("replaceAtIdx", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    });
    const s = pl.Series("new_foo", [0.12, 2.0, 9.99]);
    actual.replaceAtIdx(0, s);
    expect(actual.getColumn("new_foo")).toSeriesEqual(s);
    expect(actual.findIdxByName("new_foo")).toEqual(0);
  });
  test("row", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).row(1);
    expect(actual).toEqual([2, 7, "b"]);
  });
  test("rows", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).rows();
    expect(actual).toEqual([
      [1, 6, "a"],
      [2, 7, "b"],
      [3, 8, "c"]
    ]);
  });
  test("sample:n", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).sample(2);
    expect(actual.height).toStrictEqual(2);
  });
  test("sample:default", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).sample();
    expect(actual.height).toStrictEqual(1);
  });
  test("sample:frac", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).sample({frac: 0.5});
    expect(actual.height).toStrictEqual(2);
  });
  test("sample:frac", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).sample({frac: 0.75});
    expect(actual.height).toStrictEqual(3);
  });
  test("sample:invalid", () => {
    const fn = () => pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).sample({} as any);
    expect(fn).toThrow(TypeError);
  });
  test("select:strings", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).select("ham", "foo");
    const foo = pl.Series("foo", [1, 2, 3, 1]);
    const ham = pl.Series("ham", ["a", "b", "c", null]);
    expect(actual.width).toStrictEqual(2);
    expect(actual.getColumn("foo")).toSeriesEqual(foo);
    expect(actual.getColumn("ham")).toSeriesEqual(ham);
  });
  test("select:expr", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
      "ham": ["a", "b", "c", null]
    }).select(pl.col("foo"), "ham");
    const foo = pl.Series("foo", [1, 2, 3, 1]);
    const ham = pl.Series("ham", ["a", "b", "c", null]);
    expect(actual.width).toStrictEqual(2);
    expect(actual.getColumn("foo")).toSeriesEqual(foo);
    expect(actual.getColumn("ham")).toSeriesEqual(ham);
  });
  test("shift:pos", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shift(1);
    const expected = pl.DataFrame({
      "foo": [null, 1, 2, 3],
      "bar": [null, 6, 7, 8],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shift:neg", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shift(-1);
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, null],
      "bar": [7, 8, 1, null],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shiftAndFill:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shiftAndFill(-1, 99);
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, 99],
      "bar": [7, 8, 1, 99],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shiftAndFill:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shiftAndFill({periods: -1, fillValue: 99});
    const expected = pl.DataFrame({
      "foo": [2, 3, 1, 99],
      "bar": [7, 8, 1, 99],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shrinkToFit:inPlace", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    });
    actual.shrinkToFit(true);
    const expected = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("shrinkToFit", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).shrinkToFit();
    const expected = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("slice:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).slice(0, 2);
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("slice:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).slice({offset: 0, length: 2});
    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:positional", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).sort("bar");
    const expected = pl.DataFrame({
      "foo": [1, 1, 2, 3],
      "bar": [1, 6, 7, 8],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:named", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, 1],
      "bar": [6, 7, 8, 1],
    }).sort({by: "bar", reverse: true});
    const expected = pl.DataFrame({
      "foo": [3, 2, 1, 1],
      "bar": [8, 7, 6, 1],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sort:multi-args", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3, -1],
      "bar": [6, 7, 8, 2],
      "baz": ["a", "b", "d", "A"],
    }).sort({
      by: [
        pl.col("baz"),
        pl.col("bar")
      ]
    });
    const expected = pl.DataFrame({
      "foo": [-1, 1, 2, 3],
      "bar": [2, 6, 7, 8],
      "baz": ["A", "a", "b", "d"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("std", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).std();
    const expected = pl.DataFrame([
      pl.Series("foo", [1]),
      pl.Series("bar", [1]),
      pl.Series("ham", [null], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("sum:axis:0", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).sum();
    expect(actual.row(0)).toEqual([6, 21, null]);
  });
  test("sum:axis:1", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).sum(1).rename("sum");
    const expected  = pl.Series("sum", [7, 4, 17]);
    expect(actual).toSeriesEqual(expected);
  });
  test("tail", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 9],
      "bar": [6, 2, 8],
    }).tail(1).row(0);
    const expected  = [9, 8];
    expect(actual).toEqual(expected);
  });
  test.skip("transpose", () => {
    const expected = pl.DataFrame({
      "column_0": [1, 1],
      "column_1": [2, 2],
      "column_2": [3, 3]
    });
    const df = pl.DataFrame({
      a: [1, 2, 3],
      b: [1, 2, 3]
    });
    const actual = df.transpose();
    expect(actual).toFrameEqual(expected);
  });
  test.skip("transpose:includeHeader", () => {
    const expected = pl.DataFrame({
      "column": ["a", "b"],
      "column_0": [1, 1],
      "column_1": [2, 2],
      "column_2": [3, 3]
    });
    const df = pl.DataFrame({
      a: [1, 2, 3],
      b: [1, 2, 3]
    });
    const actual = df.transpose({includeHeader:true});
    expect(actual).toFrameEqual(expected);
  });
  test.skip("transpose:columnNames", () => {
    const expected = pl.DataFrame({
      "a": [1, 1],
      "b": [2, 2],
      "c": [3, 3]
    });
    const df = pl.DataFrame({
      a: [1, 2, 3],
      b: [1, 2, 3]
    });
    const actual = df.transpose({includeHeader:false, columnNames: "abc"});
    expect(actual).toFrameEqual(expected);
  });
  test.skip("transpose:columnNames:generator", () => {
    const expected = pl.DataFrame({
      "col_0": [1, 1],
      "col_1": [2, 2],
      "col_2": [3, 3]
    });
    function *namesGenerator() {
      const baseName = "col_";
      let count = 0;
      while(true) {
        let name = `${baseName}${count}`;
        yield name;
        count++;
      }
    }
    const df = pl.DataFrame({
      a: [1, 2, 3],
      b: [1, 2, 3]
    });
    const actual = df.transpose({includeHeader:false, columnNames: namesGenerator()});
    expect(actual).toFrameEqual(expected);
  });
  test("var", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"]
    }).var();
    const expected = pl.DataFrame([
      pl.Series("foo", [1]),
      pl.Series("bar", [1]),
      pl.Series("ham", [null], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("vstack", () => {
    const df1 = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6, 7],
      "ham": ["a", "b"]
    });
    const df2 = pl.DataFrame({
      "foo": [3, 4],
      "bar": [8, 9],
      "ham": ["c", "d"]
    });

    const actual = df1.vstack(df2);
    const expected = pl.DataFrame([
      pl.Series("foo", [1, 2, 3, 4]),
      pl.Series("bar", [6, 7, 8, 9]),
      pl.Series("ham", ["a", "b", "c", "d"]),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumn:series", () => {
    const actual = df
      .clone()
      .withColumn(pl.Series("col_a", ["a", "a", "a"], pl.Utf8));
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumn:expr", () => {
    const actual = df
      .clone()
      .withColumn(pl.lit("a").alias("col_a"));

    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumns:series", () => {
    const actual = df
      .clone()
      .withColumns(
        pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
        pl.Series("col_b", ["b", "b", "b"], pl.Utf8)
      );
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
      pl.Series("col_b", ["b", "b", "b"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumns:expr", () => {
    const actual = df
      .clone()
      .withColumns(
        pl.lit("a").alias("col_a"),
        pl.lit("b").alias("col_b")
      );
    const expected =  pl.DataFrame([
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
      pl.Series("col_a", ["a", "a", "a"], pl.Utf8),
      pl.Series("col_b", ["b", "b", "b"], pl.Utf8),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumnRenamed:positional", () => {
    const actual = df
      .clone()
      .withColumnRenamed("foo", "apple");

    const expected =  pl.DataFrame([
      pl.Series("apple", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withColumnRenamed:named", () => {
    const actual = df
      .clone()
      .withColumnRenamed({existing: "foo", replacement: "apple"});

    const expected =  pl.DataFrame([
      pl.Series("apple", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
  test("withRowCount", () => {
    const actual = df
      .clone()
      .withRowCount();

    const expected =  pl.DataFrame([
      pl.Series("row_nr", [0, 1, 2], pl.UInt32),
      pl.Series("foo", [1, 2, 9], pl.Int16),
      pl.Series("bar", [6, 2, 8], pl.Int16),
    ]);
    expect(actual).toFrameEqual(expected);
  });
});
describe("join", () => {
  test("on", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"]
    });
    const actual = df.join(otherDF, {on: "ham"});

    const expected = pl.DataFrame({
      "foo": [1, 2],
      "bar": [6.0, 7.0],
      "ham": ["a", "b"],
      "apple": ["x", "y"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("on:multiple-columns", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo": [1, 10, 11],

    });
    const actual = df.join(otherDF, {on: ["ham", "foo"]});

    const expected = pl.DataFrame({
      "foo": [1],
      "bar": [6.0],
      "ham": ["a"],
      "apple": ["x"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("on:left&right", () => {
    const df = pl.DataFrame({
      "foo_left": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo_right": [1, 10, 11],

    });
    const actual = df.join(otherDF, {
      leftOn: ["foo_left", "ham"],
      rightOn: ["foo_right", "ham"]
    });

    const expected = pl.DataFrame({
      "foo_left": [1],
      "bar": [6.0],
      "ham": ["a"],
      "apple": ["x"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("on:left&right", () => {
    const df = pl.DataFrame({
      "foo_left": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo_right": [1, 10, 11],

    });
    const actual = df.join(otherDF, {
      leftOn: ["foo_left", "ham"],
      rightOn: ["foo_right", "ham"]
    });

    const expected = pl.DataFrame({
      "foo_left": [1],
      "bar": [6.0],
      "ham": ["a"],
      "apple": ["x"],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("on throws error if only 'leftOn' is specified", () => {
    const df = pl.DataFrame({
      "foo_left": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo_right": [1, 10, 11],

    });
    const f = () => df.join(otherDF, {
      leftOn: ["foo_left", "ham"],
    } as any);
    expect(f).toThrow(TypeError);
  });
  test("on throws error if only 'rightOn' is specified", () => {
    const df = pl.DataFrame({
      "foo_left": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo_right": [1, 10, 11],

    });
    const f = () => df.join(otherDF, {
      rightOn: ["foo_right", "ham"],
    } as any);
    expect(f).toThrow(TypeError);
  });
  test("on takes precedence over left&right", () => {
    const df = pl.DataFrame({
      "foo_left": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo_right": [1, 10, 11],

    });
    const actual = df.join(otherDF, {
      on: "ham",
      leftOn: ["foo_left", "ham"],
      rightOn: ["foo_right", "ham"],
    } as any);
    const expected = pl.DataFrame({
      "foo_left": [1, 2],
      "bar": [6.0, 7.0],
      "ham": ["a", "b"],
      "apple": ["x", "y"],
      "foo_right": [1, 10],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("how:left", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo": [1, 10, 11],

    });
    const actual = df.join(otherDF, {
      on: "ham",
      how: "left"
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": ["x", "y", null],
      "fooright": [1, 10, null],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("how:outer", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y"],
      "ham": ["a", "d"],
      "foo": [1, 10],

    });
    const actual = df.join(otherDF, {
      on: "ham",
      how: "outer"
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3, null],
      "bar": [6, 7, 8, null],
      "ham": ["a", "b", "c", "d"],
      "apple": ["x", null, null, "y"],
      "fooright": [1, null, null, 10],
    });
    expect(actual).toFrameEqual(expected);
  });
  test("suffix", () => {
    const df = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6.0, 7.0, 8.0],
      "ham": ["a", "b", "c"]
    });
    const otherDF = pl.DataFrame({
      "apple": ["x", "y", "z"],
      "ham": ["a", "b", "d"],
      "foo": [1, 10, 11],

    });
    const actual = df.join(otherDF, {
      on: "ham",
      how: "left",
      suffix: "_other"
    });
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [6, 7, 8],
      "ham": ["a", "b", "c"],
      "apple": ["x", "y", null],
      "foo_other": [1, 10, null],
    });
    expect(actual).toFrameEqual(expected);
  });
});
describe("io", () => {
  const df = pl.DataFrame([
    pl.Series("foo", [1, 2, 9], pl.Int16),
    pl.Series("bar", [6, 2, 8], pl.Int16),
  ]);
  test("writeCSV:string", () => {
    const actual = df.clone().writeCSV().toString();
    const expected  = "foo,bar\n1,6\n2,2\n9,8\n";
    expect(actual).toEqual(expected);
  });
  test("writeCSV:string:sep", () => {
    const actual = df.clone().writeCSV({sep: "X"}).toString();
    const expected  = "fooXbar\n1X6\n2X2\n9X8\n";
    expect(actual).toEqual(expected);
  });
  test("writeCSV:string:header", () => {
    const actual = df.clone().writeCSV({sep: "X", hasHeader: false}).toString();
    const expected  = "1X6\n2X2\n9X8\n";
    expect(actual).toEqual(expected);
  });
  test("writeCSV:stream", (done) => {
    const df = pl.DataFrame([
      pl.Series("foo", [1, 2, 3], pl.UInt32),
      pl.Series("bar", ["a", "b", "c"])
    ]);
    let body = "";
    const writeStream = new Stream.Writable({
      write(chunk, encoding, callback) {
        body += chunk;
        callback(null);

      }
    });
    df.writeCSV(writeStream);
    const newDF = pl.readCSV(body);
    expect(newDF).toFrameEqual(df);
    done();
  });
  test("writeCSV:path", (done) => {
    const df = pl.DataFrame([
      pl.Series("foo", [1, 2, 3], pl.UInt32),
      pl.Series("bar", ["a", "b", "c"])
    ]);
    df.writeCSV("./test.csv");
    const newDF = pl.readCSV("./test.csv");
    expect(newDF).toFrameEqual(df);
    fs.rmSync("./test.csv");
    done();
  });
  test("JSON.stringify", () => {
    const df = pl.DataFrame({
      foo: [1],
      bar: ["a"]
    });
    const expected = JSON.stringify({
      columns: [
        {
          name: "foo",
          datatype: "Float64",
          values: [1.0]
        },
        {
          name: "bar",
          datatype: "Utf8",
          values: ["a"]
        },
      ]
    });
    const actual = JSON.stringify(df);
    expect(actual).toEqual(expected);
  });
  test("toRecords", () => {
    const df = pl.DataFrame({
      foo: [1],
      bar: ["a"]
    });
    const expected = [
      {foo: 1.0, bar: "a"}
    ];
    const actual = df.toRecords();
    expect(actual).toEqual(expected);
  });
  test("toObject", () => {
    const expected = {
      foo: [1],
      bar: ["a"]
    };
    const df = pl.DataFrame(expected);

    const actual = df.toObject();
    expect(actual).toEqual(expected);
  });
  test("writeJSON:lines", () => {
    const rows = [
      {foo: 1.1},
      {foo: 3.1},
      {foo: 3.1}
    ];
    const actual = pl.DataFrame(rows).writeJSON({format:"lines"}).toString();
    const expected = rows.map(r => JSON.stringify(r)).join("\n").concat("\n");
    expect(actual).toEqual(expected);
  });
  test("writeJSON:stream", (done) => {
    const df = pl.DataFrame([
      pl.Series("foo", [1, 2, 3], pl.UInt32),
      pl.Series("bar", ["a", "b", "c"])
    ]);

    let body = "";
    const writeStream = new Stream.Writable({
      write(chunk, encoding, callback) {
        body += chunk;
        callback(null);

      }
    });
    df.writeJSON(writeStream, {format:"lines"});
    const newDF = pl.readJSON(body).select("foo", "bar");
    expect(newDF).toFrameEqual(df);
    done();
  });
  test("writeJSON:path", (done) => {
    const df = pl.DataFrame([
      pl.Series("foo", [1, 2, 3], pl.UInt32),
      pl.Series("bar", ["a", "b", "c"])
    ]);
    df.writeJSON("./test.json", {format:"lines"});
    const newDF = pl.readJSON("./test.json").select("foo", "bar");
    expect(newDF).toFrameEqual(df);
    fs.rmSync("./test.json");
    done();
  });

  test("writeJSON:rows", () => {
    const rows = [
      {foo: 1.1},
      {foo: 3.1},
      {foo: 3.1}
    ];
    const expected = JSON.stringify(rows);
    const actual = pl.readRecords(rows).writeJSON({format:"json"}).toString();
    expect(actual).toEqual(expected);
  });
  test("toSeries", () => {
    const s = pl.Series([1, 2, 3]);
    const actual = s.clone().toFrame().toSeries();
    expect(actual).toSeriesEqual(s);
  });
});
describe("create", () => {
  test("from empty", () => {
    const df = pl.DataFrame();
    expect(df.isEmpty()).toStrictEqual(true);
  });
  test("from empty-object", () => {
    const df = pl.DataFrame({});
    expect(df.isEmpty()).toStrictEqual(true);
  });
  test("all supported types", () => {
    const df = pl.DataFrame({
      bool: [true, null],
      date: pl.Series("", [new Date(), new Date()], pl.Date),
      date_nulls: pl.Series("", [null, new Date()], pl.Date),
      datetime: pl.Series("", [new Date(), new Date()]),
      datetime_nulls: pl.Series("", [null, new Date()]),
      string: ["a", "b"],
      string_nulls: [null, "a"],
      categorical: pl.Series("", ["one", "two"], pl.Categorical),
      categorical_nulls: pl.Series("", ["apple", null], pl.Categorical),
      list: [[1], [2, 3]],
      float_64: [1, 2],
      float_64_nulls: [1, null],
      uint_64: [1n, 2n],
      uint_64_null: [null, 2n],
      int_8_typed: Int8Array.from([1, 2]),
      int_16_typed: Int16Array.from([1, 2]),
      int_32_typed: Int32Array.from([1, 2]),
      int_64_typed: BigInt64Array.from([1n, 2n]),
      uint_8_typed: Uint8Array.from([1, 2]),
      uint_16_typed: Uint16Array.from([1, 2]),
      uint_32_typed: Uint32Array.from([1, 2]),
      uint_64_typed: BigUint64Array.from([1n, 2n]),
      float_32_typed: Float32Array.from([1.1, 2.2]),
      float_64_typed: Float64Array.from([1.1, 2.2]),
    });
    const expectedSchema = {
      bool: pl.Bool,
      date: pl.Date,
      date_nulls: pl.Date,
      datetime: pl.Datetime,
      datetime_nulls: pl.Datetime,
      string: pl.Utf8,
      string_nulls: pl.Utf8,
      categorical: pl.Categorical,
      categorical_nulls: pl.Categorical,
      list: pl.List,
      float_64: pl.Float64,
      float_64_nulls: pl.Float64,
      uint_64: pl.UInt64,
      uint_64_null: pl.UInt64,
      int_8_typed: pl.Int8,
      int_16_typed: pl.Int16,
      int_32_typed: pl.Int32,
      int_64_typed: pl.Int64,
      uint_8_typed: pl.UInt8,
      uint_16_typed: pl.UInt16,
      uint_32_typed: pl.UInt32,
      uint_64_typed: pl.UInt64,
      float_32_typed: pl.Float32,
      float_64_typed: pl.Float64,
    };
    const actual = df.schema;
    expect(actual).toEqual(expectedSchema);
  });
  test("from series-array", () => {
    const s1 = pl.Series("num", [1, 2, 3]);
    const s2 = pl.Series("date", [null, Date.now(), Date.now()], pl.Datetime);
    const df = pl.DataFrame([s1, s2]);
    expect(df.getColumn("num")).toSeriesEqual(s1);
    expect(df.getColumn("date")).toSeriesEqual(s2);
  });
  test("from arrays", () => {
    const columns = [
      [1, 2, 3],
      [1, 2, 2]
    ];

    const df = pl.DataFrame(columns);

    expect(df.getColumns()[0].toArray()).toEqual(columns[0]);
    expect(df.getColumns()[1].toArray()).toEqual(columns[1]);
  });
  test("from arrays: orient=col", () => {
    const columns = [
      [1, 2, 3],
      [1, 2, 2]
    ];

    const df = pl.DataFrame(columns, {orient: "col"});

    expect(df.getColumns()[0].toArray()).toEqual(columns[0]);
    expect(df.getColumns()[1].toArray()).toEqual(columns[1]);
  });
  test("from arrays: orient=row", () => {
    const rows = [
      [1, 2, 3],
      [1, 2, 2]
    ];

    const df = pl.readRecords(rows);

    expect(df.row(0).sort()).toEqual(rows[0]);
    expect(df.row(1).sort()).toEqual(rows[1]);
  });
  test("from arrays with column names: orient=col", () => {
    const columns = [
      [1, 2, 3],
      [1, 2, 2]
    ];

    const expectedColumnNames = ["a", "b"];
    const df = pl.DataFrame(columns, {columns: expectedColumnNames, orient: "col"});

    expect(df.getColumns()[0].toArray()).toEqual(columns[0]);
    expect(df.getColumns()[1].toArray()).toEqual(columns[1]);
    expect(df.columns).toEqual(expectedColumnNames);

  });
  test("from arrays: invalid ", () => {
    const columns = [
      [1, 2, 3],
      [1, 2, 2]
    ];

    const fn = () =>  pl.DataFrame(columns, {columns: ["a", "b", "c", "d"]});
    expect(fn).toThrow();
  });
  test("from arrays with columns, orient=row", () => {
    const rows = [
      [1, 2, 3],
      [1, 2, 2]
    ];
    const expectedColumns = ["a", "b", "c"];
    const df = pl.DataFrame(rows, {columns: expectedColumns, orient: "row"});

    expect(df.row(0).sort()).toEqual(rows[0].sort());
    expect(df.row(1).sort()).toEqual(rows[1].sort());
    expect(df.columns).toEqual(expectedColumns);
  });
  test("from row objects, inferred schema", () => {
    const rows = [
      {"num": 1, "date": new Date(Date.now()), "string": "foo1"},
      {"num": 1, "date": new Date(Date.now()), "string": 1}
    ];

    const expected = [
      rows[0],
      {num: 1, date: rows[1].date, string: rows[1].string.toString()}
    ];

    const df = pl.readRecords(rows, {inferSchemaLength: 1});
    expect(df.toRecords()).toEqual(expected);
  });
  test("from row objects, with schema", () => {
    const rows = [
      {"num": 1, "date": "foo", "string": "foo1"},
      {"num": 1, "date": "foo"}
    ];

    const expected = [
      {num: 1, date: rows[0].date.toString(), string: "foo1"},
      {num: 1, date: rows[1].date.toString(), string: null}
    ];

    const schema = {
      num: pl.Int32,
      date: pl.Utf8,
      string: pl.Utf8
    };
    const df = pl.readRecords(rows, {schema});
    expect(df.toRecords()).toEqual(expected);
    expect(df.schema).toEqual(schema);
  });

  test("from nulls", () => {
    const df = pl.DataFrame({"nulls": [null, null, null]});
    const expected = pl.DataFrame([pl.Series("nulls", [null, null, null], pl.Float64)]);
    expect(df).toFrameStrictEqual(expected);
  });
  test("from list types", () => {
    const int8List = [
      Int8Array.from([1, 2, 3]),
      Int8Array.from([2]),
      Int8Array.from([1, 1, 1])
    ];
    const expected: any = {
      "num_list": [[1, 2], [], [3, null]],
      "bool_list": [[true, null], [], [false]],
      "str_list": [["a", null], ["b", "c"], []],
      "bigint_list": [[1n], [2n, 3n], []],
      "int8_list": int8List
    };
    expected.int8_list = int8List.map(i => [...i]);
    const df = pl.DataFrame(expected);

    expect(df.toObject()).toEqual(expected);
  });
});
describe("arithmetic", () => {
  test("add", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).add(1);
    const expected = pl.DataFrame({
      "foo": [2, 3, 4],
      "bar": [5, 6, 7]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sub", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).sub(1);
    const expected = pl.DataFrame({
      "foo": [0, 1, 2],
      "bar": [3, 4, 5]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("div", () => {
    const actual = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [2, 2, 2]
    }).div(2);
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [1, 1, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("mul", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).mul(2);
    const expected = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [6, 4, 2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("rem", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).rem(2);
    const expected = pl.DataFrame({
      "foo": [1, 0, 1],
      "bar": [1, 0, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("plus", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).plus(1);
    const expected = pl.DataFrame({
      "foo": [2, 3, 4],
      "bar": [5, 6, 7]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("minus", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).minus(1);
    const expected = pl.DataFrame({
      "foo": [0, 1, 2],
      "bar": [3, 4, 5]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("divideBy", () => {
    const actual = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [2, 2, 2]
    }).divideBy(2);
    const expected = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [1, 1, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("multiplyBy", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).multiplyBy(2);
    const expected = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [6, 4, 2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("modulo", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).modulo(2);
    const expected = pl.DataFrame({
      "foo": [1, 0, 1],
      "bar": [1, 0, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("add:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).add(pl.Series([3, 2, 1]));
    const expected = pl.DataFrame({
      "foo": [4, 4, 4],
      "bar": [7, 7, 7]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("sub:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).sub(pl.Series([1, 2, 3]));
    const expected = pl.DataFrame({
      "foo": [0, 0, 0],
      "bar": [3, 3, 3]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("div:series", () => {
    const actual = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [2, 2, 2]
    }).div(pl.Series([2, 2, 1]));
    const expected = pl.DataFrame({
      "foo": [1, 2, 6],
      "bar": [1, 1, 2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("mul:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).mul(pl.Series([2, 3, 1]));
    const expected = pl.DataFrame({
      "foo": [2, 6, 3],
      "bar": [6, 6, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("rem:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).rem(pl.Series([1, 1, 3]));
    const expected = pl.DataFrame({
      "foo": [0, 0, 0],
      "bar": [0, 0, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("plus:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).plus(pl.Series([3, 2, 1]));
    const expected = pl.DataFrame({
      "foo": [4, 4, 4],
      "bar": [7, 7, 7]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("minus:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [4, 5, 6]
    }).minus(pl.Series([1, 2, 3]));
    const expected = pl.DataFrame({
      "foo": [0, 0, 0],
      "bar": [3, 3, 3]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("divideBy:series", () => {
    const actual = pl.DataFrame({
      "foo": [2, 4, 6],
      "bar": [2, 2, 2]
    }).divideBy(pl.Series([2, 2, 1]));
    const expected = pl.DataFrame({
      "foo": [1, 2, 6],
      "bar": [1, 1, 2]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("multiplyBy:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).multiplyBy(pl.Series([2, 3, 1]));
    const expected = pl.DataFrame({
      "foo": [2, 6, 3],
      "bar": [6, 6, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
  test("modulo:series", () => {
    const actual = pl.DataFrame({
      "foo": [1, 2, 3],
      "bar": [3, 2, 1]
    }).modulo(pl.Series([1, 1, 3]));
    const expected = pl.DataFrame({
      "foo": [0, 0, 0],
      "bar": [0, 0, 1]
    });
    expect(actual).toFrameEqual(expected);
  });
});

describe("meta", () => {
  test("array destructuring", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    const [col0] = df;
    expect(col0).toSeriesEqual(df.getColumn("os"));
    const [,version] = df;
    expect(version).toSeriesEqual(df.getColumn("version"));
    const [[row0Index0], [,row1Index1]] = df;
    expect(row0Index0).toStrictEqual("apple");
    expect(row1Index1).toStrictEqual(18.04);
  });
  test("object destructuring", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    const {os, version} = <any>df;
    expect(os).toSeriesEqual(df.getColumn("os"));
    expect(version).toSeriesEqual(df.getColumn("version"));
    const df2 = pl.DataFrame({
      fruits: ["apple", "orange"],
      cars: ["ford", "honda"]
    });
    const df3 = pl.DataFrame({...df, ...df2});
    const expected = df.hstack(df2);
    expect(df3).toFrameEqual(expected);
  });
  test("object bracket notation", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });

    expect(df["os"]).toSeriesEqual(df.getColumn("os"));
    expect(df["os"][1]).toStrictEqual("linux");

    df["os"] = pl.Series(["mac", "ubuntu"]);
    expect(df["os"][0]).toStrictEqual("mac");
  });
  test("object.keys shows column names", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    const keys = Object.keys(df);
    expect(keys).toEqual(df.columns);
  });
  test("object.values shows column values", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    const values = Object.values(df);
    expect(values[0]).toSeriesEqual(df["os"]);
    expect(values[1]).toSeriesEqual(df["version"]);
  });
  test("df rows", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    const actual = df[0][0];
    expect(actual).toStrictEqual(df.getColumn("os").get(0));
  });

  test("proxy:has", () => {
    const df = pl.DataFrame({
      os: ["apple", "linux"],
      version: [10.12, 18.04]
    });
    expect("os" in df).toBe(true);
  });
  test("inspect & toString", () => {
    const df = pl.DataFrame({
      a: [1]
    });
    const expected = `shape: (1, 1)
┌─────┐
│ a   │
│ --- │
│ f64 │
╞═════╡
│ 1.0 │
└─────┘`;
    const actualInspect = df[Symbol.for("nodejs.util.inspect.custom")]();
    const dfString = df.toString();
    expect(actualInspect).toStrictEqual(expected);
    expect(dfString).toStrictEqual(expected);
  });
});


describe("additional", () => {
  test("partitionBy", () => {
    const df = pl.DataFrame({
      label: ["a", "a", "b", "b"],
      value: [1, 2, 3, 4]
    });
    const dfs = df.partitionBy(["label"], true).map(df => df.toObject());
    const expected = [
      {
        label: ["a", "a"],
        value: [1, 2]
      },
      {
        label: ["b", "b"],
        value: [3, 4]
      }
    ];

    expect(dfs).toEqual(expected);
  });
  test("partitionBy with callback", () => {
    const df = pl.DataFrame({
      label: ["a", "a", "b", "b"],
      value: [1, 2, 3, 4]
    });
    const dfs = df.partitionBy(["label"], true, df => df.toObject());
    const expected = [
      {
        label: ["a", "a"],
        value: [1, 2]
      },
      {
        label: ["b", "b"],
        value: [3, 4]
      }
    ];

    expect(dfs).toEqual(expected);
  });
});
