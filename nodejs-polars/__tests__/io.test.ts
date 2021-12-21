import pl from "@polars";
import path from "path";
// eslint-disable-next-line no-undef
const csvpath = path.resolve(__dirname, "../../examples/aggregate_multiple_files_in_chunks/datasets/foods1.csv");
// eslint-disable-next-line no-undef
const jsonpath = path.resolve(__dirname, "./examples/foods.json");

describe("io", () => {
  describe("read", () => {
    describe("csv", () => {
      it("can read from a csv file", () => {
        const df = pl.readCSV(csvpath);
        expect(df.shape).toStrictEqual({height: 27, width: 4});
      });
      it("can read from a csv file with options", () => {
        const df = pl.readCSV(csvpath, {hasHeader: false, startRows: 1, endRows: 4});
        expect(df.shape).toStrictEqual({height: 4, width: 4});
      });
      it("can read from a csv string", () => {
        const csvString = "foo,bar,baz\n1,2,3\n4,5,6\n";
        const df = pl.readCSV(csvString);
        expect(df.toCSV()).toEqual(csvString);
      });
      it("can parse datetimes", () => {
        const csv = `timestamp,open,high
2021-01-01 00:00:00,0.00305500,0.00306000
2021-01-01 00:15:00,0.00298800,0.00300400
2021-01-01 00:30:00,0.00298300,0.00300100
2021-01-01 00:45:00,0.00299400,0.00304000`;
        const df = pl.readCSV(csv);
        expect(df.dtypes).toEqual(["Datetime", "Float64", "Float64"]);
      });
      it.each`
      csv                         | nullValues
      ${"a,b,c\nna,b,c\na,na,c"}  | ${"na"}
      ${"a,b,c\nna,b,c\na,n/a,c"} | ${["na", "n/a"]}
      ${"a,b,c\nna,b,c\na,n/a,c"} | ${{"a": "na", "b": "n/a"}}
      `("can handle null values", ({csv, nullValues}) => {
        const df = pl.readCSV(csv, {nullValues});
        expect(df.getColumn("a")[0]).toBeNull();
        expect(df.getColumn("b")[1]).toBeNull();
      });
      it.todo("can read from a stream");
    });
    describe("json", () => {
      it("can read from a json file", () => {
        const df = pl.readJSON(jsonpath);
        expect(df.shape).toStrictEqual({height: 27, width: 4});
      });
      it("can specify read options", () => {
        const df = pl.readJSON({file: jsonpath, batchSize: 10, inferSchemaLength: 100});
        expect(df.shape).toStrictEqual({height: 27, width: 4});
      });
      it("can read from a json string", () => {
        const jsonString = JSON.stringify({foo: "bar"});
        const df = pl.readJSON(jsonString);
        expect(df.toJSON().replace("\n", "")).toEqual(jsonString);
      });
    });
  });
  describe.skip("scan", () => {
    describe.skip("csv", () => {
      it("can lazy load (scan) from a csv file", () => {
        const df = pl.scanCSV(csvpath).collectSync();
        expect(df.shape).toStrictEqual({height: 27, width: 4});
      });
      it("can lazy load (scan) from a csv file with options", () => {
        const df = pl
          .scanCSV(csvpath, {
            hasHeader: false,
            startRows: 1,
            endRows: 4
          })
          .collectSync();

        expect(df.shape).toStrictEqual({height: 4, width: 4});
      });
      it.todo("can read from a stream");
    });
  });
});
