import pl from "@polars";
import path from "path";
import {Stream} from "stream";
import fs from "fs";
// eslint-disable-next-line no-undef
const csvpath = path.resolve(__dirname, "../../examples/datasets/foods1.csv");
// eslint-disable-next-line no-undef
const parquetpath = path.resolve(__dirname, "./examples/foods.parquet");
// eslint-disable-next-line no-undef
const avropath = path.resolve(__dirname, "./examples/foods.avro");
// eslint-disable-next-line no-undef
const ipcpath = path.resolve(__dirname, "./examples/foods.ipc");
// eslint-disable-next-line no-undef
const jsonpath = path.resolve(__dirname, "./examples/foods.json");
describe("read:csv", () => {
  it("can read from a csv file", () => {
    const df = pl.readCSV(csvpath);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });

  it("can read from a relative file", () => {
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
  it("can read from a csv buffer", () => {
    const csvBuffer = Buffer.from("foo,bar,baz\n1,2,3\n4,5,6\n", "utf-8");
    const df = pl.readCSV(csvBuffer);
    expect(df.toCSV()).toEqual(csvBuffer.toString());
  });
  it("can read from a csv buffer with options", () => {
    const csvBuffer = Buffer.from("foo,bar,baz\n1,2,3\n4,5,6\n", "utf-8");
    const df = pl.readCSV(csvBuffer, {hasHeader: true, batchSize: 10});
    expect(df.toCSV()).toEqual(csvBuffer.toString());
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
  test("csv with rowcount", () => {
    const df = pl.readCSV(csvpath, {rowCount: {name: "rc", offset: 11}});
    const expectedMaxRowCount = df.height + 10;

    const maxRowCount = df.getColumn("rc").max();
    expect(expectedMaxRowCount).toStrictEqual(maxRowCount);

  });
  it.todo("can read from a stream");
});

describe("read:json", () => {
  it("can read from a json file", () => {
    const df = pl.readJSON(jsonpath);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  it("can specify read options", () => {
    const df = pl.readJSON(jsonpath, {batchSize: 10, inferSchemaLength: 100});
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  it("can read from a json buffer", () => {
    const json = [
      JSON.stringify({foo: 1, bar: "1"}),
      JSON.stringify({foo: 2, bar: "1"}),
      ""
    ].join("\n");
    const df = pl.readJSON(Buffer.from(json));
    expect(df.toJSON({multiline:true})).toEqual(json.toString());
  });
});

describe("scan", () => {
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
  it("can lazy load (scan) from a ipc file", () => {
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

  it("can lazy load (scan) from a parquet file with options", () => {
    pl
      .readCSV(csvpath, {
        hasHeader: false,
        startRows: 1,
        endRows: 4
      }).writeParquet(parquetpath);

    const df = pl.readParquet(parquetpath);

    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });
});

describe("parquet", () => {
  beforeEach(() => {
    pl.readCSV(csvpath).writeParquet(parquetpath);
  });
  afterEach(() => {
    fs.rmSync(parquetpath);
  });

  test("read", () => {
    const df = pl.readParquet(parquetpath);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  test("read:buffer", () => {
    const buff = fs.readFileSync(parquetpath);
    const df = pl.readParquet(buff);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });

  test("read:compressed", () => {
    const csvDF = pl.readCSV(csvpath);
    csvDF.writeParquet(parquetpath, {compression: "lz4"});
    const df = pl.readParquet(parquetpath);
    expect(df).toFrameEqual(csvDF);
  });

  test("read:options", () => {
    const df = pl.readParquet(parquetpath, {numRows: 4});
    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });

  test("scan", () => {
    const df = pl.scanParquet(parquetpath).collectSync();
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });

  test("scan:options", () => {
    const df = pl.scanParquet(parquetpath, {numRows: 4}).collectSync();
    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });
});
describe("ipc", () => {
  beforeEach(() => {
    pl.readCSV(csvpath).writeIPC(ipcpath);
  });
  afterEach(() => {
    fs.rmSync(ipcpath);
  });

  test("read", () => {
    const df = pl.readIPC(ipcpath);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  test("read/write:buffer", () => {

    const buff =  pl.readCSV(csvpath).writeIPC();
    const df = pl.readIPC(buff);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  test("read:compressed", () => {
    const csvDF = pl.readCSV(csvpath);
    csvDF.writeIPC(ipcpath, {compression: "lz4"});
    const ipcDF = pl.readIPC(ipcpath);
    expect(ipcDF).toFrameEqual(csvDF);
  });

  test.skip("read:options", () => {
    const df = pl.readIPC(ipcpath, {numRows: 4});
    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });

  test("scan", () => {
    const df = pl.scanIPC(ipcpath).collectSync();
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });

  test.skip("scan:options", () => {
    const df = pl.scanIPC(ipcpath, {numRows: 4}).collectSync();
    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });

  test("writeIPC", () => {
    const csvDF = pl.readCSV(csvpath);
    csvDF.writeIPC(ipcpath);
    const ipcDF = pl.readIPC(ipcpath);
    expect(ipcDF).toFrameEqual(csvDF);
  });

});


describe("avro", () => {
  beforeEach(() => {
    pl.readCSV(csvpath).writeAvro(avropath);
  });
  afterEach(() => {
    fs.rmSync(avropath);
  });

  test("round trip", () => {
    const expected = pl.DataFrame({
      foo: [1, 2, 3],
      bar: ["a", "b", "c"]
    });
    const buf = expected.writeAvro();
    const actual = pl.readAvro(buf);
    expect(actual).toFrameEqual(expected);
  });
  test("read", () => {
    const df = pl.readAvro(avropath);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });
  test("read:buffer", () => {
    const buff = fs.readFileSync(avropath);
    const df = pl.readAvro(buff);
    expect(df.shape).toStrictEqual({height: 27, width: 4});
  });

  test("read:compressed", () => {
    const csvDF = pl.readCSV(csvpath);
    csvDF.writeAvro(avropath, {compression: "snappy"});
    const df = pl.readAvro(avropath);
    expect(df).toFrameEqual(csvDF);
  });

  test.skip("read:options", () => {
    const df = pl.readAvro(avropath, {numRows: 4});
    expect(df.shape).toStrictEqual({height: 4, width: 4});
  });
});

describe("stream", () => {
  test("readCSV", async () => {
    const readStream = new Stream.Readable({read(){}});
    readStream.push(`a,b\n`);
    readStream.push(`1,2\n`);
    readStream.push(`2,2\n`);
    readStream.push(`3,2\n`);
    readStream.push(`4,2\n`);
    readStream.push(null);
    const expected = pl.DataFrame({
      a: pl.Series("a", [1, 2, 3, 4], pl.Int64),
      b: pl.Series("b", [2, 2, 2, 2], pl.Int64)
    });
    const df = await pl.readCSVStream(readStream, {batchSize: 2});
    expect(df).toFrameEqual(expected);
  });

  test("readCSV:schema mismatch", async () => {
    const readStream = new Stream.Readable({read(){}});
    readStream.push(`a,b,c\n`);
    readStream.push(`1,2\n`);
    readStream.push(`2,2\n`);
    readStream.push(`3,2\n`);
    readStream.push(`11,1,2,3,4,5,1\n`);
    readStream.push(`null`);
    readStream.push(null);

    const promise =  pl.readCSVStream(readStream, {inferSchemaLength: 2, ignoreErrors: false});
    await expect(promise).rejects.toBeDefined();
  });

  test("readJSON", async () => {
    const readStream = new Stream.Readable({read(){}});
    readStream.push(`${JSON.stringify({a: 1, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 2, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 3, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 4, b: 2})} \n`);
    readStream.push(null);

    const expected = pl.DataFrame({
      a: pl.Series("a", [1, 2, 3, 4], pl.Int64),
      b: pl.Series("b", [2, 2, 2, 2], pl.Int64)
    });
    const df = await pl.readJSONStream(readStream);
    expect(df).toFrameEqual(expected);
  });

  test("readJSON:error", async () => {
    const readStream = new Stream.Readable({read(){}});
    readStream.push(`${JSON.stringify({a: 1, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 2, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 3, b: 2})} \n`);
    readStream.push(`not parseable json `);
    readStream.push(null);

    await expect(pl.readJSONStream(readStream)).rejects.toBeDefined();

  });
  test("readJSON:schema mismatch", async () => {
    const readStream = new Stream.Readable({read(){}});
    readStream.push(`${JSON.stringify({a: 1, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 2, b: 2})} \n`);
    readStream.push(`${JSON.stringify({a: 3, b: 2})} \n`);
    readStream.push(`${JSON.stringify({b: "3", d: 2})} \n`);
    readStream.push(null);

    await expect(pl.readJSONStream(readStream, {batchSize: 2})).rejects.toBeDefined();

  });
});
