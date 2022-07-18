import pl from "@polars/index";

expect.extend({
  toSeriesStrictEqual(actual, expected){
    const seriesEq = actual.seriesEqual(expected);
    const typesEq = actual.dtype.equals(expected.dtype);
    if(seriesEq && typesEq) {
      return {
        message: () => "series matches",
        pass: true
      };

    } else {
      return {
        message: () => `
Expected: 
>>${expected} 
Received:
>>${actual}`,
        pass: false
      };
    }
  },
  toSeriesEqual(actual, expected) {
    const pass = actual.seriesEqual(expected);
    if(pass) {
      return {
        message: () => "series matches",
        pass: true
      };

    } else {
      return {
        message: () => `
Expected: 
>>${expected} 
Received:
>>${actual}`,
        pass: false
      };
    }
  },
  toFrameEqual(actual, expected, nullEqual?) {
    const pass = actual.frameEqual(expected, nullEqual);
    if(pass) {
      return {
        message: () => "dataframes match",
        pass: true
      };
    } else {

      return {
        message: () => `
Expected: 
>>${expected} 
Received:
>>${actual}`,
        pass: false
      };
    }
  },
  toFrameStrictEqual(actual, expected) {
    const frameEq = actual.frameEqual(expected);
    const dtypesEq = this.equals(actual.dtypes, expected.dtypes);
    if(frameEq && dtypesEq) {
      return {
        message: () => "dataframes match",
        pass: true
      };
    } else {
      return {
        message: () => `
Expected: 
>>${expected} 
Received:
>>${actual}`,
        pass: false
      };
    }
  },
  toFrameEqualIgnoringOrder(actual, expected) {
    actual = actual.sort(actual.columns.sort());
    expected = expected.sort(expected.columns.sort());
    const pass = actual.frameEqual(expected);
    if(pass) {
      return {
        message: () => "dataframes match",
        pass: true
      };
    } else {

      return {
        message: () => `
Expected: 
>>${expected} 
Received:
>>${actual}`,
        pass: false
      };
    }
  }
});

export const df = () => {
  const df = pl.DataFrame(
    {
      "bools": [false, true, false],
      "bools_nulls": [null, true, false],
      "int": [1, 2, 3],
      "int_nulls": [1, null, 3],
      "bigint": [1n, 2n, 3n],
      "bigint_nulls": [1n, null, 3n],
      "floats": [1.0, 2.0, 3.0],
      "floats_nulls": [1.0, null, 3.0],
      "strings": ["foo", "bar", "ham"],
      "strings_nulls": ["foo", null, "ham"],
      "date": [new Date(), new Date(), new Date()],
      "datetime": [13241324, 12341256, 12341234],
    });

  return df.withColumns(

    pl.col("date").cast(pl.Date),
    pl.col("datetime").cast(pl.Datetime("ms")),
    pl.col("strings").cast(pl.Categorical)
      .alias("cat")
  );
};
