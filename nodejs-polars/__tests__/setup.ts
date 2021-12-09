expect.extend({
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
  toFrameEqual(actual, expected) {
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


const x = `
  [foo:Float64]: [1, 2]
`;