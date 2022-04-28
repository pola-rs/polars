
import pl from "@polars";
import {DataType} from "@polars/datatypes";

describe("struct", () => {
  describe("series", () => {
    test("series <--> array round trip", () => {
      const data = [
        {utf8: "a", f64: 1, },
        {utf8: "b", f64: 2, }
      ];
      const name = "struct";
      const s = pl.Series(name, data);
      expect(s.name).toEqual(name);
      expect(s.toArray()).toEqual(data);
    });
  });
});
