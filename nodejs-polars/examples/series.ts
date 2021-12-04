import pl from "../polars";

const df = pl.DataFrame([
  pl.Series("date", [new Date(), new Date()], pl.Date),
  pl.Series("datetime", [new Date(), new Date()], pl.Datetime),
]);
console.log(df);