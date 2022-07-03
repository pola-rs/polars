import pl from "@polars";
describe("datelike", () => {
  test("asof join", () => {
    const fmt = "%F %T%.3f";
    const quotes = pl.DataFrame(
      {
        dates: pl.Series([
          "2016-05-25 13:30:00.023",
          "2016-05-25 13:30:00.023",
          "2016-05-25 13:30:00.030",
          "2016-05-25 13:30:00.041",
          "2016-05-25 13:30:00.048",
          "2016-05-25 13:30:00.049",
          "2016-05-25 13:30:00.072",
          "2016-05-25 13:30:00.075"
        ]).str.strptime(pl.Datetime, fmt),
        ticker: [
          "GOOG",
          "MSFT",
          "MSFT",
          "MSFT",
          "GOOG",
          "AAPL",
          "GOOG",
          "MSFT",
        ],
        bid: [
          720.5,
          51.95,
          51.97,
          51.99,
          720.50,
          97.99,
          720.50,
          52.01
        ],
      });
    const trades = pl.DataFrame({
      dates: pl.Series([
        "2016-05-25 13:30:00.023",
        "2016-05-25 13:30:00.038",
        "2016-05-25 13:30:00.048",
        "2016-05-25 13:30:00.048",
        "2016-05-25 13:30:00.048"
      ]).str.strptime(pl.Datetime, fmt),
      ticker: [
        "MSFT",
        "MSFT",
        "GOOG",
        "GOOG",
        "AAPL",
      ],
      bid: [
        51.95,
        51.95,
        720.77,
        720.92,
        98.0
      ],
    });
    let out: any = trades.joinAsof(quotes, {on: "dates"});
    expect(out.columns).toEqual(["dates", "ticker", "bid", "ticker_right", "bid_right"]);
    expect(out.getColumn("dates").cast(pl.Float64)
      .div(1000)
      .toArray()).toEqual([
      1464183000023,
      1464183000038,
      1464183000048,
      1464183000048,
      1464183000048,
    ]);
    out = trades.joinAsof(quotes, {on:"dates", strategy:"forward"}).getColumn("bid_right")
      .toArray();
    expect(out).toEqual([720.5, 51.99, 720.5, 720.5, 720.5]);

    out = trades.joinAsof(quotes, {on:"dates", by:"ticker"});
    expect(out.getColumn("bid_right").toArray()).toEqual([51.95, 51.97, 720.5, 720.5, null]);
    out = quotes.joinAsof(trades, {on:"dates", by:"ticker"});
    expect(out.getColumn("bid_right").toArray()).toEqual([
      null,
      51.95,
      51.95,
      51.95,
      720.92,
      98.0,
      720.92,
      51.95,
    ]);
    out = quotes.joinAsof(trades, {on:"dates", strategy:"backward", tolerance:"5ms"})[
      "bid_right"
    ].toArray();
    expect(out).toEqual([51.95, 51.95, null, 51.95, 98.0, 98.0, null, null]);
    out = quotes.joinAsof(trades, {on:"dates", strategy:"forward", tolerance:"5ms"})[
      "bid_right"
    ].toArray();
    expect(out).toEqual([51.95, 51.95, null, null, 720.77, null, null, null]);
  });
  test("asofjoin tolerance grouper", () => {

    const df1 = pl.DataFrame({"date": [new Date(2020, 1, 5), new Date(2020, 1, 10)], "by": [1, 1]});
    const df2 = pl.DataFrame(
      {
        "date": [new Date(2020, 1, 5), new Date(2020, 1, 6)],
        "by": [1, 1],
        "values": [100, 200],
      }
    );

    const out = df1.joinAsof(df2, {by: "by", on:"date", tolerance:"3d"});

    const expected = pl.DataFrame(
      {
        "date": [new Date(2020, 1, 5), new Date(2020, 1, 10)],
        "by": [1, 1],
        "values": [100, null],
      }
    );

    expect(out).toFrameEqual(expected);

  });
});
