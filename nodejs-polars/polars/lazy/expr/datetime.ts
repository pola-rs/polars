import {Expr} from "../expr";
import pli from "../../internals/polars_internal";

export interface ExprDateTime {
  /**
   * Extract day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of month starting from 1.
   * The return value ranges from 1 to 31. (The last day of month differs by months.)
   * @returns day as pl.UInt32
   */
  day(): Expr;
  /**
   * Extract hour from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the hour number from 0 to 23.
   * @returns Hour as UInt32
   */
  hour(): Expr;
  /**
   * Extract minutes from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the minute number from 0 to 59.
   * @returns minute as UInt32
   */
  minute(): Expr;
  /**
   * Extract month from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the month number starting from 1.
   * The return value ranges from 1 to 12.
   * @returns Month as UInt32
   */
  month(): Expr;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the number of nanoseconds since the whole non-leap second.
   * The range from 1,000,000,000 to 1,999,999,999 represents the leap second.
   * @returns Nanosecond as UInt32
   */
  nanosecond(): Expr;
  /**
   * Extract ordinal day from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the day of year starting from 1.
   * The return value ranges from 1 to 366. (The last day of year differs by years.)
   * @returns Day as UInt32
   */
  ordinalDay(): Expr;
  /**
   * Extract seconds from underlying DateTime representation.
   * Can be performed on Datetime.
   *
   * Returns the second number from 0 to 59.
   * @returns Second as UInt32
   */
  second(): Expr;
  /**
   * Format Date/datetime with a formatting rule: See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
   */
  strftime(fmt: string): Expr;
  /** Return timestamp in ms as Int64 type. */
  timestamp(): Expr;
  /**
   * Extract the week from the underlying Date representation.
   * Can be performed on Date and Datetime
   *
   * Returns the ISO week number starting from 1.
   * The return value ranges from 1 to 53. (The last week of year differs by years.)
   * @returns Week number as UInt32
   */
  week(): Expr;
  /**
   * Extract the week day from the underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the weekday number where monday = 0 and sunday = 6
   * @returns Week day as UInt32
   */
  weekday(): Expr;
  /**
   * Extract year from underlying Date representation.
   * Can be performed on Date and Datetime.
   *
   * Returns the year number in the calendar date.
   * @returns Year as Int32
   */
  year(): Expr;
}

export const ExprDateTimeFunctions = (_expr: any): ExprDateTime => {
  const wrap = (method, args?): Expr => {
    return Expr(pli.expr.date[method]({_expr, ...args }));
  };

  const wrapNullArgs = (method: string) => () => wrap(method);

  return {
    day: wrapNullArgs("day"),
    hour: wrapNullArgs("hour"),
    minute: wrapNullArgs("minute"),
    month: wrapNullArgs("month"),
    nanosecond: wrapNullArgs("nanosecond"),
    ordinalDay: wrapNullArgs("ordinalDay"),
    second: wrapNullArgs("second"),
    strftime: (fmt) => wrap("strftime", {fmt}),
    timestamp: wrapNullArgs("timestamp"),
    week: wrapNullArgs("week"),
    weekday: wrapNullArgs("weekday"),
    year: wrapNullArgs("year"),
  };
};
