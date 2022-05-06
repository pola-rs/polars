import {Series, _Series} from "./series";
import {DateFunctions} from "../shared_traits";

export type SeriesDateFunctions = DateFunctions<Series>;

export const SeriesDateFunctions = (_s): SeriesDateFunctions => {
  const wrap = (method, ...args: any[]): Series => {

    return _Series(_s[method](...args)) as any;
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
    strftime: (fmt) => wrap("strftime", fmt),
    timestamp: wrapNullArgs("timestamp"),
    week: wrapNullArgs("week"),
    weekday: wrapNullArgs("weekday"),
    year: wrapNullArgs("year"),
  };
};
