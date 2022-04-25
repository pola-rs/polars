import pli from "./internals/polars_internal";
import {arrayToJsSeries} from "./internals/construction";
import {DataType, DtypeToPrimitive, DTYPE_TO_FFINAME, Optional} from "./datatypes";
// import {DataFrame, dfWrapper} from "../dataframe";
// import {StringFunctions} from "./string";
// import {ListFunctions} from "./list";
// import {DateTimeFunctions} from "./datetime";
import {InvalidOperationError, todo} from "./error";
// import {RankMethod} from "../utils";
// import {col} from "../lazy/functions";
import {isExternal, isTypedArray} from "util/types";
import {Arithmetic, Comparison, Cumulative, Rolling, Round} from "./shared_traits";

const inspect = Symbol.for("nodejs.util.inspect.custom");


// replace these!!
type DataFrame = any;
type RankMethod = any;
const col = (...args: any[]): any =>  {};
const DataFrame: DataFrame = (...args: any[]): any => {};


export interface Series {
  name: string
  dtype: DataType
  // str: StringFunctions
  // lst: ListFunctions<T>
  // date: DateTimeFunctions
  [inspect](): string;
  // [Symbol.iterator](): IterableIterator<any>;
  /**
   * __Append a Series to this one.__
   * ___
   * @param {Series} other - Series to append.
   * @example
   * > const s = pl.Series("a", [1, 2, 3])
   * > const s2 = pl.Series("b", [4, 5, 6])
   * > s.append(s2)
   * shape: (6,)
   * Series: 'a' [i64]
   * [
   *         1
   *         2
   *         3
   *         4
   *         5
   *         6
   * ]
   */
  append(other: Series): void
}


export interface SeriesConstructor {
  // (name: string, values: any[], dtype?: DataType, strict?: boolean): Series
  new(name: string, values: any[]): Series
  // isSeries(arg: any): arg is Series;
  // readonly prototype: any[];
}

export interface SeriesConstructor {
  /**
   * Creates a series from an iterable object.
   * @param iterable An iterable object to convert to an array.
   */
  from<T>(iterable: Iterable<T> | ArrayLike<T>): Series;
  /**
   * Returns a new series from a set of elements.
   * @param items A set of elements to include in the new array object.
   */
  of<T>(...items: T[]): Series;
}

class _Series implements Series  {
  #s: pli.Series;

  static from(iterable) {
    return new Series("", iterable);
  }
  static of(...items) {
    return new Series("", items);
  }
  private unwrap<T>(method: keyof pli.Series, ...args: any[]): T {
    return (this.#s[method] as any)(...args);
  }
  private wrap(method: keyof pli.Series, ...args: any[]) {
    const _s = this.unwrap(method, ...args);

    return new (Series as any)(_s);
  }
  private dtypeAccessor<T>(method, ...args): T {
    const dtype = this.dtype;
    const dt = DTYPE_TO_FFINAME[dtype];
    const internalMethod = `${method}_${dt}`;

    return this.unwrap(internalMethod as any, ...args);
  }
  [inspect]() {
    return this.unwrap<string>("as_str");
  }
  // *[Symbol.iterator]() {
  //   let start = 0;
  //   let len = this.unwrap<number>("len");
  //   while (start < len) {
  //     const v = this.get(start);
  //     start++;
  //     yield v;
  //   }
  // }
  toString() {
    return this.unwrap<string>("as_str");
  }
  get [Symbol.toStringTag]() {
    return "Series";
  }
  get dtype(): DataType {
    return this.unwrap("dtype");
  }
  get name(): string {
    return this.unwrap("name");
  }
  get length(): number {
    return this.unwrap("len");
  }
  abs() {
    this.unwrap("abs");
  }
  add(other) {
    return this.dtypeAccessor<Series>("add", other);
  }
  alias(name) {
    // const s = this.clone();
    // unwrap("rename", {name}, s._series);

    // return s;
  }
  append(other: Series) {
    this.unwrap("append", (other as _Series).#s);
  }
  argMax() {
    this.unwrap("argMax");
  }
  argMin() {
    this.unwrap("argMin");
  }
  argSort(reverse = false) {
    return this.wrap("argSort");
  }
  argTrue() {
    return this.wrap("argTrue");
  }
  argUnique(){
    return this.wrap("argUnique");
  }
  as(name) {
    return this.alias(name);
  }
  bitand(other) {
    return this.wrap("bitand", other.#s);
  }
  bitor(other) {
    return this.wrap("bitor", other.#s);
  }
  bitxor(other) {
    return this.wrap("bitxor", other.#s);
  }
  cast(dtype, strict = false) {
    return this.wrap("cast", dtype, strict);
  }
  chunkLengths() {
    return this.wrap("chunkLengths");
  }
  clone() {
    return this.wrap("clone");
  }
  concat(other) {
    const s = this.clone();
    s.append(other);

    return s;
  }
  // cumCount(reverse?) {
  //   return this
  //     .toFrame()
  //     .select(col(this.name).cumCount(reverse))
  //     .getColumn(this.name);
  // },
  cumMax(reverse: any = false) {
    return typeof reverse === "boolean" ?
      this.wrap("cumMax", reverse) :
      this.wrap("cumMax", reverse.reverse);
  }
  cumMin(reverse: any = false) {
    return typeof reverse === "boolean" ?
      this.wrap("cumMin", reverse) :
      this.wrap("cumMin", reverse.reverse);
  }
  cumProd(reverse: any = false) {
    return typeof reverse === "boolean" ?
      this.wrap("cumProd", reverse) :
      this.wrap("cumProd", reverse.reverse);
  }
  cumSum(reverse: any = false) {
    return typeof reverse === "boolean" ?
      this.wrap("cumSum", reverse) :
      this.wrap("cumSum", reverse.reverse);
  }
  constructor(arg0: any, arg1?: any, dtype?: any, strict?: any) {
    if (typeof arg0 === "string") {
      const _s = arrayToJsSeries(arg0, arg1, dtype, strict);
      this.#s = _s;
    } else if (arg0?.ptr) {
      this.#s = arg0;
    } else {
      const _s = arrayToJsSeries("", arg0);
      this.#s = _s;

    }
  }

}


export const Series: SeriesConstructor = _Series;

let s = new Series("foo", [1, 2, 3]);
const s2 = new Series("foo", [4, 5, 6]);
s = (s as any).concat(s2).cumSum();
console.log(s);

// export interface SeriesConstructor {
//   <V extends ArrayLike<any>>(values: V): ValueOrNever<V>
//   <V extends ArrayLike<any>>(name: string, values: V): ValueOrNever<V>
//   <T extends DataType, U extends ArrayLikeDataType<T>>(name: string, values: U, dtype: T): Series<DtypeToPrimitive<T>>
//   <T extends DataType, U extends boolean, V extends ArrayLikeOrDataType<T, U>>(name: string, values: V, dtype?: T, strict?: U): Series<DataTypeOrValue<T, U>>

//   /**
//    * Creates an array from an array-like object.
//    * @param arrayLike — An array-like object to convert to an array.
//    */
//   from<T>(arrayLike: ArrayLike<T>): Series<T>
//   /**
//    * Returns a new Series from a set of elements.
//    * @param items — A set of elements to include in the new series object.
//    */
//   of<T>(...items: T[]): Series<T>
//   isSeries(arg: any): arg is Series<any>;

// }


// const isSeries = <T>(anyVal: any): anyVal is Series<T> => isExternal(anyVal?._series);

// const from = <T>(values: ArrayLike<T>): Series<T> => {
//   if(isTypedArray(values)) {
//     throw new Error("todo!()");
//     // return seriesWrapper(pli.series.new_from_typed_array({name: "", values}));
//   }

//   return SeriesConstructor("", values);
// };

// const of = <T>(...values: T[]): Series<T> => {
//   return from(values);
// };

// export const Series: SeriesConstructor = Object.assign(SeriesConstructor, {
//   isSeries,
//   from,
//   of
// });
