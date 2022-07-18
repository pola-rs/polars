import {Field} from "./field";

export abstract class DataType {
  get variant() {
    return this.constructor.name.slice(1);
  }
  protected identity = "DataType";
  protected get inner(): null | any[] {
    return null;
  }
  equals(other: DataType): boolean {
    return (
      this.variant === other.variant &&
      this.inner === null &&
      other.inner === null
    );
  }

  /** Null type */
  public static get Null(): DataType {
    return new _Null();
  }
  /** `true` and `false`. */
  public static get Bool(): DataType {
    return new _Bool();
  }
  /** An `i8` */
  public static get Int8(): DataType {
    return new _Int8();
  }
  /** An `i16` */
  public static get Int16(): DataType {
    return new _Int16();
  }
  /** An `i32` */
  public static get Int32(): DataType {
    return new _Int32();
  }
  /** An `i64` */
  public static get Int64(): DataType {
    return new _Int64();
  }
  /** An `u8` */
  public static get UInt8(): DataType {
    return new _UInt8();
  }
  /** An `u16` */
  public static get UInt16(): DataType {
    return new _UInt16();
  }
  /** An `u32` */
  public static get UInt32(): DataType {
    return new _UInt32();
  }
  /** An `u64` */
  public static get UInt64(): DataType {
    return new _UInt64();
  }

  /** A `f32` */
  public static get Float32(): DataType {
    return new _Float32();
  }
  /** A `f64` */
  public static get Float64(): DataType {
    return new _Float64();
  }
  public static get Date(): DataType {
    return new _Date();
  }

  /** Time of day type */
  public static get Time(): DataType {
    return new _Time();
  }
  /** Type for wrapping arbitrary JS objects */
  public static get Object(): DataType {
    return new _Object();
  }
  /** A categorical encoding of a set of strings  */
  public static get Categorical(): DataType {
    return new _Categorical();
  }

  /**
   * Calendar date and time type
   * @param timeUnit any of 'ms' | 'ns' | 'us'
   * @param timeZone timezone string as defined by Intl.DateTimeFormat `America/New_York` for example.
   *
   */
  public static Datetime(timeUnit: TimeUnit, timeZone?): DataType;
  public static Datetime(timeUnit: "ms" | "ns" | "us", timeZone?): DataType;
  public static Datetime(timeUnit, timeZone: string | null | undefined = null): DataType {
    return new _Datetime(timeUnit, timeZone as any);
  }
  /**
   * Nested list/array type
   *
   * @param inner The `DataType` of values within the list
   *
   */
  public static List(inner: DataType): DataType {
    return new _List(inner);
  }
  public static Struct(fields: Field[]): DataType;
  public static Struct(fields: {[key: string]: DataType}): DataType;
  public static Struct(
    fields: Field[] | {[key: string]: DataType}
  ): DataType {
    return new _Struct(fields);
  }
  /** A variable-length UTF-8 encoded string whose offsets are represented as `i64`. */
  public static get Utf8(): DataType {
    return new _Utf8();
  }

  toString() {
    return `${this.identity}.${this.variant}`;
  }
  toJSON() {
    const inner = (this as any).inner;
    if (inner) {
      return {
        [this.identity]: {
          variant: this.variant,
          inner,
        },
      };
    } else {
      return {
        [this.identity]: {
          variant: this.variant,
        },
      };
    }
  }
  [Symbol.for("nodejs.util.inspect.custom")]() {
    return this.toJSON();
  }
  static from(obj): DataType {

    return null as any;
  }
}

class _Null extends DataType { }
class _Bool extends DataType { }
class _Int8 extends DataType { }
class _Int16 extends DataType { }
class _Int32 extends DataType { }
class _Int64 extends DataType { }
class _UInt8 extends DataType { }
class _UInt16 extends DataType { }
class _UInt32 extends DataType { }
class _UInt64 extends DataType { }
class _Float32 extends DataType { }
class _Float64 extends DataType { }
class _Date extends DataType { }
class _Time extends DataType { }
class _Object extends DataType { }
class _Utf8 extends DataType { }

class _Categorical extends DataType { }
class _Datetime extends DataType {
  constructor(
    private timeUnit: TimeUnit,
    private timeZone?: string
  ) {
    super();
  }
  override get inner() {

    return [this.timeUnit, this.timeZone];
  }

  override equals(other: DataType): boolean {
    if (other.variant === this.variant) {
      return (
        this.timeUnit === (other as _Datetime).timeUnit &&
        this.timeZone === (other as _Datetime).timeZone
      );
    } else {
      return false;
    }
  }
}

class _List extends DataType {
  constructor(protected __inner: DataType) {
    super();
  }
  override get inner() {
    return [this.__inner];
  }
  override equals(other: DataType): boolean {
    if (other.variant === this.variant) {
      return this.inner[0].equals((other as _List).inner[0]);
    } else {
      return false;
    }
  }
}

class _Struct extends DataType {
  private fields: Field[];

  constructor(
    inner:
      | {
        [name: string]: DataType;
      }
      | Field[]
  ) {
    super();
    if (Array.isArray(inner)) {
      this.fields = inner;
    } else {
      this.fields = Object.entries(inner).map(Field.from);
    }
  }
  override get inner() {
    return this.fields;
  }
  override equals(other: DataType): boolean {
    if (other.variant === this.variant) {
      return this.inner
        .map((fld, idx) => {
          const otherfld = (other as _Struct).fields[idx];

          return otherfld.name === fld.name && otherfld.dtype.equals(fld.dtype);
        })
        .every((value) => value);
    } else {
      return false;
    }
  }
  override toJSON() {
    return {
      variant: this.variant,
      fields: this.fields.map(fld => fld.toJSON())
    } as any;
  }
}

export enum TimeUnit {
  Nanoseconds = "ns",
  Microseconds = "us",
  Milliseconds = "ms",
}

export namespace TimeUnit {
  export function from(s: "ms" | "ns" | "us"): TimeUnit {
    return TimeUnit[s];
  }
}
import util from "util";
export namespace DataType {
  export type Null = _Null;

  export type Bool = _Bool;
  export type Int8 = _Int8;
  export type Int16 = _Int16;
  export type Int32 = _Int32;
  export type Int64 = _Int64;
  export type UInt8 = _UInt8;
  export type UInt16 = _UInt16;
  export type UInt32 = _UInt32;
  export type UInt64 = _UInt64;
  export type Float32 = _Float32;
  export type Float64 = _Float64;
  export type Date = _Date;
  export type Datetime = _Datetime;
  export type Utf8 = _Utf8;
  export type Categorical = _Categorical;
  export type List = _List;
  export type Struct = _Struct;
  /**
   * deserializes a datatype from the serde output of rust polars `DataType`
   * @param dtype dtype object
   */
  export function deserialize(dtype: any): DataType {
    if (typeof dtype === "string") {
      return DataType[dtype];
    }

    let {variant, inner} = dtype;
    if(variant === "Struct") {
      inner = [inner[0].map(fld => Field.from(fld.name, deserialize(fld.dtype)))];
    }
    if(variant === "List") {
      inner = [deserialize(inner[0])];
    }

    return DataType[variant](...inner);

  }
}
