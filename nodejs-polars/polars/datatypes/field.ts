import { DataType } from "./datatype";

export interface Field {
  name: string,
  dtype: DataType
}

export class Field {
  constructor(public name: string, public dtype: DataType) { }
  toString() {
    return `Field("${this.name}": ${this.dtype})`;
  }
  toJSON() {
    return {
      name: this.name,
      dtype: this.dtype.toString(),
    };
  }
  [Symbol.for("nodejs.util.inspect.custom")]() {
    return this.toJSON();
  }
}

export namespace Field {
  export function from(name: string, dtype: DataType): Field;
  export function from([string, DataType]): Field;
  export function from(obj: {name: string; dtype: DataType}): Field;
  export function from(nameOrObj, dtype?: DataType): Field {
    if (typeof nameOrObj === "string") {
      return new Field(nameOrObj, dtype!);
    } else if (Array.isArray(nameOrObj)) {
      return new Field(nameOrObj[0], nameOrObj[1]);
    } else {
      return new Field(nameOrObj.name, nameOrObj.dtype);

    }
  }
}
