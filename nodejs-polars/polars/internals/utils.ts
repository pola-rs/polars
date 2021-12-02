import { DataType, DTYPE_TO_FFINAME } from "../datatypes";
import pli from "./polars_internal";

export const todo = () => new Error("not yet implemented");

/**
 *
 * @param name function or method name where dtype is replaced by <>
 *
 * *for example: `"call_foo_<>"`
 * @param dtype polars dtype.
 * @param defaultFunc default function to use if not found.
 * @returns internal function
 */
export function getInternalFunc(
  name: string,
  dtype: DataType,
  obj?: any,
  defaultFunc?: CallableFunction,
): CallableFunction {
  const ffiName = DTYPE_TO_FFINAME[dtype];
  const fName = name.replace("<>", ffiName);

  if (obj) {
    return (<any>obj)[fName] ?? defaultFunc;
  } else {
    return pli.series[fName] ?? defaultFunc;
  }
}
