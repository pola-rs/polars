import { Dtype, DTYPE_TO_FFINAME } from '../datatypes';
import polars_internal from '../polars_internal';

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
  dtype: Dtype,
  obj?: any,
  defaultFunc?: CallableFunction,
): CallableFunction {
  const ffiName = DTYPE_TO_FFINAME[dtype];
  const fName = name.replace('<>', ffiName);

  if (obj) {
    return (<any>obj)[fName] ?? defaultFunc;
  } else {
    return polars_internal.series[fName] ?? defaultFunc;
  }
}
