import ctypes

__pdoc__ = {
    "dtype_to_ctype": False,
    "dtype_to_int": False,
    "dtype_to_primitive": False,
    "pytype_to_polars_type": False,
}


class DataType:
    pass


class Int8(DataType):
    pass


class Int16(DataType):
    pass


class Int32(DataType):
    pass


class Int64(DataType):
    pass


class UInt8(DataType):
    pass


class UInt16(DataType):
    pass


class UInt32(DataType):
    pass


class UInt64(DataType):
    pass


class Float32(DataType):
    pass


class Float64(DataType):
    pass


class Boolean(DataType):
    pass


class Utf8(DataType):
    pass


class List(DataType):
    pass


class Date32(DataType):
    pass


class Date64(DataType):
    pass


class Time32Millisecond(DataType):
    pass


class Time32Second(DataType):
    pass


class Time64Nanosecond(DataType):
    pass


class Time64Microsecond(DataType):
    pass


class DurationNanosecond(DataType):
    pass


class DurationMicrosecond(DataType):
    pass


class DurationMillisecond(DataType):
    pass


class DurationSecond(DataType):
    pass


class TimestampNanosecond(DataType):
    pass


class TimestampMicrosecond(DataType):
    pass


class TimestampMillisecond(DataType):
    pass


class TimestampSecond(DataType):
    pass


class Object(DataType):
    pass


class Categorical(DataType):
    pass


# Don't change the order of these!
dtypes = [
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Boolean,
    Utf8,
    List,
    Date32,
    Date64,
    Time64Nanosecond,
    DurationNanosecond,
    DurationMillisecond,
    Object,
    Categorical,
]
DTYPE_TO_FFINAME = {
    Int8: "i8",
    Int16: "i16",
    Int32: "i32",
    Int64: "i64",
    UInt8: "u8",
    UInt16: "u16",
    UInt32: "u32",
    UInt64: "u64",
    Float32: "f32",
    Float64: "f64",
    Boolean: "bool",
    Utf8: "str",
    List: "list",
    Date32: "date32",
    Date64: "date64",
    Time64Nanosecond: "time64_nanosecond",
    DurationNanosecond: "duration_nanosecond",
    DurationMillisecond: "duration_millisecond",
    Object: "object",
    Categorical: "categorical",
}


def dtype_to_primitive(dtype: "DataType") -> "DataType":
    #  TODO: add more
    if dtype == Date32:
        return Int32
    if dtype == Date64:
        return Int64
    ffi_name = DTYPE_TO_FFINAME[dtype]
    if "duration" in ffi_name:
        return Int64
    return dtype


def dtype_to_ctype(dtype: "DataType") -> "ctype":  # noqa: F821
    if dtype == UInt8:
        ptr_type = ctypes.c_uint8
    elif dtype == UInt16:
        ptr_type = ctypes.c_uint16
    elif dtype == UInt32:
        ptr_type = ctypes.c_uint
    elif dtype == UInt64:
        ptr_type = ctypes.c_ulong
    elif dtype == Int8:
        ptr_type = ctypes.c_int8
    elif dtype == Int16:
        ptr_type = ctypes.c_int16
    elif dtype == Int32:
        ptr_type = ctypes.c_int
    elif dtype == Int64:
        ptr_type = ctypes.c_long
    elif dtype == Float32:
        ptr_type = ctypes.c_float
    elif dtype == Float64:
        ptr_type = ctypes.c_double
    elif dtype == Date32:
        ptr_type = ctypes.c_int
    elif dtype == Date64:
        ptr_type = ctypes.c_long
    else:
        raise NotImplementedError
    return ptr_type


def dtype_to_int(dtype: "DataType") -> int:
    i = 0
    for dt in dtypes:
        if dt == dtype:
            return i
        i += 1


def pytype_to_polars_type(data_type):
    if data_type == int:
        data_type = Int64
    elif data_type == str:
        data_type = Utf8
    elif data_type == float:
        data_type = Float64
    else:
        pass
    return data_type
