import ctypes

__pdoc__ = {"dtype_to_ctype": False, "dtype_to_int": False}


class Int8:
    pass


class Int16:
    pass


class Int32:
    pass


class Int64:
    pass


class UInt8:
    pass


class UInt16:
    pass


class UInt32:
    pass


class UInt64:
    pass


class Float32:
    pass


class Float64:
    pass


class Boolean:
    pass


class Utf8:
    pass


class List:
    pass


class Date32:
    pass


class Date64:
    pass


class Time32Millisecond:
    pass


class Time32Second:
    pass


class Time64Nanosecond:
    pass


class Time64Microsecond:
    pass


class DurationNanosecond:
    pass


class DurationMicrosecond:
    pass


class DurationMillisecond:
    pass


class DurationSecond:
    pass


class IntervalDayTime:
    pass


class IntervalYearMonth:
    pass


class TimestampNanosecond:
    pass


class TimestampMicrosecond:
    pass


class TimestampMillisecond:
    pass


class TimestampSecond:
    pass


class Object:
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
    Time32Millisecond,
    Time32Second,
    Time64Nanosecond,
    Time64Microsecond,
    DurationNanosecond,
    DurationMicrosecond,
    DurationMillisecond,
    DurationSecond,
    IntervalDayTime,
    IntervalYearMonth,
    TimestampNanosecond,
    TimestampMicrosecond,
    TimestampMillisecond,
    TimestampSecond,
    Object,
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
    Time32Millisecond: "time32_millisecond",
    Time32Second: "time32_second",
    Time64Nanosecond: "time64_nanosecond",
    Time64Microsecond: "time64_microsecond",
    DurationNanosecond: "duration_nanosecond",
    DurationMicrosecond: "duration_microsecond",
    DurationMillisecond: "duration_millisecond",
    DurationSecond: "duration_second",
    IntervalDayTime: "interval_daytime",
    IntervalYearMonth: "interval_yearmonth",
    TimestampNanosecond: "timestamp_nanosecond",
    TimestampMicrosecond: "timestamp_microsecond",
    TimestampMillisecond: "timestamp_millisecond",
    TimestampSecond: "timestamp_second",
    Object: "object",
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


def dtype_to_ctype(dtype: "DataType") -> "ctype":
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
        return NotImplemented
    return ptr_type


def dtype_to_int(dtype: "DataType") -> int:
    i = 0
    for dt in dtypes:
        if dt == dtype:
            return i
        i += 1
