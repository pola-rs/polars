import os

import pyorc


data = {
    "float_nullable": [1.0, 2.0, None, 4.0, 5.0],
    "float_required": [1.0, 2.0, 3.0, 4.0, 5.0],
    "bool_nullable": [True, False, None, True, False],
    "bool_required": [True, False, True, True, False],
    "int_nullable": [5, -5, None, 5, 5],
    "int_required": [5, -5, 1, 5, 5],
    "double_nullable": [1.0, 2.0, None, 4.0, 5.0],
    "double_required": [1.0, 2.0, 3.0, 4.0, 5.0],
    "bigint_nullable": [5, -5, None, 5, 5],
    "bigint_required": [5, -5, 1, 5, 5],
    "utf8_required": ["a", "bb", "ccc", "dddd", "eeeee"],
    "utf8_nullable": ["a", "bb", None, "dddd", "eeeee"],
}

def infer_schema(data):
    schema = "struct<"
    for key, value in data.items():
        dt = type(value[0])
        if dt == float:
            dt = "float"
        elif dt == int:
            dt = "int"
        elif dt == bool:
            dt = "boolean"
        elif dt == str:
            dt = "string"
        else:
            raise NotImplementedError
        if key.startswith("double"):
            dt = "double"
        if key.startswith("bigint"):
            dt = "bigint"
        schema += key + ":" + dt + ","

    schema = schema[:-1] + ">"
    return schema


def _write(
    data,
    file_name: str,
    compression=pyorc.CompressionKind.NONE,
    dict_key_size_threshold=0.0,
):
    schema = infer_schema(data)

    output = open(file_name, "wb")
    writer = pyorc.Writer(
        output,
        schema,
        dict_key_size_threshold=dict_key_size_threshold,
        compression=compression,
    )
    num_rows = len(list(data.values())[0])
    for x in range(num_rows):
        row = tuple(values[x] for values in data.values())
        writer.write(row)
    writer.close()

os.makedirs("fixtures/pyorc", exist_ok=True)
_write(data, "fixtures/pyorc/test.orc")
