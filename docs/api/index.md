---
hide:
  - navigation
---

# API reference

The API reference contains detailed descriptions of all public functions and objects.
It's the best place to look if you need information on a specific function.

## Python

The Python API reference is built using Sphinx.
It's available in [our docs](https://docs.pola.rs/py-polars/html/reference/index.html).


## polars.read_excel

Read Excel spreadsheet data into a DataFrame.

## Parameters

- `source`: Path to a file or a file-like object.
- `sheet_id`: Sheet number(s) to convert (set 0 to load all sheets as DataFrames) and return a `{sheetname:frame,}` dict.
- `sheet_name`: Sheet name(s) to convert; cannot be used in conjunction with `sheet_id`.
- `engine`: Library used to parse the spreadsheet file.
- `engine_options`: Additional options passed to the underlying engine's primary parsing constructor.
- `read_options`: Extra options passed to the function that reads the sheet data.
- `schema_overrides`: Support type specification or override of one or more columns.
- `raise_if_empty`: When there is no data in the sheet, `NoDataError` is raised. If set to False, an empty DataFrame is returned instead.

## Returns

- `DataFrame`: If reading a single sheet.
- `dict`: If reading multiple sheets, a `{sheetname: DataFrame, …}` dict is returned.

## Notes

- The default engine is `"xlsx2csv"`.
- You can pass additional options to `read_options` to influence the parsing pipeline.
- The `openpyxl` engine is slower but supports additional automatic type inference.
- The `pyxlsb` engine is used for Excel Binary Workbooks.
- The `calamine` engine can be used for reading all major types of Excel Workbook and is faster than other options.

## Examples

```python
import polars as pl

# Read the “data” worksheet from an Excel file into a DataFrame.
pl.read_excel(
    source="test.xlsx",
    sheet_name="data",
)

# Read table data from sheet 3 in an Excel workbook as a DataFrame while skipping empty lines.
pl.read_excel(
    source="test.xlsx",
    sheet_id=3,
    engine_options={"skip_empty_lines": True},
    read_options={"has_header": False, "new_columns": ["a", "b", "c"]},
)

# Use schema_overrides to specify column types.
pl.read_excel(
    source="test.xlsx",
    read_options={"infer_schema_length": 1000},
    schema_overrides={"dt": pl.Date},
)

# Use the openpyxl engine for better type detection.
pl.read_excel(
    source="test.xlsx",
    engine="openpyxl",
    schema_overrides={"dt": pl.Datetime, "value": pl.Int32},
)
```

## Rust

The Rust API reference is built using Cargo.
It's available on [docs.rs](https://docs.rs/polars/latest/polars/).
