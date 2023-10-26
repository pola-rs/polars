# Excel

Polars can read and write to Excel files from Python. From a performance perspective we recommend using other formats if possible such as Parquet or CSV files.

## Read

Polars does not have a native Excel reader. Instead it uses external libraries to parse Excel files into objects that Polars can parse. To read Excel files we must install either the (default) xls2csv library or one of the alternatives as an additional dependency
=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install xls2csv openpyxl pyxlsb
    ```

The default Excel reader is xls2csv. It is a Python library
which parses the Excel file into a CSV file which Polars then reads with the native CSV reader. We read an Excel file with `read_excel`:

{{code_block('user-guide/io/excel','read',['read_excel'])}}

We can specify the sheet name to read with the `sheet_name` argument. If we do not specify a sheet name, the first sheet will be read.

{{code_block('user-guide/io/excel','read_sheet_name',['read_excel'])}}

## Write

We need the xlswriter library installed as an additional dependency to write to Excel files.
=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install xlsxwriter
    ```

Writing to Excel files is not currently available in Rust Polars though it is possible to [use this crate](https://docs.rs/crate/xlsxwriter/latest) to write to Excel files from Rust.

Writing a `DataFrame` to an Excel file is done with the `write_excel` function:

{{code_block('user-guide/io/excel','write',['write_excel'])}}

The name of the worksheeet can be specified with the `worksheet` argument.

{{code_block('user-guide/io/excel','write_sheet_name',['write_excel'])}}

Polars can create rich Excel files with multiple sheets and formatting. For more details see the API docs for `write_excel`.
