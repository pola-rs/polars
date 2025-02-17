# Excel

Polars can read and write to Excel files from Python. From a performance perspective, we recommend
using other formats if possible, such as Parquet or CSV files.

## Read

Polars does not have a native Excel reader. Instead, it uses an external library called an "engine"
to parse Excel files into a form that Polars can parse. The available engines are:

- fastexcel: This engine is based on the Rust [calamine](https://github.com/tafia/calamine) crate
  and is (by far) the fastest reader.
- xlsx2csv: This reader parses the .xlsx file to an in-memory CSV that Polars then reads with its
  own CSV reader.
- openpyxl: Typically slower than xls2csv, but can provide more flexibility for files that are
  difficult to parse.

We recommend working with the default fastexcel engine. The xlsx2csv and openpyxl engines are slower
but may have more features for parsing tricky data. These engines may be helpful if the fastexcel
reader does not work for a specific Excel file.

To use one of these engines, the appropriate Python package must be installed as an additional
dependency.

=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install fastexcel xlsx2csv openpyxl 
    ```

The default engine for reading .xslx files is fastexcel. This engine uses the Rust calamine crate to
read .xslx files into an Apache Arrow in-memory representation that Polars can read without needing
to copy the data.

{{code_block('user-guide/io/excel','read',['read_excel'])}}

We can specify the sheet name to read with the `sheet_name` argument. If we do not specify a sheet
name, the first sheet will be read.

{{code_block('user-guide/io/excel','read_sheet_name',['read_excel'])}}

## Write

We need the xlswriter library installed as an additional dependency to write to Excel files.

=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install xlsxwriter
    ```

Writing to Excel files is not currently available in Rust Polars, though it is possible to
[use this crate](https://docs.rs/crate/xlsxwriter/latest) to write to Excel files from Rust.

Writing a `DataFrame` to an Excel file is done with the `write_excel` method:

{{code_block('user-guide/io/excel','write',['write_excel'])}}

The name of the worksheet can be specified with the `worksheet` argument.

{{code_block('user-guide/io/excel','write_sheet_name',['write_excel'])}}

Polars can create rich Excel files with multiple sheets and formatting. For more details, see the
API docs for `write_excel`.
