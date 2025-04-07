# Google Sheets (via Colab)

Google Colab provides a utility class to read from and write to Google Sheets.

## Opening and reading from a sheet

We can open existing sheets by initializing `sheets.InteractiveSheet` with either:

- the `url` parameter, for example
  https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/
- the `sheet_id` parameter for example 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms

By default the left-most worksheets will be used, we can change this by providing either
`worksheet_id` or `worksheet_name`.

The first time in each session that we use `InteractiveSheet` we will need to give Colab permission
to edit our drive assets on our behalf.

{{code_block('user-guide/io/sheets_colab','open',[])}}

## Creating a new sheet

When you don't provide the source of the spreadsheet one will be created for you.

{{code_block('user-guide/io/sheets_colab','create_title',[])}}

When you pass the `df` parameter the data will be written to the sheet immediately.

{{code_block('user-guide/io/sheets_colab','create_df',[])}}

## Writing to a sheet

By default the `update` method will clear the worksheet and write the dataframe in the top left
corner.

{{code_block('user-guide/io/sheets_colab','update',[])}}

We can modify where the data is written with the `location` parameter and whether the worksheet is
cleared before with `clear`.

{{code_block('user-guide/io/sheets_colab','update_loc',[])}}

A good way to write multiple dataframes onto a worksheet in a loop is:

{{code_block('user-guide/io/sheets_colab','update_loop',[])}}

This clears the worksheet then writes the dataframes next to each other, one every five columns.
