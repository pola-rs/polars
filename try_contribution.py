import polars as pl
import zipfile

def read_csv_from_zip(path):
    dataframes = []

    with zipfile.ZipFile(path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            print(file_name)




read_csv_from_zip("docs/assets/data/test_csv.zip")

