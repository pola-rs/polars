import polars as pl
import zipfile

# TODO: Read only single file from zip
def read_csv_from_zip(path, target_file=None):
    with zipfile.ZipFile(path, 'r') as zip_ref:

        if target_file is not None:
            dataframe = None
            with zip_ref.open(target_file) as csv_file:
                df = pl.read_csv(csv_file)
                dataframe = df
                return dataframe

        else:
            dataframes = {}
            for file_name in zip_ref.namelist():
                if file_name.endswith(".csv") and not file_name.startswith("__") and not file_name.startswith("."):
                    with zip_ref.open(file_name) as csv_file:
                        df = pl.read_csv(csv_file)
                        dataframes[file_name] = df
            return dataframes

dataframes = read_csv_from_zip("docs/assets/data/test_csv.zip", "apple_stock.csv")

print(dataframes)

