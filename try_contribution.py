import polars as pl
import zipfile

def read_csv_from_zip(path, target_files=None):
    dataframes = {}

    with zipfile.ZipFile(path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith(".csv") and not file_name.startswith("__") and not file_name.startswith("."):
                    if target_files is not None and file_name not in target_files:
                        continue
                    with zip_ref.open(file_name) as csv_file:
                        df = pl.read_csv(csv_file)
                        dataframes[file_name] = df

            return dataframes

dataframes = read_csv_from_zip("docs/assets/data/test_csv.zip", ["apple_stock.csv", "iris.csv"])

print(dataframes)

