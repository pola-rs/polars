============
Input/output
============
.. currentmodule:: polars

CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
   read_csv_batched
   scan_csv
   DataFrame.write_csv
   LazyFrame.sink_csv

Feather/ IPC
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_ipc
   read_ipc_stream
   scan_ipc
   read_ipc_schema
   DataFrame.write_ipc
   DataFrame.write_ipc_stream
   LazyFrame.sink_ipc

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   scan_parquet
   read_parquet_schema
   DataFrame.write_parquet
   LazyFrame.sink_parquet

Database
~~~~~~~~
.. autosummary::
   :toctree: api/

   read_database
   read_database_uri
   DataFrame.write_database

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   read_ndjson
   scan_ndjson
   DataFrame.write_json
   DataFrame.write_ndjson

AVRO
~~~~
.. autosummary::
   :toctree: api/

   read_avro
   DataFrame.write_avro

Spreadsheet
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_excel
   read_ods
   DataFrame.write_excel

Apache Iceberg
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   scan_iceberg

Delta Lake
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   scan_delta
   read_delta
   DataFrame.write_delta

Datasets
~~~~~~~~
Connect to pyarrow datasets.

.. autosummary::
   :toctree: api/

   scan_pyarrow_dataset


BatchedCsvReader
~~~~~~~~~~~~~~~~
This reader comes available by calling `pl.read_csv_batched`.

.. currentmodule:: polars.io.csv.batched_reader

.. autosummary::
   :toctree: api/

    BatchedCsvReader.next_batches
