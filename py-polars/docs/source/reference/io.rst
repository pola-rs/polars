============
Input/output
============
.. currentmodule:: polars

Avro
~~~~
.. autosummary::
   :toctree: api/

   read_avro
   DataFrame.write_avro

Clipboard
~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_clipboard
   DataFrame.write_clipboard

CSV
~~~
.. autosummary::
   :toctree: api/

   read_csv
   read_csv_batched
   scan_csv
   DataFrame.write_csv
   LazyFrame.sink_csv

.. currentmodule:: polars.io.csv.batched_reader

.. autosummary::
   :toctree: api/

    BatchedCsvReader.next_batches

.. currentmodule:: polars

Database
~~~~~~~~
.. autosummary::
   :toctree: api/

   read_database
   read_database_uri
   DataFrame.write_database

Delta Lake
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_delta
   scan_delta
   DataFrame.write_delta

Excel / ODS
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_excel
   read_ods
   DataFrame.write_excel

Feather / IPC
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_ipc
   read_ipc_schema
   read_ipc_stream
   scan_ipc
   DataFrame.write_ipc
   DataFrame.write_ipc_stream
   LazyFrame.sink_ipc

Iceberg
~~~~~~~
.. autosummary::
   :toctree: api/

   scan_iceberg

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   read_ndjson
   scan_ndjson
   DataFrame.write_json
   DataFrame.write_ndjson
   LazyFrame.sink_ndjson

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   read_parquet_schema
   scan_parquet
   DataFrame.write_parquet
   LazyFrame.sink_parquet

PyArrow Datasets
~~~~~~~~~~~~~~~~
Connect to pyarrow datasets.

.. autosummary::
   :toctree: api/

   scan_pyarrow_dataset

Cloud Credentials
~~~~~~~~~~~~~~~~~
Configuration for cloud credential provisioning.

.. autosummary::
   :toctree: api/

   CredentialProvider
   CredentialProviderAWS
   CredentialProviderGCP
