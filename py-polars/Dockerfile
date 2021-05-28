FROM ritchie46/py-polars-base

RUN pip install pandas jupyterlab matplotlib
RUN mkdir /notebooks

WORKDIR /notebooks
COPY examples/* /notebooks/

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8890", "--no-browser", "--allow-root", "--NotebookApp.token=''"]