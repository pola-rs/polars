FROM rustlang/rust:nightly-buster-slim
# Let's you run polars from jupyter. Compilation will happen at first run
# and will take some time.

RUN apt-get update \
&& apt-get install libssl-dev pkg-config cmake jupyter-notebook -y \
&& cargo install evcxr_jupyter \
&& evcxr_jupyter --install \
# cache compilations
&& cargo install sccache \
&& rm -rf /var/lib/apt/lists/*

COPY polars /polars
RUN mkdir --parents ~/.config/evcxr \
&& echo ':dep polars = { path = "/polars" }' > ~/.config/evcxr/init.evcxr

CMD [ "bash", "-c", "jupyter-notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=''" ]
