FROM rustlang/rust:nightly-slim

RUN apt-get update \
&& apt-get install \
    libssl-dev \
    lld \
    cmake \
    jupyter-notebook \
    pkg-config \
    git \
    -y \
&& rm -rf /var/lib/apt/lists/*

RUN useradd -m -d /home/polars -s /bin/bash -U -u 1000 polars \
&& chown polars /usr/local/
USER 1000
RUN mkdir --parents home/polars/.config/evcxr \
&& cargo install evcxr_jupyter \
&& cargo install sccache \
&& evcxr_jupyter --install \
&& echo ':dep polars = { path = "/polars" }' > home/polars/.config/evcxr/init.evcxr

RUN mkdir -p $(jupyter --data-dir)/nbextensions \
&& cd $(jupyter --data-dir)/nbextensions \
&& git clone --depth=1 https://github.com/lambdalisue/jupyter-vim-binding vim_binding \
&& jupyter nbextension enable vim_binding/vim_binding


CMD [ "bash", "-c", "jupyter-notebook --no-browser --ip=0.0.0.0 --NotebookApp.token=''" ]
