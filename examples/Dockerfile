FROM rustlang/rust:nightly-slim

RUN apt-get update \
&& apt-get install \
    libssl-dev \
    lld \
    cmake \
    jupyter-notebook \
    pkg-config \
    -y \
&& cargo install evcxr_jupyter \
&& cargo install sccache \
&& evcxr_jupyter --install \
&& rm -rf /var/lib/apt/lists/*