FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# install the linker
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends cmake gcc

# install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

RUN ls ~/.cargo/bin

RUN echo $HOME
ENV PATH="$PATH:$HOME/.cargo/bin"


#COPY polars /polars

