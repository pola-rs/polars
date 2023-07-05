FROM mcr.microsoft.com/devcontainers/python:3.9-bullseye

USER vscode

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# install the linker
USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

USER vscode

# install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

ENV PATH="$PATH:/home/vscode/.cargo/bin"

# Use this for experiments
# COPY polars /polars

