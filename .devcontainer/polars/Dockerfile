FROM mcr.microsoft.com/devcontainers/rust:1

# install the linker
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get -y install --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

USER vscode

ENV PATH="$PATH:/home/vscode/.cargo/bin"

# Use this for experiments
# COPY --chown=vscode polars /home/vscode/polars

