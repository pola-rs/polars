# Start with the Python 3.12 base image
FROM python:3.12-slim

# Install necessary packages
RUN apt-get update && \
    apt-get install -y curl build-essential libffi-dev pkg-config zlib1g-dev cmake make git libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Rust using rustup with stable toolchain
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable

# Add Rust to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install the nightly toolchain
COPY rust-toolchain.toml .
RUN rustup toolchain install nightly && rustup default nightly && rustup component add miri --toolchain nightly

# Verify the installation of Rust and Cargo
RUN rustup --version && rustc --version && cargo --version

RUN pip install maturin[patchelf]

CMD ["/bin/bash"]