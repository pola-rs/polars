# Use the Microsoft devcontainer Rust image as the base
FROM mcr.microsoft.com/devcontainers/rust:latest

# Update the package lists
RUN apt-get update

# Install Python and other necessary packages
RUN apt-get install -y python3 python3-pip pkg-config build-essential zlib1g-dev

# Set the working directory
WORKDIR /workspace

# Install nightly toolchain and Miri
RUN rustup toolchain install nightly --component miri

# Install dprint
RUN curl -fsSL https://dprint.dev/install.sh | bash
ENV PATH="/home/vscode/.dprint/bin:${PATH}"

# # Set up Python aliases and environment variables
# RUN echo "alias python=python3" >> ~/.bashrc && \
#     echo "alias pip=pip3" >> ~/.bashrc && \
#     echo "export PATH=$PATH:/home/vscode/.local/bin:/home/vscode/.cargo/bin:/home/vscode/.dprint/bin" >> ~/.bashrc

# # Ensure the virtual environment is activated in the terminal if it exists
# RUN echo "if [ -d \"\${PWD}/.venv\" ]; then source \${PWD}/.venv/bin/activate; fi" >> ~/.bashrc

CMD ["/bin/bash"]