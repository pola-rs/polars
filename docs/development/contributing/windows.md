# Windows Development

This guide describes in detail how to get polars development up and running on a Windows PC. If any of these
instructions do not work in your particular case, please [create an issue](https://github.com/pola-rs/polars/issues)
in the Polars GitHub issue tracker and we will work to resolve your case.

## Windows Subsystem for Linux (WSL)

Windows does not natively support `make` tasks, and aspects of the code base may not immediately compile on Windows. For this reason we
suggest using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/), which allows developers in Windows to
run a Linux environment without dual-booting or a separate virtual machine. Setting up WSL involves the following steps (see the official
[WSL Installation](https://learn.microsoft.com/en-us/windows/wsl/install) for details). Note that this requires Administrator privileges.

> **Note**: WSL creates a separate hard drive partition that uses a different filesystem from Windows. While the WSL can technically access
> your main Windows hard drives, results can be fairly volatile. Ensure that all files used (e.g. from a `git clone`) are located within the
> WSL environment.

### Install WSL

1. Ensure that Virtualization is enabled in your BIOS settings. This is usually found in the `Advanced` or `CPU` section.
2. Open up a command prompt with Administrator access and install WSL:

   ```shell
   C:\>wsl --install
   Installing: Windows Subsystem for Linux
   Windows Subsystem for Linux has been installed.
   Installing: Ubuntu
   Ubuntu has been installed.
   The requested operation is successful. Changes will not be effective until the system is rebooted.
   ```

   A reboot is required.

3. You can now enter WSL in a command prompt by typing `wsl`.
4. Update your system with `sudo apt update && sudo apt upgrade`.
5. Python should already be installed on your system, which you can run with the `python3` command.

### Fork and clone Polars

#### Fork polars

1. Navigate to https://github.com/pola-rs/polars and select `Fork` on the top right to fork your own version of polars.
2. Click the ![image](code.png) button and copy the URL to clipboard. HTTPS is the preferred method.

#### Clone repository

1. Create or navigate to your directory of choice, e.g.:

   ```shell
   user@PC:~$ mkdir ~/projects
   user@PC:~$ cd ~/projects
   ```
2. Clone your forked repo.

   ```shell
   user@PC:~/projects$ git clone https://github.com/user/polars.git
   ```

### Use VSCode to remote into WSL

[VSCode](https://code.visualstudio.com/) has the ability to connect to your WSL virtual system as if you had opened
it on a linux machine, with the side-bar file explorer and command windows all operating inside the WSL. To use this
feature:

1. Install the [WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) extension
   by Microsoft.
2. Type `ctrl+shift+p` to bring up the command palette and type `WSL: Connect to WSL`
3. Open your polars project with `File → Open Folder...` and select the root polars folder of your cloned repository.
4. Open a command prompt (in VS Code this is `` ctrl + ` ``).
5. Create and setup your virtual environment:

   ```shell
   user@PC:~/projects/polars$ make .venv
   ```

6. Build polars:

   ```shell
   user@PC:~/projects/polars$ make build
   ```

7. Test your polars build:

   ```shell
   user@PC:~/projects/polars$ cd py-polars
   user@PC:~/projects/polars/py-polars$ make test
   ```

## Native Windows compilation

If you wish to develop polars locally on your Windows machine (not recommended), or if you wish to compile polars in
Windows for maximum performance, a few steps must be taken to ensure compilation will succeed. This section assumes
that rust is installed.

Check the file `py-polars/rust-toolchain.toml`. This file is meant to be a symbolic link of the file `rust-toolchain.toml`
in the root directory. Git rarely successfully creates this symbolic link, and so the file is most likely invalid. To override
this file, we will create a file at `py-polars/rust-toolchain` (with no extension), which will take precedence over the existing
toml file. From the root polars directory, run the following command:

```shell
C:\Projects\polars>copy "rust-toolchain.toml" "py-polars/rust-toolchain"
        1 file(s) copied.
```

### Makefile-equivalent commands

Because `make` is not natively supported in Windows, each Make command must be manually run. Inspect the `Makefile` in the
root directory for the specific build commands. First create and then activate a virtual environment:

```shell
C:\Projects\polars>python -m venv .venv
C:\Projects\polars>.venv\scripts\activate
```

At this point, we can run following (subset of) commands which are rough equivalents to those found in the Makefile.

<details>
<summary>Make commands (click to expand)</summary>

#### `make requirements`

_Install/refresh Python project requirements_

```shell
python -m pip install --upgrade -r py-polars\requirements-dev.txt
python -m pip install --upgrade -r py-polars\requirements-lint.txt
python -m pip install --upgrade -r py-polars\docs\requirements-docs.txt
python -m pip install --upgrade -r docs\requirements.txt
```

#### `make build`

_Compile and install Python Polars for development_

```shell
maturin develop -m py-polars/Cargo.toml
```

#### `make build-release`

_Compile and install a faster Python Polars binary with full optimizations_

```shell
maturin maturin develop -m py-polars\Cargo.toml --release
```

#### `make build-release-native`

_Same as build-release, except with native CPU optimizations turned on_

```shell
maturin maturin develop -m py-polars\Cargo.toml --release -- -C target-cpu=native
```

#### `make fmt`

```shell
ruff check
ruff format
cargo fmt --all
typos
```

#### `make pre-commit`

```shell
ruff check
ruff format
cargo fmt --all
typos
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo clippy --all-targets --locked -- -D warnings
```

#### `make clean`

```shell
if exist .venv ( rmdir .venv /s /q )
if exist target ( rmdir target /s /q )
if exist Cargo.lock ( del Cargo.lock /s /q )
if exist py-polars\target ( rmdir %py_path%\target /s /q )
if exist py-polars\target ( rmdir %py_path%\target /s /q )
if exist py-polars\docs\build ( rmdir %py_path%\docs\build /s /q )
if exist py-polars\docs\source\reference\api ( rmdir %py_path%\docs\source\reference\api /s /q )
if exist py-polars\.hypothesis ( rmdir %py_path%\.hypothesis /s /q )
if exist py-polars\.mypy_cache ( rmdir %py_path%\.mypy_cache /s /q )
if exist py-polars\.pytest_cache ( rmdir %py_path%\.pytest_cache /s /q )
if exist py-polars\.ruff_cache ( rmdir %py_path%\.ruff_cache /s /q )
if exist py-polars\.coverage ( rmdir %py_path%\.coverage /s /q )
if exist py-polars\coverage.xml ( del %py_path%/coverage.xml /s /q )
if exist py-polars\polars/polars.abi3.so ( del %py_path%/polars/polars.abi3.so /s /q )
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
cargo clean
cd py-polars && cargo clean
```

</details>
