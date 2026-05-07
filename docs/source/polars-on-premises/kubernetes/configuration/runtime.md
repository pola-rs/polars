# Runtime

Polars On-Prem consists of a single scheduler and multiple workers. Both components are contained in
a single binary. While the scheduler can run without any system-level dependencies, the worker node
needs the following:

- Python runtime (any version or a matching client version for running UDFs)
- Polars (i.e. pip wheel)
- Additional python requirements (e.g. numpy, pyarrow)

If your Kubernetes cluster has internet access, we support installing these all at worker boot. For
example, you can use the following configuration:

```yaml
runtime:
  prebuilt:
    enabled: false

  composed:
    enabled: true

    dist:
      repository: "polarscloud/polars-on-premises"
      tag: ""
      pullPolicy: "IfNotPresent"

    runtime:
      repository: "python"
      tag: "3.13.9-slim-bookworm"
      pullPolicy: ""

    requirements: |
      boto3==1.40.70
      urllib==2.5.0

    polarsExtras: "async,cloudpickle,database,deltalake,fsspec,iceberg,numpy,pandas,pyarrow,pydantic,timezone"
```

Behind the scenes, this mechanism copies the Polars On-Prem binary, wheel, uv, and a setup script
from an init-container to the pod's main container. On startup of the main container, the setup
script uses uv to install the polars wheel with the additional specified packages before starting
the worker.

If you prefer self-building a Docker image, you can instead configure the chart to use your image:

```yaml
runtime:
  prebuilt:
    enabled: false

    runtime:
      repository: "your-prebuilt-image"
      tag: ""
      pullPolicy: ""

  composed:
    enabled: false
```

The Dockerfile for a prebuilt image can look something like this:

```Dockerfile
# your specific python version (you could also use any other base image and install python in a run statement)
FROM python:3.13.9-slim-bookworm

WORKDIR /opt

# your os-level dependencies
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# install polars with specific features enabled for these packages
ENV POLARS_EXTRAS="async,cloudpickle,database,deltalake,fsspec,iceberg,numpy,pandas,pyarrow,pydantic,timezone"

# enale bash so we can write multiline strings
SHELL ["/bin/bash", "-c"]
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.8,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=from=polarscloud/polars-on-premises:20251203,source=/opt/whl/polars-1.35.2-py3-none-any.whl,target=/opt/polars-1.35.2-py3-none-any.whl \
    --mount=type=bind,source=./requirements.txt,target=/opt/requirements.txt \
    # install polars with extras from wheel overriding to prevent installation of runtimes as those are already included in pc-cublet \
    echo -e "\
polars[$POLARS_EXTRAS] @ file:///opt/polars-1.35.2-py3-none-any.whl\n\
polars-runtime-32; sys_platform == 'never'\n\
polars-runtime-64; sys_platform == 'never'\n\
polars-runtime-compat; sys_platform == 'never'\n" | \
  uv pip install \
  -r requirements.txt \
  "polars[$POLARS_EXTRAS] @ file:///opt/polars-1.35.2-py3-none-any.whl" \
  --system \
  --overrides=-

COPY --from=polarscloud/polars-on-premises:20251203 /opt/bin/pc-cublet /opt/bin/pc-cublet

CMD ["/opt/bin/pc-cublet", "service"]
```

Note that the helm tests will still use the runtime defined in `.Values.runtime.composed.runtime`,
so ensure that this image contains the same Python version as your prebuilt image. All the Python
dependencies required for the tests are already included in the image used in the helm test, and the
test does not require internet access, so a prebuilt image for the tests has no direct advantage.
