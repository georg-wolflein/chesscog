FROM python:3.8

# Install zip
RUN apt update && \
    apt install -y zip

# Install poetry
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false

# Install dependencies
RUN mkdir -p /chess
WORKDIR /chess
COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry install --no-root
ENV PYTHONPATH "/chess:${PYTHONPATH}"

# Setup data mount
RUN mkdir -p /data
ENV DATA_DIR /data
VOLUME /data

# Setup config mount
RUN mkdir -p /config
ENV CONFIG_DIR /config
VOLUME /config

# Setup run mount
RUN mkdir -p /chess/runs
ENV RUN_DIR /chess/runs
VOLUME /chess/runs

# Setup results mount
RUN mkdir -p /chess/results
ENV RESULTS_DIR /chess/results
VOLUME /chess/results

# Setup models mount
RUN mkdir -p /chess/models
ENV MODELS_DIR /chess/models
VOLUME /chess/models

# Copy files
COPY chesscog ./chesscog

# Scratch volume
VOLUME /chess/scratch

# Entrypoint (password is "chesscog")
CMD poetry run tensorboard --logdir ./runs --host 0.0.0.0 --port 9999  & \
    poetry run jupyter lab --no-browser --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.password "sha1:22fda334b4b5:770a9d781f1e689afdcd2c55e7abae94ba74d925"