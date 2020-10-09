FROM python:3

RUN mkdir -p /chess /data
WORKDIR /chess
ENV DATA_DIR /data
RUN pip install --upgrade pip && \
    pip install poetry

COPY chesscog poetry.lock pyproject.toml ./
RUN poetry install

VOLUME /data

CMD python -m chesscog.synthesized_data.download_dataset