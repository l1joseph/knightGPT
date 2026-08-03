# docker/postgres/Dockerfile
ARG RUST_IMAGE=rust:1.96.0-bookworm@sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc
ARG POSTGRES_IMAGE=postgres:17-bookworm@sha256:4f736ae292687621d4dbe0d499ffd024a36bd2ee7d8ca6f2ccd4c800f047b394
ARG PG_MAJOR=17
ARG PGRX_VERSION=0.19.1
ARG PGGRAPH_REF=v1.0.0

# ---- pgGraph builder ----
FROM ${RUST_IMAGE} AS pggraph-builder
ARG PG_MAJOR
ARG PGRX_VERSION
ARG PGGRAPH_REF
RUN apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release git \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get -o Acquire::Retries=3 update \
    && apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
        postgresql-${PG_MAJOR} postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-pgrx --version ${PGRX_VERSION} --locked
RUN git clone --depth 1 --branch ${PGGRAPH_REF} https://github.com/evokoa/pggraph.git /src/pggraph
WORKDIR /src/pggraph/graph
RUN cargo pgrx init --pg${PG_MAJOR}=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config \
    && cargo pgrx package --pg-config=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

# ---- final image ----
FROM ${POSTGRES_IMAGE}
ARG PG_MAJOR
LABEL org.opencontainers.image.description="PostgreSQL with pgGraph for knightGPT"

COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/share/postgresql/${PG_MAJOR}/extension/graph* \
    /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=pggraph-builder /src/pggraph/graph/target/release/graph-pg${PG_MAJOR}/usr/lib/postgresql/${PG_MAJOR}/lib/graph.so \
    /usr/lib/postgresql/${PG_MAJOR}/lib/

ENV POSTGRES_DB=knightgpt
COPY docker/postgres/init/01-create-extensions.sql /docker-entrypoint-initdb.d/

CMD ["postgres", "-c", "shared_preload_libraries=graph"]
