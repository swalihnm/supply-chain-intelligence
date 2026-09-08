# Serving image for the API and dashboard.
#
# Deliberately does NOT contain the pipeline's heavy dependencies. Training
# needs LightGBM, DuckDB and the full panel matrices; serving needs none of
# them, because scripts/build_serving_bundle.py has already reduced everything
# to compact numpy arrays. Installing the modelling extras here would roughly
# quadruple the image for code that never runs in it.
#
# Build:
#   uv run python scripts/build_serving_bundle.py
#   uv run python scripts/stage_deploy.py
#   docker build -t sci .
#
# Run the dashboard (default):   docker run -p 8501:8501 sci
# Run the API instead:           docker run -p 8000:8000 sci api

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SCI_DATA_DIR=/data

WORKDIR /app

# Serving dependencies only. Pinned to majors that match pyproject.
RUN pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "polars>=1.9" \
        "pyarrow>=17" \
        "pandas>=2.2" \
        "scipy>=1.14" \
        "pydantic>=2.9" \
        "pydantic-settings>=2.5" \
        "python-dotenv>=1.0" \
        "rich>=13.9" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "streamlit>=1.39" \
        "plotly>=5.24"

# Source. Only the packages the serving path imports.
COPY sci/       ./sci/
COPY app/       ./app/
COPY optimize/  ./optimize/
COPY evaluation/ ./evaluation/
COPY features/  ./features/
COPY models/    ./models/
COPY README.md  ./

# Artifacts staged by scripts/stage_deploy.py.
COPY deploy/data/artifacts/ /data/artifacts/

COPY docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

EXPOSE 8501 8000

# Streamlit's own health endpoint; the API has /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
p=os.environ.get('PORT','8501'); \
u=f'http://127.0.0.1:{p}/_stcore/health' if p=='8501' else f'http://127.0.0.1:{p}/health'; \
sys.exit(0 if urllib.request.urlopen(u, timeout=4).status==200 else 1)"

ENTRYPOINT ["entrypoint"]
CMD ["dashboard"]
