FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements-ml.txt
COPY . .
RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000 8501
CMD ["sh", "docker/entrypoint.sh"]
