# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgl1\
    libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
# copydependaency files
COPY pyproject.toml poetry.lock* ./
# install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi 

# Copy the rest of the application code
COPY . .
# Install the project itself
RUN poetry install --no-interaction --no-ansi
# Expose the FastAPI port
EXPOSE 8080

# Set environment variables (Placeholders, should be provided at runtime)
# ENV OPENAI_API_KEY=
# ENV QDRANT_URL=
# ENV QDRANT_API_KEY=
# ENV QDRANT_COLLECTION_NAME=
# ENV EMBEDDING_MODEL=
# ENV LLM_MODEL=

# Set the command to run the FastAPI server
CMD ["sh", "-c", "uvicorn karierai.server:app --host 0.0.0.0 --port ${PORT:-8080}"]

