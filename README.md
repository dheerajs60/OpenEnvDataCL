# OpenEnv Data Cleaning Environment

A rigorous data cleaning benchmark for AI agents, complying with OpenEnv standards.

## Running the API

You can start the FastAPI server via Docker or locally:

```bash
# Locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t dataclean-env .
docker run -p 8000:8000 dataclean-env
```

## Running Baseline Inference

Ensure the API is running locally on port 8000, then execute the inference script:

```bash
export API_BASE_URL="https://api.openai.com/v1" # or HF endpoint
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-token"

python inference.py
```

## Validation

Run validation to ensure structural requirements:

```bash
openenv validate
```
