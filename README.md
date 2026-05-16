# Smart Document Intelligence Platform

An end-to-end AI/ML project covering the full machine learning lifecycle.

## Components
- Data pipeline: ingestion, cleaning, labeling, feature engineering
- NLP model: DistilBERT fine-tuned for document classification
- Experiment tracking: MLflow (Day 3-4)
- LLM + RAG: LangChain (Day 5-6)
- REST API: FastAPI (Day 7-8)
- Deployment: Docker (Day 7-8)
- CI/CD: GitHub Actions (Day 9-10)

## Tech Stack
Python, Hugging Face Transformers, PyTorch, scikit-learn,
LangChain, FastAPI, Docker, MLflow, GitHub Actions

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/pipeline.py   # build data pipeline
python src/train.py      # fine-tune DistilBERT
```