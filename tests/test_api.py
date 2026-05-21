import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
import time

os.environ["TESTING"] = "1"

from api import app, startup_time
import api

# Set startup_time so health check doesn't fail with NoneType
api.startup_time = time.time()

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "uptime_seconds" in data
    assert "rag_loaded" in data
    assert "classifier_loaded" in data
    assert "version" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "api_version" in data
    assert "classifier_model" in data
    assert "supported_classes" in data
    assert len(data["supported_classes"]) == 4


def test_query_empty_question():
    # Returns 400 if retriever loaded, 503 if not — both are acceptable
    response = client.post("/query", json={"question": ""})
    assert response.status_code in [400, 503]


def test_classify_empty_text():
    # Returns 400 if classifier loaded, 503 if not — both are acceptable
    response = client.post("/classify", json={"text": ""})
    assert response.status_code in [400, 503]


def test_query_valid_question():
    response = client.post(
        "/query",
        json={"question": "What is machine learning?"}
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_time" in data


def test_classify_valid_text():
    response = client.post(
        "/classify",
        json={"text": "deep learning neural network classification"}
    )
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0