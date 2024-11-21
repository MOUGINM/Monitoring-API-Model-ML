import pytest
from app.api import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()

def test_train(client):
    response = client.post("/train")
    assert response.status_code == 200
    data = response.get_json()
    assert "accuracy" in data

def test_predict(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
