import pytest
from app.model import Model

@pytest.fixture
def model_instance():
    return Model()

def test_training(model_instance):
    accuracy = model_instance.train()
    assert accuracy > 0.8, "L'accuracy est trop faible !"

def test_prediction(model_instance):
    model_instance.train()
    result = model_instance.predict([5.1, 3.5, 1.4, 0.2])
    assert len(result) == 1, "La prédiction doit retourner un seul résultat"
