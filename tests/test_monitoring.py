import pytest
from app.monitoring import Monitor

def test_log_metrics():
    monitor = Monitor()
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]

    try:
        monitor.log_metrics(y_true, y_pred)
    except Exception as e:
        pytest.fail(f"Échec du log des métriques dans MLflow ! Erreur : {e}")

def test_log_training_time():
    monitor = Monitor()

    def dummy_training():
        return 0.95  # Simule une accuracy

    try:
        monitor.log_training_time(dummy_training)
    except Exception as e:
        pytest.fail(f"Échec du log du temps d'entraînement dans MLflow ! Erreur : {e}")
