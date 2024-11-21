import mlflow
import mlflow.sklearn
from app.model import Model
from app.monitoring import Monitor
import time

if __name__ == "__main__":
    # Initialiser le modèle et le monitoring
    model = Model()
    monitor = Monitor()

    # Définir les hyperparamètres
    params = {"max_iter": 200, "solver": "lbfgs", "C": 1.0}

    # Simuler des métriques
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 1]

    # Tout regrouper dans un seul run
    with mlflow.start_run() as run:
        # Loguer les hyperparamètres
        print("Enregistrement des hyperparamètres...")
        mlflow.log_params(params)

        # Entraîner le modèle et enregistrer l'accuracy
        print("Entraînement du modèle...")
        accuracy = model.train_with_params(params)
        mlflow.log_metric("accuracy", accuracy)

        # Enregistrer le modèle dans MLflow
        print("Enregistrement du modèle...")
        mlflow.sklearn.log_model(model.model, "iris_logistic_regression")

        # Enregistrer les métriques supplémentaires
        print("Enregistrement des métriques...")
        monitor.log_metrics(y_true, y_pred)

        # Enregistrer le temps d'entraînement
        print("Enregistrement du temps d'entraînement...")
        start_time = time.time()
        model.train_with_params(params)
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

    print("Tout est enregistré dans un seul run dans MLflow !")
