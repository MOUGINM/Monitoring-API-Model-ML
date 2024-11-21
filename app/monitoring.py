import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Monitor:
    def __init__(self, experiment_name="Monitoring_Project"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_metrics(self, y_true, y_pred):
        """
        Logue des métriques avancées. Utilise un run déjà actif.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Pas de nouveau `mlflow.start_run()`, on logue directement
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        print("Métriques enregistrées : accuracy, f1_score, precision, recall")

    def log_model(self, model, model_name="sklearn_model"):
        """
        Enregistre le modèle dans MLflow.
        """
        mlflow.sklearn.log_model(model, model_name)
        print(f"Modèle enregistré sous le nom : {model_name}")

    def log_training_time(self, training_function):
        """
        Logue le temps d'entraînement dans MLflow. Utilise un run déjà actif.
        """
        import time
        start_time = time.time()
        result = training_function()
        training_time = time.time() - start_time

        # Pas de nouveau `mlflow.start_run()`, on logue directement
        mlflow.log_metric("training_time", training_time)
        print(f"Temps d'entraînement enregistré : {training_time}s")
        return result
