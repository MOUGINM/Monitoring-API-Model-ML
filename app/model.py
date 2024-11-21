from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

class Model:
    def __init__(self, model_path="iris_model.pkl"):
        self.model_path = model_path
        self.model = None

    def train_with_params(self, params):
        """
        Entraîner le modèle avec des hyperparamètres spécifiques.
        """
        # Charger les données Iris
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        # Entraîner avec les hyperparamètres
        self.model = LogisticRegression(**params)
        self.model.fit(X_train, y_train)

        # Calculer l'accuracy
        accuracy = self.model.score(X_test, y_test)

        # Sauvegarder le modèle localement
        joblib.dump(self.model, self.model_path)
        print(f"Modèle entraîné avec une accuracy de {accuracy}")
        return accuracy
