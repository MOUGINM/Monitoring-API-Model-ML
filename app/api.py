from flask import Flask, request, jsonify
from app.model import Model
from app.monitoring import Monitor

app = Flask(__name__)
model = Model()
monitor = Monitor()

@app.route("/train", methods=["POST"])
def train():
    """Entraîne le modèle et retourne l'accuracy."""
    accuracy = model.train()
    return jsonify({"message": "Modèle entraîné", "accuracy": accuracy})

@app.route("/predict", methods=["POST"])
def predict():
    """Effectue une prédiction à partir des features fournies."""
    data = request.json
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "Aucune feature fournie"}), 400

    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

@app.route("/log_metrics", methods=["POST"])
def log_metrics():
    """Logue des métriques dans MLflow."""
    data = request.json
    y_true = data.get("y_true", [])
    y_pred = data.get("y_pred", [])
    if not y_true or not y_pred:
        return jsonify({"error": "y_true et y_pred doivent être fournis"}), 400

    monitor.log_metrics(y_true, y_pred)
    return jsonify({"message": "Métriques enregistrées"})
