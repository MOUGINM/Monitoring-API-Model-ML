# Monitoring API Model ML

Ce projet implémente un système complet de **monitoring d’un modèle de machine learning**, avec les fonctionnalités suivantes :
- Entraînement et utilisation d’un modèle de classification basé sur les données Iris.
- Interface API Flask pour interagir avec le modèle via des requêtes HTTP.
- Suivi des performances et des métriques avec **MLflow**.

L'objectif principal est de fournir un exemple concret d’un pipeline **MLOps** simple et efficace, incluant :
1. La gestion du cycle de vie du modèle.
2. Le suivi des métriques et hyperparamètres.
3. L’interaction avec le modèle via une API REST.
---

## Prérequis

Avant de commencer, assurez-vous d’avoir les éléments suivants installés sur votre machine :
- **Python 3.8+**
- **pip** (gestionnaire de paquets Python)
- **MLflow**
- **Flask**

---

## Installation

1. Clonez le projet :
   ```bash
   git clone https://github.com/MOUGINM/Monitoring-API-Model-ML.git
   cd Monitoring-API-Model-ML
   
2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   
## Démarrage rapide

1. Lancer MLflow :
   ```bash
   mlflow ui
Accédez à l’interface de suivi via http://127.0.0.1:5000.
   
2. Lancer l’API Flask :
   ```bash
   python -m flask --app app.api run
Cela démarre l'API REST à http://127.0.0.1:5000.

3. Entraîner et surveiller le modèle avec le script principal :
   ```bash
   python main.py
Ce script entraîne le modèle, logue les métriques et effectue des prédictions de démonstration.


## Tests

1. Lancer les tests
Utilisez **pytest** pour vérifier la robustesse du projet :
   ```bash
   pytest --disable-warnings

## Utilisation de l’API REST

1. Entraîner le modèle
Envoyez une requête pour entraîner ou réentraîner le modèle :
   ```bash
   curl -X POST http://127.0.0.1:5000/train

Réponse :
         {
            "accuracy": 0.95
         }

2. Faire une prédiction
   ```bash
   curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

Réponse :
         {
             "prediction": "setosa"
         }

