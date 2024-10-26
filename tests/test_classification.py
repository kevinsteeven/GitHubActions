import unittest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from src.main import create_mock_dataset, train_model

class TestClassificationModel(unittest.TestCase):

    def test_create_mock_dataset(self):
        X, y = create_mock_dataset(100)
        self.assertEqual(X.shape, (100, 2), "El dataset debe tener 100 muestras y 2 características")
        self.assertEqual(len(y), 100, "El vector de etiquetas debe tener 100 elementos")
        self.assertEqual(np.unique(y).tolist(), [0, 1], "Las clases deben ser 0 y 1")

    def test_train_model_accuracy(self):
        X, y = create_mock_dataset(100)
        model = train_model(X, y)  # Guarda el modelo para evaluar
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        self.assertGreaterEqual(accuracy, 0.5, "La precisión debe ser al menos 0.5")

    def test_train_model_precision(self):
        X, y = create_mock_dataset(100)
        model = train_model(X, y)  # Guarda el modelo para evaluar
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred)
        self.assertGreaterEqual(precision, 0.5, "La precisión debe ser al menos 0.5")

# Ejecutar las pruebas solo si este archivo se ejecuta directamente
if __name__ == "__main__":
    unittest.main()

# python -m unittest discover -s . -p "test_*.py"