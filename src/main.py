import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Generar un dataset mock
def create_mock_dataset(num_samples=100):
    np.random.seed(42)
    X = np.random.rand(num_samples, 2)  # 2 características
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Clase 1 si la suma de las características es > 1
    return pd.DataFrame(X, columns=['feature1', 'feature2']), y

# Entrenar el modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    
    return model  # Devuelve el modelo entrenado

if __name__ == "__main__":
    X, y = create_mock_dataset()
    train_model(X, y)
