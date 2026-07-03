import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from homework_model_modification import AdvancedLinearRegression
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')


def run_experiment(lr, batch_size, optimizer_name, X, y):
    """Запускает обучение модели с заданными гиперпараметрами."""
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AdvancedLinearRegression(in_features=X.shape[1])
    criterion = nn.MSELoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    losses = []
    epochs = 10
    for _ in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch) + model.compute_penalty()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(loader))
    return losses


def feature_engineering(X):
    """Добавляет полиномиальные, интерактивные и статистические признаки."""
    poly = X ** 2
    interaction = (X[:, 0] * X[:, 1]).unsqueeze(1)
    mean = X.mean(dim=1, keepdim=True)
    return torch.cat([X, poly, interaction, mean], dim=1)


if __name__ == "__main__":
    X, y = make_regression_data(source="diabetes")
    logging.info("--- Сравнение оптимизаторов ---")
    plt.figure(figsize=(8, 5))
    for opt in ["SGD", "Adam", "RMSprop"]:
        history = run_experiment(0.01, 16, opt, X, y)
        plt.plot(history, label=opt)
        logging.info(f"Optimizer {opt}: Final Loss {history[-1]:.4f}")
    plt.title("Сравнение оптимизаторов")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(PLOTS_DIR / "optimizer_comparison.png")
    plt.close()
    logging.info("\n--- Сравнение Learning Rate ---")
    for lr in [0.001, 0.01, 0.1]:
        history = run_experiment(lr, 16, "SGD", X, y)
        logging.info(f"Learning rate {lr}: Final Loss {history[-1]:.4f}")
    logging.info("\n--- Сравнение Batch Size ---")
    for batch in [8, 16, 32]:
        history = run_experiment(0.01, batch, "SGD", X, y)
        logging.info(f"Batch size {batch}: Final Loss {history[-1]:.4f}")
    logging.info("\n--- Feature Engineering ---")
    base_history = run_experiment(0.01, 16, "SGD", X, y)
    X_new = feature_engineering(X)
    new_history = run_experiment(0.01, 16, "SGD", X_new, y)
    logging.info(f"Исходных признаков: {X.shape[1]}")
    logging.info(f"Новых признаков: {X_new.shape[1]}")
    logging.info(f"Loss до Feature Engineering: {base_history[-1]:.4f}")
    logging.info(f"Loss после Feature Engineering: {new_history[-1]:.4f}")
