import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from homework_model_modification import AdvancedLinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')


class CustomCSVDataset(Dataset):
    """Датасет для работы с CSV файлами."""

    def __init__(
        self,
        csv_file,
        target_col,
        feature_cols=None,
        scale_target=False,
    ):
        df = pd.read_csv(csv_file)
        if target_col not in df.columns:
            raise ValueError(f"Столбец '{target_col}' не найден.")
        if feature_cols is None:
            X = df.drop(columns=[target_col])
        else:
            X = df[feature_cols]
        y = df[target_col]
        X = pd.get_dummies(X, drop_first=True)
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        if scale_target:
            self.target_scaler = StandardScaler()
            y = self.target_scaler.fit_transform(
                y.values.reshape(-1, 1)
            )
        else:
            y = y.values.reshape(-1, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_classification_titanic():
    logging.info("Обучение модели на Titanic")
    dataset = CustomCSVDataset(
        DATA_DIR / "train.csv",
        target_col="Survived",
        feature_cols=[
            "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
        ],
    )
    logging.info(f"Размер датасета: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = nn.Linear(len(dataset.feature_names), 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch}: loss={total_loss / len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            pred = (torch.sigmoid(model(batch_X)) > 0.5).float()
            correct += (pred == batch_y).float().sum().item()
            total += len(batch_y)
    logging.info(f"Accuracy: {correct / total:.4f}")
    torch.save(model.state_dict(), MODELS_DIR / "titanic_logistic.pth")


def train_regression_insurance():
    logging.info("Обучение модели на Insurance")
    dataset = CustomCSVDataset(
        DATA_DIR / "insurance.csv", target_col="charges", scale_target=True
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AdvancedLinearRegression(dataset.X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y) + model.compute_penalty()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 2 == 0:
            logging.info(f"Epoch {epoch}: loss={total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), MODELS_DIR / "insurance_linear.pth")


if __name__ == "__main__":
    train_classification_titanic()
    train_regression_insurance()
