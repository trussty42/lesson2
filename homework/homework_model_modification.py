import torch.nn as nn
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')


class AdvancedLinearRegression(nn.Module):
    """Линейная регрессия с L1 и L2 регуляризацией."""

    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def compute_penalty(self):
        l1_penalty = 0
        l2_penalty = 0
        for param in self.parameters():
            l1_penalty += param.abs().sum()
            l2_penalty += param.square().sum()
        return self.l1_lambda * l1_penalty + self.l2_lambda * l2_penalty


class LogisticRegressionMulticlass(nn.Module):
    """Логистическая регрессия для многоклассовой классификации."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)


class EarlyStopping:
    """
    Класс для ранней остановки обучения,
    если ошибка перестала уменьшаться.
    """

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
