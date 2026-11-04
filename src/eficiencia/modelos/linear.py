"""Regressão linear ridge com limites de sinal nas colunas monotônicas, um modelo por alvo."""

from typing import ClassVar

import numpy as np
import optuna
from scipy.optimize import lsq_linear

from eficiencia.modelos.base import CandidatoPorAlvo
from eficiencia.modelos.restricoes import sinais_monotonia


class _RidgeComSinal:
    """Ridge por mínimos quadrados limitados: coef >= 0 onde o sinal é +1, <= 0 onde é -1."""

    def __init__(self, alpha: float, sinais: np.ndarray) -> None:
        self.alpha = alpha
        self.inferior = np.where(sinais > 0, 0.0, -np.inf)
        self.superior = np.where(sinais < 0, 0.0, np.inf)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # ||Xw - (y - ȳ)||² + alpha||w||² como um único problema de mínimos quadrados
        p = X.shape[1]
        A = np.vstack([X, np.sqrt(self.alpha) * np.eye(p)])
        b = np.concatenate([y - y.mean(), np.zeros(p)])
        self.intercepto = y.mean()
        self.coef = lsq_linear(A, b, bounds=(self.inferior, self.superior), method="bvls").x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercepto


class Linear(CandidatoPorAlvo):
    nome = "linear"
    PARAMS_RAPIDOS: ClassVar[dict] = {}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return _RidgeComSinal(self.params.get("alpha", 1.0), sinais_monotonia(nomes, alvo))

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
