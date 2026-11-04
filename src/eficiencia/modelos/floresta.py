"""Random Forest, um modelo por alvo."""

from typing import ClassVar

import optuna
from sklearn.ensemble import RandomForestRegressor

from eficiencia.modelos.base import CandidatoPorAlvo


class Floresta(CandidatoPorAlvo):
    nome = "floresta"
    PARAMS_RAPIDOS: ClassVar[dict] = {"n_estimators": 20}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return RandomForestRegressor(random_state=self.semente, n_jobs=-1, **self.params)

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [200, 500]),
            "max_features": trial.suggest_categorical("max_features", [0.3, 0.6, 1.0]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
        }
