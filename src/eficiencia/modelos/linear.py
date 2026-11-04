"""Regressão linear regularizada (Ridge), um modelo por alvo."""

from typing import ClassVar

import optuna
from sklearn.linear_model import Ridge

from eficiencia.modelos.base import CandidatoPorAlvo


class Linear(CandidatoPorAlvo):
    nome = "linear"
    PARAMS_RAPIDOS: ClassVar[dict] = {}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return Ridge(alpha=self.params.get("alpha", 1.0))

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}
