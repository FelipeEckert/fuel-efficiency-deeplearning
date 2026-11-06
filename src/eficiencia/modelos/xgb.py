"""XGBoost com restrição monotônica nas colunas de massa, inércia e forças F0/F2."""

from typing import ClassVar

import optuna
from xgboost import XGBRegressor

from eficiencia.modelos.base import CandidatoPorAlvo
from eficiencia.modelos.restricoes import sinais_monotonia


class XGBoost(CandidatoPorAlvo):
    nome = "xgboost"
    PARAMS_RAPIDOS: ClassVar[dict] = {"n_estimators": 20, "learning_rate": 0.3}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        restricoes = tuple(int(s) for s in sinais_monotonia(nomes, alvo))
        return XGBRegressor(
            monotone_constraints=restricoes, random_state=self.semente, n_jobs=-1, **self.params
        )

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }


# (mínimo, máximo, escala_log): fonte única da faixa ampliada
FAIXAS_AMPLO: dict[str, tuple[float, float, bool]] = {
    "n_estimators": (200, 3000, False),
    "max_depth": (2, 10, False),
    "learning_rate": (0.005, 0.3, True),
    "subsample": (0.4, 1.0, False),
    "colsample_bytree": (0.2, 1.0, False),
    "min_child_weight": (0, 20, False),
    "reg_lambda": (1e-3, 100, True),
}


class XGBoostAmplo(XGBoost):
    """XGBoost com faixa de busca ampliada; só para experimentos (fora de CANDIDATOS)."""

    nome = "xgboost_amplo"

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        f = FAIXAS_AMPLO
        inteiros = ("n_estimators", "max_depth", "min_child_weight")
        return {
            k: trial.suggest_int(k, int(lo), int(hi))
            if k in inteiros
            else trial.suggest_float(k, lo, hi, log=log)
            for k, (lo, hi, log) in f.items()
        }
