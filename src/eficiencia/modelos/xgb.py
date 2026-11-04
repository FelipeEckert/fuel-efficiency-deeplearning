"""XGBoost com restrição monotônica nas colunas de massa, inércia e forças F0/F2."""

from typing import ClassVar

import optuna
from xgboost import XGBRegressor

from eficiencia.esquema import MONOTONAS
from eficiencia.modelos.base import CandidatoPorAlvo


class XGBoost(CandidatoPorAlvo):
    nome = "xgboost"
    PARAMS_RAPIDOS: ClassVar[dict] = {"n_estimators": 20, "learning_rate": 0.3}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        # consumo cresce com massa/F0/F2 (mj_km) e cai (km/l): sinal oposto por tipo de alvo
        sinal = 1 if alvo.startswith("mj_km") else -1
        restricoes = tuple(sinal if n in MONOTONAS else 0 for n in nomes)
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
