"""CatBoost e LightGBM com a mesma monotonia do XGBoost; só para experimentos (fora de CANDIDATOS)."""

from typing import ClassVar

import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from eficiencia.modelos.base import CandidatoPorAlvo
from eficiencia.modelos.restricoes import sinais_monotonia


class CatBoost(CandidatoPorAlvo):
    nome = "catboost"
    PARAMS_RAPIDOS: ClassVar[dict] = {"iterations": 20, "learning_rate": 0.3}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return CatBoostRegressor(
            monotone_constraints=[int(s) for s in sinais_monotonia(nomes, alvo)],
            random_seed=self.semente,
            thread_count=-1,
            verbose=0,
            allow_writing_files=False,  # senão cria a pasta catboost_info/ a cada fit
            **self.params,
        )

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-2, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        }


class LightGBM(CandidatoPorAlvo):
    nome = "lightgbm"
    PARAMS_RAPIDOS: ClassVar[dict] = {"n_estimators": 20, "learning_rate": 0.3}

    def _novo_estimador(self, alvo: str, nomes: list[str]):
        return LGBMRegressor(
            monotone_constraints=[int(s) for s in sinais_monotonia(nomes, alvo)],
            random_state=self.semente,
            n_jobs=-1,
            subsample_freq=1,  # sem isso o subsample é ignorado
            verbose=-1,
            **self.params,
        )

    @staticmethod
    def sugerir(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # o padrão (20 linhas por folha) é grande para ~500 carros
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
