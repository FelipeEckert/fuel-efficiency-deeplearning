"""Interface comum dos candidatos: fit/predict em DataFrames e espaço de busca do Optuna."""

from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import optuna
import pandas as pd

from eficiencia.esquema import ALVOS
from eficiencia.preprocess import Preprocessador


class Candidato(ABC):
    nome: str
    PARAMS_RAPIDOS: ClassVar[dict] = {}

    def __init__(self, semente: int = 0, **params) -> None:
        self.semente = semente
        self.params = params

    @staticmethod
    @abstractmethod
    def sugerir(trial: optuna.Trial) -> dict:
        """Hiperparâmetros sugeridos pelo Optuna para um trial."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        """X: colunas ENTRADAS; Y: colunas ALVOS, NaN onde o alvo não se aplica."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Colunas ALVOS, em unidades originais, com o índice de X."""


class CandidatoPorAlvo(Candidato):
    """Um estimador por alvo, treinado em log(alvo) e previsto de volta com exp."""

    def __init__(self, semente: int = 0, **params) -> None:
        super().__init__(semente, **params)
        self.prep: Preprocessador
        self.estimadores: dict[str, object] = {}

    @abstractmethod
    def _novo_estimador(self, alvo: str, nomes: list[str]):
        """Objeto com fit(X, y) e predict(X) em numpy."""

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> Self:
        self.prep = Preprocessador().fit(X)
        Xt = self.prep.transform(X)
        self.estimadores = {}
        for alvo in ALVOS:
            presente = Y[alvo].notna().to_numpy()
            estimador = self._novo_estimador(alvo, self.prep.nomes)
            estimador.fit(Xt[presente], np.log(Y[alvo].to_numpy()[presente]))
            self.estimadores[alvo] = estimador
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self.prep.transform(X)
        previsoes = {alvo: np.exp(self.estimadores[alvo].predict(Xt)) for alvo in ALVOS}
        return pd.DataFrame(previsoes, index=X.index)
