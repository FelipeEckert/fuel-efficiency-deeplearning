"""Pré-processamento das entradas: one-hot das categóricas e padronização das numéricas."""

import numpy as np
import pandas as pd

from eficiencia.esquema import CATEGORICAS, NUMERICAS


class Preprocessador:
    def __init__(self) -> None:
        self.nomes: list[str] = []
        self._categorias: dict[str, list[str]] = {}
        self._media: pd.Series
        self._desvio: pd.Series

    @staticmethod
    def _numericas(X: pd.DataFrame) -> pd.DataFrame:
        N = X[NUMERICAS].astype(float)
        # vazio em CVT/DHT significa "sem marcha fixa"; bateria_estimada (bool) vira 0/1 pelo astype
        N["n_marchas"] = N["n_marchas"].fillna(0)
        return N

    def fit(self, X: pd.DataFrame) -> Preprocessador:
        self._categorias = {c: sorted(X[c].dropna().unique()) for c in CATEGORICAS}
        self.nomes = [f"{c}={v}" for c in CATEGORICAS for v in self._categorias[c]] + list(NUMERICAS)
        N = self._numericas(X)
        self._media = N.mean()
        # coluna constante ganha desvio 1 para não dividir por zero
        self._desvio = N.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        onehot = [X[c].to_numpy() == v for c in CATEGORICAS for v in self._categorias[c]]
        # NaN nas numéricas (poucas linhas) vira 0 após padronizar, ou seja, a média do fit
        Z = ((self._numericas(X) - self._media) / self._desvio).fillna(0.0)
        blocos = [np.column_stack(onehot).astype(np.float32), Z.to_numpy(np.float32)]
        return np.hstack(blocos)
