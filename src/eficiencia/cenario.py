"""Simulador "e se": suaviza a resposta em degraus dos modelos de árvore."""

import numpy as np
import pandas as pd

from eficiencia.esquema import ENTRADAS
from eficiencia.fisica import classe_inercia, combinar_mj

_DELTAS = np.linspace(-0.05, 0.05, 11)
_AJUSTADAS = ["massa_om_kg", "f0_n", "f2_n_kmh2", "potencia_kw"]


def _media(modelo, veiculos: list[dict]) -> np.ndarray:
    linhas = []
    for v in veiculos:
        for d in _DELTAS:
            linha = {**v, **{c: v[c] * (1 + d) for c in _AJUSTADAS}}
            linha["classe_inercia_kg"] = classe_inercia(linha["massa_om_kg"])
            linhas.append(linha)
    p = modelo.predict(pd.DataFrame(linhas)[ENTRADAS])
    mj = combinar_mj(p["mj_km_urbano"], p["mj_km_estrada"]).to_numpy()
    return mj.reshape(len(veiculos), len(_DELTAS)).mean(axis=1)


def fator_suave(modelo, original: dict, alterado: dict) -> float:
    """Razão alterado/original do MJ/km combinado, média sobre ±5 % conjunto de massa, F0, F2 e potência."""
    ori, alt = _media(modelo, [original, alterado])  # uma só chamada ao modelo
    return float(alt / ori)
