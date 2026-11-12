"""Checagem de monotonia física: MJ/km não cai e km/l não sobe quando uma entrada monotônica cresce."""

import numpy as np
import pandas as pd

from eficiencia.esquema import ALVOS, ENTRADAS, MONOTONAS

_TOLERANCIA = 1e-9


def checar_monotonia(
    modelo,
    df: pd.DataFrame,
    n_veiculos: int = 30,
    fatores: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
    semente: int = 0,
) -> list[str]:
    rng = np.random.default_rng(semente)
    base = df[ENTRADAS].iloc[rng.choice(len(df), size=min(n_veiculos, len(df)), replace=False)]
    mensagens: list[str] = []
    # a derivada não entra aqui: não dá para multiplicá-la na base
    for coluna in [c for c in MONOTONAS if c in ENTRADAS]:
        previsoes = []
        for fator in fatores:
            X = base.copy()
            X[coluna] = X[coluna] * fator
            previsoes.append(modelo.predict(X)[ALVOS].to_numpy())
        P = np.stack(previsoes)  # fatores x veiculos x alvos
        for j, alvo in enumerate(ALVOS):
            a, b = P[:-1, :, j], P[1:, :, j]
            if alvo.startswith("mj_km"):
                violou = (b < a - _TOLERANCIA * np.abs(a)).any(axis=0)
            else:
                violou = (b > a + _TOLERANCIA * np.abs(a)).any(axis=0)
            for linha in base.index[violou]:
                mensagens.append(f"{alvo} breaks monotonicity in {coluna} (vehicle at index {linha})")
    return mensagens
