"""Sinais de monotonia por coluna e alvo, compartilhados por todos os candidatos."""

import numpy as np

from eficiencia.esquema import MONOTONAS


def sinal_alvo(alvo: str) -> int:
    # consumo cresce com massa/F0/F2 (mj_km) e cai (km/l): sinal oposto por tipo de alvo
    return 1 if alvo.startswith("mj_km") else -1


def sinais_monotonia(nomes: list[str], alvo: str) -> np.ndarray:
    s = sinal_alvo(alvo)
    return np.array([s if n in MONOTONAS else 0 for n in nomes], dtype=int)
