import numpy as np
import pandas as pd
import pytest

from eficiencia.cenario import fator_suave
from eficiencia.entrada import validar
from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.fisica import combinar_mj


def _validado(linha: pd.Series) -> dict:
    return validar(pd.DataFrame([linha[ENTRADAS].to_dict()]))[0].iloc[0].to_dict()


def _bruto(modelo, v: dict) -> float:
    p = modelo.predict(pd.DataFrame([v])[ENTRADAS])
    return float(combinar_mj(p["mj_km_urbano"], p["mj_km_estrada"]).iloc[0])


@pytest.fixture(scope="module")
def veiculos():
    base = carregar_base()
    return [_validado(base.iloc[i]) for i in (0, len(base) // 2, len(base) - 1)]


def test_sem_mudanca_fator_e_um(pacote_rapido, veiculos):
    for v in veiculos:
        assert fator_suave(pacote_rapido.modelo, v, v) == 1.0


def test_varredura_f2_monotona_e_sem_saltos_maiores(pacote_rapido, veiculos):
    m = pacote_rapido.modelo
    for v in veiculos:
        passos = np.linspace(-0.20, 0.20, 41)
        suave, bruto = [], []
        for d in passos:
            alt = {**v, "f2_n_kmh2": v["f2_n_kmh2"] * (1 + d)}
            suave.append(fator_suave(m, v, alt))
            bruto.append(_bruto(m, alt) / _bruto(m, v))
        assert np.all(np.diff(suave) >= -1e-12)
        assert np.diff(suave).max() <= np.diff(bruto).max() + 1e-12
