import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.validacao import cv_aninhada


def test_xgboost_monotono_em_f2():
    df = carregar_base()
    m = CANDIDATOS["xgboost"](**CANDIDATOS["xgboost"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    base = df[ENTRADAS].iloc[[0]]
    grade = pd.concat([base.assign(f2_n_kmh2=v) for v in np.linspace(0.02, 0.12, 15)], ignore_index=True)
    p = m.predict(grade)
    assert (np.diff(p["mj_km_urbano"]) >= -1e-9).all()
    assert (np.diff(p["kml_gasolina_urbano"]) <= 1e-9).all()


@pytest.mark.parametrize("nome", ["floresta", "xgboost"])
def test_cv_rapida(nome):
    r = cv_aninhada(
        nome, carregar_base(), n_tentativas=2, storage=None, n_externas=2, n_internas=2, rapido=True
    )
    assert r.erro_mj < 0.12
