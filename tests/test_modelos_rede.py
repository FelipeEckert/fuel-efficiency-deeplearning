import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.validacao import cv_aninhada


def test_rede_multi_alvos_ausentes_nao_viram_nan():
    df = carregar_base()
    m = CANDIDATOS["rede_multi"](**CANDIDATOS["rede_multi"].PARAMS_RAPIDOS).fit(df[ENTRADAS], df[ALVOS])
    p = m.predict(df[ENTRADAS])
    assert np.isfinite(p.values).all() and (p.values > 0).all()


def test_rede_deterministica():
    df = carregar_base()
    a = (
        CANDIDATOS["rede_por_alvo"](semente=1, max_epocas=5)
        .fit(df[ENTRADAS], df[ALVOS])
        .predict(df[ENTRADAS])
    )
    b = (
        CANDIDATOS["rede_por_alvo"](semente=1, max_epocas=5)
        .fit(df[ENTRADAS], df[ALVOS])
        .predict(df[ENTRADAS])
    )
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("nome", ["rede_por_alvo", "rede_multi"])
def test_cv_rapida_rede(nome):
    r = cv_aninhada(
        nome, carregar_base(), n_tentativas=2, storage=None, n_externas=2, n_internas=2, rapido=True
    )
    assert np.isfinite(r.erro_mj)


def test_parte_monotona_cresce_e_nao_e_so_convexa():
    import torch

    from eficiencia.modelos.rede import _MlpPositivo

    torch.manual_seed(0)
    m = _MlpPositivo(1, 1, camadas=2, largura=16)
    for b in m.vies:
        torch.nn.init.normal_(b)  # espalha as dobras pela faixa
    x = torch.linspace(-3, 3, 601).reshape(-1, 1)
    with torch.no_grad():
        y = m(x)[:, 0].numpy()
    assert (np.diff(y) >= -1e-6).all()  # não decrescente
    # com só ReLU e pesos positivos, a segunda diferença nunca seria negativa
    assert (np.diff(y, 2) < -1e-6).any()
