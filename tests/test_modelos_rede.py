import numpy as np
import pandas as pd
import pytest

from eficiencia.esquema import ALVOS, CAMINHO_BASE, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.monotonia import checar_monotonia
from eficiencia.validacao import cv_aninhada

# golden values measured on the real PBEV table, which is not versioned
base_real = pytest.mark.skipif(not CAMINHO_BASE.exists(), reason="real PBEV table not available")


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


# previsões das 3 primeiras linhas antes do histórico, da validação por grupo e da média de redes
PADRAO = {
    "rede_por_alvo": [
        [2.2232745604, 1.6468255565, 9.9149208337, 12.1491517917, 8.0793950657, 9.4005544287],
        [2.2945809687, 1.6676228419, 9.8321468045, 12.1113929614, 8.0377896843, 9.3858611861],
        [2.424319867, 1.7904196115, 8.646044407, 10.8119120486, 7.8771679071, 9.1827350687],
    ],
    "rede_multi": [
        [2.3907770373, 1.7845244133, 9.8667164041, 11.9113203446, 7.8429261038, 9.0855631736],
        [2.2099901973, 1.7120526627, 10.0678499283, 12.0600186422, 7.858106884, 9.1301836129],
        [2.9300695144, 2.0832989281, 8.5516226493, 10.3561973093, 7.3011790216, 8.4406848881],
    ],
}


@base_real
@pytest.mark.parametrize("nome", ["rede_por_alvo", "rede_multi"])
def test_padroes_mantem_previsoes(nome):
    df = carregar_base()
    p = CANDIDATOS[nome](semente=1, max_epocas=5).fit(df[ENTRADAS], df[ALVOS]).predict(df[ENTRADAS])
    np.testing.assert_allclose(p.iloc[:3].values, PADRAO[nome], rtol=1e-6)


def test_validacao_por_grupo_sem_irmaos():
    from eficiencia.modelos.rede import _separar_validacao

    g = carregar_base()["grupo_cv"].to_numpy()
    val, tr = _separar_validacao(len(g), g, np.random.default_rng(0))
    assert not set(g[val]) & set(g[tr])
    assert 0.15 <= len(val) / len(g) < 0.30


def test_validacao_por_grupo_com_dois_grupos():
    from eficiencia.modelos.rede import _separar_validacao

    val, tr = _separar_validacao(5, np.array(list("aaabb")), np.random.default_rng(0))
    assert len(val) and len(tr)


def test_historico_e_media_de_redes():
    df = carregar_base()
    m = CANDIDATOS["rede_por_alvo"](max_epocas=5, n_redes=3).fit(df[ENTRADAS + ["grupo_cv"]], df[ALVOS])
    mlp = m.estimadores["mj_km_urbano"].mlp
    assert len(mlp.historicos) == 3 and all(len(h) == 5 for h in mlp.historicos)
    assert np.isfinite(m.predict(df[ENTRADAS]).values).all()
    assert checar_monotonia(m, df) == []


@pytest.mark.parametrize("com_grupo", [True, False])
def test_rede_por_alvo_passa_grupos_para_a_validacao(monkeypatch, com_grupo):
    from eficiencia.modelos import rede

    vistos = []
    original = rede._separar_validacao

    def espiao(n, grupos, rng):
        vistos.append(grupos)
        return original(n, grupos, rng)

    monkeypatch.setattr(rede, "_separar_validacao", espiao)
    df = carregar_base()
    X = df[ENTRADAS + ["grupo_cv"]] if com_grupo else df[ENTRADAS]
    CANDIDATOS["rede_por_alvo"](semente=1, max_epocas=2).fit(X, df[ALVOS])
    assert vistos and all((g is not None) == com_grupo for g in vistos)
