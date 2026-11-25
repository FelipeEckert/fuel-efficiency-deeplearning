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
# medidas na base com a correção RAV4 e a entrada tipo_hibrido
PADRAO = {
    "rede_por_alvo": [
        [2.2933991385, 1.6761794897, 9.669270329, 12.018920502, 8.1009403091, 9.3707134373],
        [2.2654464506, 1.6233377406, 9.8699013277, 12.2119742778, 8.140812461, 9.4617544729],
        [2.5852554448, 1.866075057, 8.3342713941, 10.4163219993, 7.9339220307, 9.2009723623],
    ],
    "rede_multi": [
        [2.324183365, 1.7040661276, 9.8900570437, 11.8969354307, 7.7689393525, 9.3680957485],
        [2.3148292102, 1.6919446251, 9.8544716411, 11.9339283897, 7.7094189016, 9.302604822],
        [2.6830232969, 1.9252730566, 8.5261949389, 10.4826608502, 7.1279723601, 8.7427093771],
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
