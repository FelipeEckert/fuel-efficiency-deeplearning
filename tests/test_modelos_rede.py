import numpy as np
import pandas as pd
import pytest

import eficiencia.preprocess
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
# medidas na base já com a correção RAV4 (Combustão com híbrido no nome vira Híbrido)
PADRAO = {
    "rede_por_alvo": [
        [2.2879595566, 1.6900448604, 9.943234785, 12.2942937772, 8.0629320837, 9.3869633004],
        [2.2257535, 1.6221510812, 9.8674501705, 12.1785101836, 7.9980993672, 9.3715367158],
        [2.496555174, 1.842069152, 8.6090853734, 10.847159729, 7.8548924741, 9.1726758791],
    ],
    "rede_multi": [
        [2.3028376769, 1.7431357991, 10.0343193986, 12.0249373867, 7.8504757128, 9.1055006332],
        [2.1978530423, 1.6901745406, 10.0938520589, 12.108827564, 7.8635314354, 9.1340300392],
        [2.853605268, 2.0520781128, 8.654408058, 10.373022306, 7.290885652, 8.4340736533],
    ],
}


@base_real
@pytest.mark.parametrize("nome", ["rede_por_alvo", "rede_multi"])
def test_padroes_mantem_previsoes(nome, monkeypatch):
    # os valores fixos foram medidos sem as entradas derivadas
    monkeypatch.setattr(eficiencia.preprocess, "DERIVADAS", [])
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
