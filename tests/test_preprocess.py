import numpy as np

import eficiencia.preprocess as pp
from eficiencia.esquema import ENTRADAS, carregar_base
from eficiencia.preprocess import Preprocessador


def test_preprocessador_categoria_nova_e_marchas_vazias():
    df = carregar_base()
    p = Preprocessador().fit(df[ENTRADAS])
    novo = df[ENTRADAS].iloc[[0]].copy()
    novo["categoria"] = "INEXISTENTE"
    novo["n_marchas"] = np.nan
    x = p.transform(novo)
    assert np.isfinite(x).all()
    assert all(x[0, i] == 0 for i, n in enumerate(p.nomes) if n.startswith("categoria="))


def test_fracao_entra_como_numerica():
    df = carregar_base()
    p = Preprocessador().fit(df[ENTRADAS])
    assert p.nomes[-1] == "fracao_combustao"
    assert p.transform(df[ENTRADAS]).shape[1] == len(p.nomes)


def test_plugin_sem_autonomia_nao_quebra():
    df = carregar_base()
    X = df[ENTRADAS].copy()
    linha = X.index[X["propulsao"].eq("Plug-in")][0]
    X.loc[linha, "autonomia_eletrica_km"] = float("nan")
    Z = Preprocessador().fit(df[ENTRADAS]).transform(X)
    assert np.isfinite(Z).all()


def test_preprocessador_antigo_mantem_colunas(monkeypatch):
    df = carregar_base()
    monkeypatch.setattr(pp, "DERIVADAS", [])
    antigo = Preprocessador().fit(df[ENTRADAS])  # como o pacote salvo antes desta branch
    monkeypatch.setattr(pp, "DERIVADAS", ["fracao_combustao"])
    assert antigo.transform(df[ENTRADAS]).shape[1] == len(antigo.nomes)
